# AnomalyCLIP_lib/AnomalyCLIP.py
from collections import OrderedDict
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from torch import nn


# --------------------------- 基础组件 ---------------------------

class LayerNorm(nn.LayerNorm):
    """fp16 友好 LayerNorm"""
    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        y = super().forward(x.float())
        return y.to(orig_dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# --------------------------- 注意力（含双路径的原版接口保持不变） ---------------------------

class Attention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 原始自注意力
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        # “替换为 v-v”的分支
        k = v
        q = k
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]


# --------------------------- Transformer Block（接入 FiLM 调制） ---------------------------

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def _apply_film(self, x: torch.Tensor, film_params: Optional[torch.Tensor], gate: Optional[torch.Tensor]):
        """
        x: [L, N, C]
        film_params: [2*C] or [L, 2*C] or [1, 2*C]，拆成 gamma,beta
        gate: 可学习门控标量，若为 None 则恒等于 1
        """
        if film_params is None:
            return x
        C = x.shape[-1]
        if film_params.dim() == 1:
            # [2*C]
            gamma, beta = film_params[:C], film_params[C:]
            gamma = gamma.view(1, 1, C).to(x.dtype).to(x.device)
            beta = beta.view(1, 1, C).to(x.dtype).to(x.device)
        elif film_params.dim() == 2:
            # [L, 2*C]
            assert film_params.size(1) == 2 * C, "film_params second dim must be 2*C"
            gamma, beta = film_params[:, :C], film_params[:, C:]
            gamma = gamma.view(-1, 1, C).to(x.dtype).to(x.device)
            beta = beta.view(-1, 1, C).to(x.dtype).to(x.device)
        else:
            raise ValueError("film_params must be 1D or 2D")

        g = 1.0 if gate is None else gate.to(x.dtype).to(x.device)
        # FiLM 调制（用 1+gamma 保留恒等支路，避免塌缩）
        x = x * (1 + g * gamma) + g * beta
        return x

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, ffn: bool = False,
                film_params: Optional[torch.Tensor] = None,
                film_gate: Optional[torch.Tensor] = None,
                attn_bias: Optional[torch.Tensor] = None):
        # x: [L, N, C]
        # 在注意力前的规范化输出上做 FiLM
        x_res = self.ln_1(x)
        x_res = self._apply_film(x_res, film_params, film_gate)  # <<<< 深层调制

        # 注意力
        x = x + self.attention(x_res)

        # FFN
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor,
                out_layers: List[int] = [6, 12, 18, 24],
                DPAM_layer: Optional[int] = None,
                ffn: bool = False,
                deep_mod: Optional[dict] = None):
        """
        x: [L, N, C]
        deep_mod: 可选 dict，支持：
            - "film_params": 长度为 layers 的列表，每层一个 [2*C] 或 [L, 2*C] 的张量；或单个共享张量
            - "film_gate":   可选标量 gate (tensor scalar)，或每层一个 gate
            - "attn_bias":   保留接口（本实现未使用）
        """
        out_tokens = []
        # 解析 deep_mod
        film_params_all = None
        film_gate_all = None
        if deep_mod is not None:
            film_params_all = deep_mod.get("film_params", None)
            film_gate_all = deep_mod.get("film_gate", None)

        for i, r in enumerate(self.resblocks, start=1):
            # 取第 i 层的 FiLM
            film_i = None
            gate_i = None
            if film_params_all is not None:
                if isinstance(film_params_all, (list, tuple)):
                    film_i = film_params_all[i - 1] if i - 1 < len(film_params_all) else None
                else:
                    film_i = film_params_all  # 所有层共享
            if film_gate_all is not None:
                if isinstance(film_gate_all, (list, tuple)):
                    gate_i = film_gate_all[i - 1] if i - 1 < len(film_gate_all) else None
                else:
                    gate_i = film_gate_all

            x = r(x, ffn=ffn, film_params=film_i, film_gate=gate_i)

            if i in out_layers:
                out_tokens.append(x.clone())

        return x, out_tokens

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype


# --------------------------- 视觉 Transformer（插 FiLM 投影层） ---------------------------

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.width = width

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, attn_mask=None)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # ------ FiLM: 用 768 维的 SwAV prototype（ViT-L text/patch dim）投影到 2*width（gamma+beta） ------
        self.film_proj = nn.Linear(768, 2 * width)
        nn.init.xavier_uniform_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

        # 全局门控（可学习）
        self.film_gate = nn.Parameter(torch.tensor(0.5))

    @torch.no_grad()
    def DAPM_replace(self, DPAM_layer):
        # 保留接口，不在这里更改注意力实现
        return

    @torch.no_grad()
    def forward(self, x: torch.Tensor, features_list, ori_patch=False, proj_use=True,
                DPAM_layer=None, ffn=False, deep_mod: Optional[dict] = None):
        """
        deep_mod 可传入：
          - "film_source": (K, 768) 或 (768,) SwAV/投影后的原型（未投影到 2*width）
          - 或直接传 "film_params": (2*width,) / (L, 2*width)，则不再使用 film_proj
          - "film_gate": 可选标量
        """
        B = x.shape[0]
        x = self.conv1(x)                       # [B, C=width, H/P, W/P]
        x = x.reshape(B, self.width, -1).permute(0, 2, 1)   # [B, N, C]
        cls = self.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(1).repeat(B, 1, 1)  # [B,1,C]
        x = torch.cat([cls, x], dim=1)          # [B, N+1, C]

        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)
        if side != new_side:
            # 动态 resize 位置编码
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, self.width).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(-1, self.width, new_side * new_side).transpose(1, 2)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos[0]], 0)

        pos = self.positional_embedding.to(x.dtype)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # [L, B, C]

        # --------- 组装 deep_mod -> film_params ----------
        film_params = None
        film_gate = None
        if deep_mod is not None:
            if "film_params" in deep_mod and deep_mod["film_params"] is not None:
                fp = deep_mod["film_params"]  # 可能是 [2*C] 或 [L,2*C]
                film_params = fp.to(x.dtype).to(x.device)
            elif "film_source" in deep_mod and deep_mod["film_source"] is not None:
                src = deep_mod["film_source"]  # [K,768] 或 [768]
                if src.dim() == 2:
                    src = src.mean(dim=0)      # 聚合到 [768]
                # 投影到 2*width
                film_params = self.film_proj(src.to(x.device).to(self.film_proj.weight.dtype)).to(x.dtype)
            if "film_gate" in deep_mod and deep_mod["film_gate"] is not None:
                film_gate = deep_mod["film_gate"]
            else:
                film_gate = self.film_gate

        x, patch_tokens = self.transformer(
            x, out_layers=features_list, DPAM_layer=DPAM_layer, ffn=ffn,
            deep_mod={"film_params": film_params, "film_gate": film_gate}
        )
        # x: [L, B, C]

        # 收集 patch token 特征并做线性投影
        patch_token_list = []
        for pt in patch_tokens:
            pt = self.ln_post(pt.permute(1, 0, 2)) @ self.proj  # [B, L, C] -> [B, L, out_dim]
            patch_token_list.append(pt)
        patch_tokens = patch_token_list

        # CLS 的视觉全局特征
        img_feat = (x[0].to(self.ln_post.weight.dtype))  # [B, C]
        img_feat = self.ln_post(img_feat) @ self.proj    # [B, out_dim]

        return img_feat, patch_tokens


# --------------------------- 整体 CLIP 框架 ---------------------------

class AnomalyCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details=None):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,        # ViT-L: 1024
            layers=vision_layers,      # ViT-L: 24
            heads=vision_heads,
            output_dim=embed_dim       # 768
        )

        self.transformer = Transformer(
            width=transformer_width,   # 768 for text tower
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            if isinstance(block.attn, nn.MultiheadAttention):
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp[0].weight, std=fc_std)   # c_fc
            nn.init.normal_(block.mlp[2].weight, std=proj_std) # c_proj
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    # ---- 给视觉塔加 deep modulation 的新接口 ----
    def encode_image(self, image, feature_list=[], ori_patch=False, proj_use=True, DPAM_layer=None, ffn=False, deep_mod: Optional[dict] = None):
        return self.visual(image.type(self.dtype), feature_list, ori_patch=ori_patch, proj_use=proj_use, DPAM_layer=DPAM_layer, ffn=ffn, deep_mod=deep_mod)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [B, L, C]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)                          # [L, B, C]
        x, _ = self.transformer(x)
        x = x.permute(1, 0, 2)                          # [B, L, C]
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_text_learn(self, prompts, tokenized_prompts, deep_compound_prompts_text=None, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = prompts + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x, _ = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)[0]
        text_features = self.encode_text(text)
        image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-6)
        text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-6)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
