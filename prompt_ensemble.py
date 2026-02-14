import math
import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from open_clip import tokenizer
# simple_tokenizer = tokenizer.SimpleTokenizer()
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def encode_text_with_prompt_ensemble(model, texts, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = [
        'a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.',
        'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.',
        'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.',
        'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
        'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.',
        'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.',
        'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.',
        'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
        'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
        'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
        'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
        'this is the {} in the scene.', 'this is one {} in the scene.'
    ]

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(texts[0]) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence)
        class_embeddings = model.encode_text(prompted_sentence.to(device))
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device).t()

    return text_features


def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

class AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details, n_prompts=8,num_layers=1): # M=8
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.M = n_prompts 
        self.n_ctx = design_details["Prompt_length"]
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]
        dtype = clip_model.transformer.get_cast_dtype()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.num_layers = int(num_layers)
        self.state_normal_list = ["{}"]
        self.state_anomaly_list = ["damaged {}"]

        # 初始化 M 组独立的上下文向量
        # 维度: [M, n_ctx, ctx_dim]
        ctx_vectors_pos = torch.empty(self.M, self.n_ctx, ctx_dim, dtype=dtype)
        ctx_vectors_neg = torch.empty(self.M, self.n_ctx, ctx_dim, dtype=dtype)
        print('ctx_vectors_neg .shape',ctx_vectors_neg .shape)
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)
        
        self.ctx_pos = nn.Parameter(ctx_vectors_pos)
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)

        # 预制文本的前缀和后缀
        prompt_prefix = " ".join(["X"] * self.n_ctx)
        p_pos = prompt_prefix + " " + self.state_normal_list[0].format("object") + "."
        p_neg = prompt_prefix + " " + self.state_anomaly_list[0].format("object") + "."
        
        tokenized_pos = tokenize(p_pos)
        tokenized_neg = tokenize(p_neg)

        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_neg).type(dtype)

        # 这里的 token_prefix/suffix 需要适配 M 组
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :]) # [1, 1, D]
        self.register_buffer("token_suffix_pos", embedding_pos[:, 1 + self.n_ctx:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + self.n_ctx:, :])
        
        self.register_buffer("tokenized_prompts_pos", tokenized_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_neg)

        # 深层提示词
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim, dtype=dtype)) # 深层也要适配 M
            for _ in range(self.compound_prompts_depth - 1)
        ])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        # === NEW: prompt-wise gates ===
        # 初始化为小的正态分布，以确保 sigmoid 激活值在0和1之间
        # === NEW: token-wise gates (M, n_ctx, 1) ===
        #gate_std = float(design_details.get("gate_init_std", 0.2))  # 默认显著大于0.02

        # cosine-attn 的温度（用 clip_temp 初始化）
        scale_init = 1.0 / 0.07  # 14.2857
        self.logit_scale_pos = nn.Parameter(torch.tensor(math.log(scale_init)))
        self.logit_scale_neg = nn.Parameter(torch.tensor(math.log(scale_init)))
        
        # 可选：让 query 更稳定（不改变你整体结构）
        self.q_ln_pos = nn.LayerNorm(ctx_dim)
        self.q_ln_neg = nn.LayerNorm(ctx_dim)
        
        # 可选：value 投影（很推荐，且不丢幅值信息）
        self.v_proj_pos = nn.Linear(ctx_dim, ctx_dim, bias=False)
        self.v_proj_neg = nn.Linear(ctx_dim, ctx_dim, bias=False)
        nn.init.zeros_(self.v_proj_pos.weight)  # 初始不影响（更稳）
        nn.init.zeros_(self.v_proj_neg.weight)
        



    def forward(self, normal_items=None, abnormal_items=None):
        # items: [K, D] -> [1, K, D]
        feat = normal_items if normal_items is not None else abnormal_items
        feat = feat.unsqueeze(0) 
        
        ctx = self.ctx_pos if normal_items is not None else self.ctx_neg
        prefix = self.token_prefix_pos if normal_items is not None else self.token_prefix_neg
        suffix = self.token_suffix_pos if normal_items is not None else self.token_suffix_neg
        tokenized = self.tokenized_prompts_pos if normal_items is not None else self.tokenized_prompts_neg

        # Cross-Attention: [M, L, D] 与 [1, K, D] 交互
        # 结果 ctx_mod: [M, L, D]
        feat_raw = feat  # [1,K,D] 或 [K,D]（你上面 unsqueeze 后是 [1,K,D]）
        if feat_raw.dim() == 2:
            feat_raw = feat_raw.unsqueeze(0)
        
        ctx_fp = ctx.float()           # [M,L,D]
        feat_fp = feat_raw.float()     # [1,K,D]
        
        # 1) cosine attention：q、k 都 L2 normalize（只用于打分）
        q_ln = self.q_ln_pos if normal_items is not None else self.q_ln_neg
        q = F.normalize(q_ln(ctx_fp), dim=-1)                # [M,L,D]
        k = F.normalize(feat_fp, dim=-1)                     # [1,K,D]
        
        # scale：用 clip_temp 初始化（可学习）
        logit_scale = (self.logit_scale_pos if normal_items is not None else self.logit_scale_neg).exp().clamp(1.0, 100.0)
        # 如果你想固定 scale=1/0.07：直接写 logit_scale = 1.0/0.07
        
        # -------------------------
        # Grouped (per-layer) attention
        # -------------------------
        K_total = feat_fp.size(1)              # 总 proto 数 = L * K
        L_proto = max(1, int(self.num_layers))
        
        # 如果不能整除，就退回原版（保险）
        if K_total % L_proto != 0:
            print('---------------------error total proto---------------------')
            logits = torch.matmul(q, k.transpose(-1, -2)) * logit_scale  # [M, Lctx, K_total]
            attn = torch.softmax(logits, dim=-1)
        
            # gate：建议你即使退回也用 conf gate（更稳）
            conf = attn.max(dim=-1).values  # [M, Lctx]
            conf_u = 1.0 / attn.size(-1)
            eps = 1e-6
            gate = ((conf - conf_u).clamp(min=0) / (1 - conf_u + eps)).sqrt().unsqueeze(-1)  # [M, Lctx, 1]
        
            interaction = torch.matmul(attn, feat_fp)  # [M, Lctx, D]
        else:
            K = K_total // L_proto                     # 每层 proto 数
        
            # reshape 成 [L, K, D]
            feat_g = feat_fp.view(1, L_proto, K, -1)                 # [1, L, K, D]
            k_g = F.normalize(feat_g, dim=-1).squeeze(0)             # [L, K, D]
            v_g = feat_g.squeeze(0)                                  # [L, K, D]  (raw values)
        
            # logits: [M, Lctx, L, K]
            logits_g = torch.einsum("mtd,lkd->mtlk", q, k_g) * logit_scale
            attn_g = torch.softmax(logits_g, dim=-1)                 # softmax 只在每层的 K 上
        
            # interaction per layer: [M, Lctx, L, D]
            interaction_g = torch.einsum("mtlk,lkd->mtld", attn_g, v_g)
        
            # 聚合各层（先用 mean，数值最稳；想更强可改 sum/L 或 sum）
            interaction = interaction_g.mean(dim=2)                  # [M, Lctx, D]
        
            # -------------------------
            # gate: 用 conf 而不是 entropy（参数0，更不容易“关交互”）
            # -------------------------
            conf = attn_g.max(dim=-1).values.mean(dim=2)             # [M, Lctx] 先每层取max，再对层平均
            conf_u = 1.0 / K
            eps = 1e-6
            gate = ((conf - conf_u).clamp(min=0) / (1 - conf_u + eps)).sqrt().unsqueeze(-1)  # [M, Lctx, 1]
        
            # debug：看看到底有没有变尖锐
            if torch.rand(1).item() < 0.01:
                print("attn_max_mean(per-layer):", float(conf.mean().detach().cpu()),
                      "1/K:", 1.0 / K,
                      "gate_mean:", float(gate.mean().detach().cpu()),
                      "scale:", float(logit_scale.detach().cpu()))

        
        # 4) 可选：value 投影（推荐），并用残差方式更新 ctx
        v_proj = self.v_proj_pos if normal_items is not None else self.v_proj_neg
        interaction = v_proj(interaction).to(ctx.dtype)
        
        # 注意：你之前那句 F.normalize(interaction)*ctx.norm 会把幅值又抹掉
        # 这里不要再 normalize interaction，才能保留“原型幅值信息”
        ctx_mod = ctx + gate.to(ctx.dtype) * interaction



        # 拼接出 M 条完整的 Prompts: [M, 77, D]
        # prefix/suffix 需要广播到 M
        prompts = torch.cat([prefix.expand(self.M, -1, -1), ctx_mod, suffix.expand(self.M, -1, -1)], dim=1)
        
        # tokenized 也需要扩展到 M
        tokenized_M = tokenized.expand(self.M, -1)
        #if torch.rand(1).item() < 0.01:
        #    print("attn_max_mean:", attn.max(dim=-1).values.mean().item(), "1/K:", 1.0/attn.size(-1), "scale:", float(logit_scale.detach().cpu()))

        return prompts, tokenized_M, self.compound_prompts_text