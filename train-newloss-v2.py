import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
import torch.nn as nn
import itertools
from dataset import Dataset
import math
def grad_norm(loss, params):
    # 只保留需要梯度的参数
    params = [p for p in params if p is not None and getattr(p, "requires_grad", False)]
    # warmup 时 params 可能为空；或者 loss 本身不在图上
    if len(params) == 0 or (not getattr(loss, "requires_grad", False)):
        return 0.0

    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    s = 0.0
    for g in grads:
        if g is None:
            continue
        s += (g.detach() ** 2).sum()
    return float(torch.sqrt(s + 1e-12).cpu())


def get_optimizer(args, model_components):
    """
    辅助函数：创建优化器，并精细化控制 Weight Decay
    Group 0: Prompt Tokens / Biases (WD=0) - 语义敏感，不衰减
    Group 1: Projection Weights (WD=1e-4) - 防止 MLP 过拟合
    Group 2: Prototypes (WD=0) - 或者是极小值
    """
    (prompt_learner, identity_proj, deviation_proj, swav_list, swav_abn_list) = model_components

    # --- 1. Prompt Group (分为 Tokens 和 Weights) ---
    prompt_params_nowd = []  # WD = 0
    prompt_params_wd = []    # WD > 0

    # A) Prompt Learner 全是 Embedding，不加 WD
    for name, param in prompt_learner.named_parameters():
        if param.requires_grad:
            prompt_params_nowd.append(param)

    # B) Projections: Weight 加 WD，Bias/Norm 不加
    for proj in [identity_proj, deviation_proj]:
        for name, param in proj.named_parameters():
            if not param.requires_grad:
                continue
            if 'weight' in name and len(param.shape) > 1:
                prompt_params_wd.append(param)
            else:
                prompt_params_nowd.append(param)

    # --- 2. Proto Group (Prototypes & Layer Tokens) ---
    proto_params = []
    for module in [swav_list, swav_abn_list]:
        for name, param in module.named_parameters():
            if param.requires_grad:
                proto_params.append(param)

    # --- 3. 创建优化器 ---
    optimizer = torch.optim.AdamW([
        # Index 0: Prompt 无衰减组 (Tokens)
        {'params': prompt_params_nowd, 'lr': args.learning_rate, 'weight_decay': 0.0},
        # Index 1: Prompt 有衰减组 (MLP Weights)
        {'params': prompt_params_wd, 'lr': args.learning_rate, 'weight_decay': 1e-4},
        # Index 2: Proto 组
        {'params': proto_params, 'lr': args.learning_rate, 'weight_decay': 0.0}
    ], lr=args.learning_rate, betas=(0.9, 0.999))

    return optimizer


def cross_sep_loss(Pn, Pa, topk=10, margin=0.3):
    # Pn: [K_n,D], Pa: [K_a,D] (已归一化最好)
    sim = Pn @ Pa.t()  # [K_n,K_a]
    vals, _ = torch.topk(sim.reshape(-1), k=min(topk, sim.numel()))
    return F.relu(vals - margin).pow(2).mean()


def set_requires_grad(x, flag: bool):
    if isinstance(x, (torch.nn.Parameter, torch.Tensor)):
        x.requires_grad_(flag)
        return
    for p in x.parameters():
        p.requires_grad_(flag)


# =================== 1. 辅助工具函数 ===================
def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def proto_orth_loss(keys: torch.Tensor) -> torch.Tensor:
    """原型正交损失：保证不同语义原型尽可能区分开"""
    M, D = keys.shape
    if M <= 1:
        return keys.new_tensor(0.0)
    K = F.normalize(keys, dim=1)
    G = K @ K.t()
    I = torch.eye(M, device=K.device, dtype=K.dtype)
    off = G - I
    return (off.pow(2).sum()) / (M * (M - 1) + 1e-6)


def binarize_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    兼容 mask 为 0/1 或 0/255 或 float。
    返回 float mask in {0,1}，shape 不变。
    """
    mask = mask.float()
    mx = float(mask.max().detach().item()) if mask.numel() > 0 else 1.0
    thr = 0.5 if mx <= 1.0 + 1e-3 else 127.5
    return (mask > thr).float()


def subsample_rows(x: torch.Tensor, max_samples: int) -> torch.Tensor:
    """随机下采样行数，控制原型学习的计算量"""
    if max_samples <= 0 or x.numel() == 0:
        return x
    n = x.shape[0]
    if n <= max_samples:
        return x
    idx = torch.randperm(n, device=x.device)[:max_samples]
    return x[idx]


@torch.no_grad()
def count_patch_by_maskratio(dataloader, device="cuda", thr_n=0.05, thr_a=0.30, ph=37):
    total_patch = 0
    normal_patch = 0
    abnormal_patch = 0
    boundary_patch = 0  # thr_n <= r <= thr_a

    for items in tqdm(dataloader, desc="Counting patches (mask_ratio)"):
        mask = items["img_mask"].to(device).float()  # [B,1,H,W]
        B = mask.shape[0]

        mask_ratio = F.interpolate(mask, size=(ph, ph), mode="area").flatten(1)  # [B, ph*ph]
        N = mask_ratio.shape[1]

        total_patch += B * N

        n = (mask_ratio < thr_n).sum().item()
        a = (mask_ratio > thr_a).sum().item()
        b = (N * B) - n - a  # boundary

        normal_patch += n
        abnormal_patch += a
        boundary_patch += b

    rn = normal_patch / max(1, total_patch)
    ra = abnormal_patch / max(1, total_patch)
    rb = boundary_patch / max(1, total_patch)

    print("\n===== Patch Counting Result (mask_ratio rule) =====")
    print(f"Total patches:      {total_patch}")
    print(f"Normal patches:     {normal_patch}   (mask_ratio < {thr_n})")
    print(f"Abnormal patches:   {abnormal_patch} (mask_ratio > {thr_a})")
    print(f"Boundary patches:   {boundary_patch} (in [{thr_n}, {thr_a}])")
    print("--------------------------------------------------")
    print(f"Normal ratio:   {rn:.6f}")
    print(f"Abnormal ratio: {ra:.6f}")
    print(f"Boundary ratio: {rb:.6f}")
    denom = max(1, normal_patch + abnormal_patch)
    print("--------------------------------------------------")
    print(f"Normal / (Normal+Abnormal):   {normal_patch/denom:.6f}")
    print(f"Abnormal / (Normal+Abnormal): {abnormal_patch/denom:.6f}")

    return {
        "total_patch": total_patch,
        "normal_patch": normal_patch,
        "abnormal_patch": abnormal_patch,
        "boundary_patch": boundary_patch,
        "ratio_normal_all": rn,
        "ratio_abnormal_all": ra,
        "ratio_boundary_all": rb,
        "ratio_normal_in_selected": normal_patch/denom,
        "ratio_abnormal_in_selected": abnormal_patch/denom,
    }


# =================== 2. 模型组件 ===================
class ResidualPerFeatureMLP(nn.Module):
    """带残差连接的投影层，确保 CLIP 原始语义不丢失"""
    def __init__(self, d_model=768, expansion=2, dropout=0.1):
        super().__init__()
        hidden = expansion * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        residual = x
        x = self.ln1(x)
        x = self.ffn(x)
        return self.ln2(x + residual)


class VMFMixturePrototypes(nn.Module):
    """
    路线1：vMF Mixture + 变分后验 q(z|x) + 负ELBO loss
    """
    def __init__(
        self,
        num_prototypes: int,
        dim: int = 768,
        kappa: float = 30.0,
        tau_q: float = 1.0,
        learn_pi: bool = True,
        usage_reg: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.K = int(num_prototypes)
        self.dim = int(dim)
        self.kappa = float(kappa)
        self.tau_q = float(tau_q)
        self.learn_pi = bool(learn_pi)
        self.usage_reg = float(usage_reg)
        self.eps = float(eps)

        proto = torch.randn(self.K, self.dim)
        self.prototypes = nn.Parameter(F.normalize(proto, dim=1))

        if learn_pi:
            self.logits_pi = nn.Parameter(torch.zeros(self.K))
        else:
            self.register_buffer("logits_pi", torch.zeros(self.K))

    def forward(self, feats: torch.Tensor):
        if feats.numel() == 0:
            P = F.normalize(self.prototypes, dim=1)
            return feats.new_tensor(0.0), P

        x = F.normalize(feats, dim=1)            # [N, D]
        P = F.normalize(self.prototypes, dim=1)  # [K, D]

        log_pi = F.log_softmax(self.logits_pi, dim=0)  # [K]
        e = self.kappa * (x @ P.t()) + log_pi.unsqueeze(0)  # [N, K]

        log_q = F.log_softmax(e / self.tau_q, dim=1)
        q = log_q.exp()
        elbo = (q * (e - log_q)).sum(dim=1)
        loss = -elbo.mean()

        if self.usage_reg > 0:
            mean_q = q.mean(dim=0)  # [K]
            uniform = torch.full_like(mean_q, 1.0 / self.K)
            kl = (mean_q * (torch.log(mean_q + self.eps) - torch.log(uniform + self.eps))).sum()
            loss = loss + self.usage_reg * kl

        return loss, P


def train(args):
    os.makedirs(args.save_path, exist_ok=True)
    final_log_path = os.path.join(args.save_path, "total_test_results.log")
    logger = get_logger(args.save_path)
    logger.info(f"Arguments: {args}")

    with open(final_log_path, "a") as f:
        f.write(f"\n\n{'='*20} New Training Session (Optimized Loss Edition - Fixed): {args.save_path} {'='*20}\n")

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    params = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    # 1) 加载核心模型
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()

    # 2) 初始化可学习组件
    num_layers = len(args.features_list)

    swav_list = nn.ModuleList([
        VMFMixturePrototypes(
            num_prototypes=args.num_prototypes,
            dim=768,
            kappa=args.vmf_kappa_n,
            tau_q=args.vmf_tau_q_n,
            learn_pi=True,
            usage_reg=args.vmf_usage_reg_n,
        ).to(device) for _ in range(num_layers)
    ])

    swav_abn_list = nn.ModuleList([
        VMFMixturePrototypes(
            num_prototypes=max(1, args.num_prototypes // 4),
            dim=768,
            kappa=args.vmf_kappa_a,
            tau_q=args.vmf_tau_q_a,
            learn_pi=True,
            usage_reg=args.vmf_usage_reg_a,
        ).to(device) for _ in range(num_layers)
    ])

    identity_proj = ResidualPerFeatureMLP().to(device)
    deviation_proj = ResidualPerFeatureMLP().to(device)

    # 保持你原来的写法（不改其它逻辑）
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params, n_prompts=args.prompt_num, num_layers=num_layers)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=None)

    # 3) 数据与优化器
    #train_data = Dataset(root=args.train_data_path, transform=preprocess,target_transform=target_transform,dataset_name=args.dataset)
    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)

    
    train_aug_rate = 0.2 if args.dataset == "mvtec" and args.aug_rate < 0 else max(args.aug_rate, 0.0)
    train_data = Dataset(
        root=args.train_data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
        training=True,
        aug_rate=train_aug_rate,
    )
    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True if device == "cuda" else False,drop_last=True)
    model_components = (prompt_learner, identity_proj, deviation_proj, swav_list, swav_abn_list)
    optimizer = get_optimizer(args, model_components)

    logger.info("Training started with Optimized Loss Strategy (Fixed).")

    # ====== Loss超参（不强制你在 argparse 里加，用 getattr 保底）======
    thr_n = 0.05
    thr_a = 0.3

    bg_weight = float(getattr(args, "bg_loss_weight", 0.3))
    mil_img_weight = float(getattr(args, "mil_img_loss_weight", 0.1))
    bg_topk_ratio = float(getattr(args, "bg_topk_ratio", 0.20))        # 正常图 top 10% patches
    #mil_img_weight = float(getattr(args, "mil_img_weight", 0.1))        # MIL 权重
    mil_topk_ratio = float(getattr(args, "mil_topk_ratio", 0.05))       # top 5% patches
    proto_grad_scale = float(getattr(args, "proto_grad_scale", 0.1))    # 原型梯度缩放

    logger.info(
        f"[LossOpt] thr_n={thr_n}, thr_a={thr_a} | "
        f"bg_weight={bg_weight}, bg_topk_ratio={bg_topk_ratio} | "
        f"mil_img_weight={mil_img_weight}, mil_topk_ratio={mil_topk_ratio} | "
        f"proto_grad_scale={proto_grad_scale}"
    )

    # Patch 统计（ph=37，与你的实际一致）
    count_patch_by_maskratio(train_dataloader, device=device, thr_n=thr_n, thr_a=thr_a, ph=37)

    for epoch in range(args.epoch):
        # warmup 逻辑保持不变
        if epoch < args.warmup_epoch:
            optimizer.param_groups[0]['lr'] = 0.0
            optimizer.param_groups[1]['lr'] = 0.0
            optimizer.param_groups[2]['lr'] = args.learning_rate
            phase_msg = "Warm-up Phase (Training Prototypes Only)"

            set_requires_grad(prompt_learner, False)
            set_requires_grad(identity_proj, False)
            set_requires_grad(deviation_proj, False)

            prompt_learner.eval()
            identity_proj.eval()
            deviation_proj.eval()

        elif args.warmup_epoch <= epoch < args.warmup_epoch + 3:
            set_requires_grad(prompt_learner, True)
            set_requires_grad(identity_proj, True)
            set_requires_grad(deviation_proj, True)

            prompt_learner.train()
            identity_proj.train()
            deviation_proj.train()

            current_prompt_lr = args.learning_rate * 0.2
            optimizer.param_groups[0]['lr'] = current_prompt_lr
            optimizer.param_groups[1]['lr'] = current_prompt_lr
            optimizer.param_groups[2]['lr'] = args.learning_rate * 0.1
            phase_msg = "Joint Training Phase (Prompt Full + Proto Fine-tune)"
        else:
            set_requires_grad(prompt_learner, True)
            set_requires_grad(identity_proj, True)
            set_requires_grad(deviation_proj, True)

            identity_proj.train()
            deviation_proj.train()

            current_prompt_lr = args.learning_rate * 0.5
            if epoch >= 25:
                current_prompt_lr *= 0.5
            optimizer.param_groups[0]['lr'] = current_prompt_lr
            optimizer.param_groups[1]['lr'] = current_prompt_lr
            optimizer.param_groups[2]['lr'] = args.learning_rate * 0.1
            phase_msg = "Joint Training Phase (Prompt Full + Proto Fine-tune)"

        logger.info(f"Epoch {epoch+1} Strategy: {phase_msg} | "
                    f"PromptLR={optimizer.param_groups[0]['lr']} | ProtoLR={optimizer.param_groups[2]['lr']}")

        if epoch >= args.warmup_epoch:
            prompt_learner.train()
            identity_proj.train()
            deviation_proj.train()

        # 记录项（扩展日志用）
        epoch_losses = {k: [] for k in [
            'total', 'img', 'img_ce', 'img_mil',
            'pixel', 'proto', 'orth', 'cross',
            'bce_pixel', 'dice_abn', 'bg_topk', 'bg_mean'
        ]}

        # ===== 关键统计（epoch级）=====
        stat_total_img = 0
        stat_pos_img_by_mask = 0
        stat_pos_img_by_label = 0
        stat_mismatch_pos_label_emptymask = 0
        stat_mismatch_neg_label_nonemptymask = 0

        stat_total_patches = 0
        stat_valid_patches = 0
        stat_pos_patches = 0
        stat_nor_patches = 0
        stat_boundary_patches = 0

        stat_dice_used_img = 0

        stat_mil_pos_sum = 0.0
        stat_mil_pos_cnt = 0
        stat_mil_neg_sum = 0.0
        stat_mil_neg_cnt = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epoch}")

        for items in pbar:
            image = items['img'].to(device)
            label = items['anomaly'].to(device).view(-1).long()  # [B]
            mask = items.get('img_mask', None).to(device)
            mask = binarize_mask(mask)  # [B,1,H,W]

            # --- A) 视觉特征提取 ---
            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=None)

            # --- B) Patch GT（忽略边界） ---
            B = label.shape[0]
            N = patch_features[0].shape[1] - 1
            ph = int(np.sqrt(N))
            assert ph * ph == N, "Patch number N must be a perfect square."

            mask_ratio = F.interpolate(mask, size=(ph, ph), mode='area').flatten(1)  # [B,N] in [0,1]
            gt_abn = (mask_ratio > thr_a).float()   # 确信异常
            gt_nor = (mask_ratio < thr_n).float()   # 确信正常
            valid_mask = (gt_abn + gt_nor) > 0      # 非边界区域

            # === 问题2修正：异常图判定用像素级 mask 是否非空（不依赖 thr_a）===
            mask_float = mask.squeeze(1).float()  # [B,H,W]
            has_anomaly_img = (mask_float.flatten(1).sum(1) > 0)  # [B]
            is_pos_label = (label == 1)

            # epoch统计：label/mask 是否一致
            stat_total_img += B
            stat_pos_img_by_mask += int(has_anomaly_img.sum().item())
            stat_pos_img_by_label += int(is_pos_label.sum().item())
            stat_mismatch_pos_label_emptymask += int((is_pos_label & (~has_anomaly_img)).sum().item())
            stat_mismatch_neg_label_nonemptymask += int(((~is_pos_label) & has_anomaly_img).sum().item())

            # patch统计
            total_patch = B * N
            valid_patch = int(valid_mask.sum().item())
            pos_patch = int(gt_abn.sum().item())
            nor_patch = int(gt_nor.sum().item())
            boundary_patch = total_patch - valid_patch

            stat_total_patches += total_patch
            stat_valid_patches += valid_patch
            stat_pos_patches += pos_patch
            stat_nor_patches += nor_patch
            stat_boundary_patches += boundary_patch

            # 用于原型采样的辅助变量
            label_expanded = label.view(B, 1).expand(-1, N)

            # --- C) 分层原型提取（保持不变） ---
            layer_losses_n, layer_losses_a = [], []
            layer_protos_n, layer_protos_a = [], []

            for li, layer_feat in enumerate(patch_features):
                current_patch_feat = layer_feat[:, 1:, :]  # [B,N,C]

                feat_n = current_patch_feat[(mask_ratio < thr_n) ]#& (label_expanded == 0)
                feat_a = current_patch_feat[(mask_ratio > thr_a) & (label_expanded == 1)]

                feat_n = subsample_rows(feat_n, args.proto_max_samples_n)
                feat_a = subsample_rows(feat_a, args.proto_max_samples_a)

                l_n, P_n = swav_list[li](feat_n)
                l_a, P_a = swav_abn_list[li](feat_a)

                layer_losses_n.append(l_n)
                layer_losses_a.append(l_a)

                # 原型梯度缩放（你原来的逻辑）
                gs = proto_grad_scale
                p_n_slice = P_n.detach() + gs * (P_n - P_n.detach())
                p_a_slice = P_a.detach() + gs * (P_a - P_a.detach())

                # ✅ 你说不做固定层级编码 -> 直接 append slice（修掉 p_n_final 未定义）
                layer_protos_n.append(p_n_slice)
                layer_protos_a.append(p_a_slice)

            loss_proto = torch.stack(layer_losses_n).mean() + torch.stack(layer_losses_a).mean()

            combined_P_n = torch.cat(layer_protos_n, dim=0)
            combined_P_a = torch.cat(layer_protos_a, dim=0)

            # --- D) Prompt 生成 ---
            feat_n_mapped = identity_proj(combined_P_n)
            feat_a_mapped = identity_proj(combined_P_a)
            feat_imp_mapped = deviation_proj(feat_n_mapped)

            res_n = prompt_learner(normal_items=feat_n_mapped)
            res_exp = prompt_learner(abnormal_items=feat_a_mapped)
            res_imp = prompt_learner(abnormal_items=feat_imp_mapped)

            t_n = F.normalize(model.encode_text_learn(*res_n), dim=-1)      # [M,D]
            t_imp = F.normalize(model.encode_text_learn(*res_imp), dim=-1)  # [M,D]
            t_exp = F.normalize(model.encode_text_learn(*res_exp), dim=-1)  # [M,D]
            t_abn_all = torch.cat([t_imp, t_exp], dim=0)                    # [2M,D]

            # --- E) Pixel Loss（Patch Level + ignore boundary + dice pos-only + BG topk） ---
            loss_pixel_total = 0.0
            loss_bce_rec = 0.0
            loss_dice_rec = 0.0
            loss_bg_topk_rec = 0.0
            loss_bg_mean_rec = 0.0

            # ✅ 问题4修正：MIL 使用多层平均 patch_diff
            patch_diff_sum = None

            for li, layer_feat in enumerate(patch_features):
                f = F.normalize(layer_feat[:, 1:, :], dim=-1)  # [B,N,C]

                logits_n = (f @ t_n.t()) / args.clip_temp
                logits_a = (f @ t_abn_all.t()) / args.clip_temp

                score_n = torch.logsumexp(logits_n, dim=-1) - math.log(logits_n.size(-1))  # [B,N]
                score_a = torch.logsumexp(logits_a, dim=-1) - math.log(logits_a.size(-1))  # [B,N]

                patch_diff = score_a - score_n  # [B,N] (logit for anomaly)
                patch_prob = torch.sigmoid(patch_diff)

                if patch_diff_sum is None:
                    patch_diff_sum = patch_diff
                else:
                    patch_diff_sum = patch_diff_sum + patch_diff

                # 1) Patch BCE：只在 valid 区域算
                if valid_mask.any():
                    pos = gt_abn[valid_mask].sum()
                    neg = gt_nor[valid_mask].sum()
                    pos_weight = torch.sqrt(neg / (pos + 1e-6)).clamp(1.0, 10.0)
                    l_bce = F.binary_cross_entropy_with_logits(
                        patch_diff[valid_mask], gt_abn[valid_mask],
                        pos_weight=pos_weight
                    )
                else:
                    l_bce = patch_diff.new_tensor(0.0)

                # 2) Dice：只在“异常图且gt_abn有正patch”的样本上算（避免 thr_a 过严时反向拉扯）
                pos_for_dice = has_anomaly_img & (gt_abn.sum(dim=1) > 0)
                if pos_for_dice.any():
                    v = valid_mask[pos_for_dice].float()
                    p_sub = patch_prob[pos_for_dice]* v
                    g_sub = gt_abn[pos_for_dice]* v
                    inter = (p_sub * g_sub).sum(dim=1)
                    union = p_sub.sum(dim=1) + g_sub.sum(dim=1)
                    dice_score = (2.0 * inter + 1e-5) / (union + 1e-5)
                    l_dice_val = 1.0 - dice_score.mean()
                else:
                    l_dice_val = patch_diff.new_tensor(0.0)

                # 统计 dice 实际用了多少异常图
                stat_dice_used_img += int(pos_for_dice.sum().item())/num_layers

                # 3) BG suppression：只在正常图（mask 为空）上做 Top-K（问题3修正）
                neg_for_bg = ~has_anomaly_img
                if neg_for_bg.any():
                    p_neg = patch_prob[neg_for_bg]  # [Bn,N]
                    l_bg_mean = p_neg.mean()

                    k_bg = max(1, int(bg_topk_ratio * p_neg.size(1)))
                    k_bg = min(k_bg, p_neg.size(1))
                    logits_neg = patch_diff[neg_for_bg]  # [Bn,N]
                    topk_logits = logits_neg.topk(k_bg, dim=1).values
                    l_bg_topk = F.softplus(topk_logits).mean()  # = BCEWithLogits(logit, 0)

                else:
                    l_bg_mean = patch_diff.new_tensor(0.0)
                    l_bg_topk = patch_diff.new_tensor(0.0)

                # 组合当前层 Pixel Loss（用 topk 作为真正的 bg loss）
                layer_pixel_loss = l_bce + args.dice_abn * l_dice_val + bg_weight * l_bg_topk

                loss_pixel_total += layer_pixel_loss
                loss_bce_rec += l_bce
                loss_dice_rec += l_dice_val
                loss_bg_topk_rec += l_bg_topk
                loss_bg_mean_rec += l_bg_mean

            loss_pixel = loss_pixel_total / len(patch_features)
            loss_bce_rec = loss_bce_rec / len(patch_features)
            loss_dice_rec = loss_dice_rec / len(patch_features)
            loss_bg_topk_rec = loss_bg_topk_rec / len(patch_features)
            loss_bg_mean_rec = loss_bg_mean_rec / len(patch_features)

            # --- F) Image Loss：保留原 CE + 小权重 MIL（问题5修正） ---
            # CE: image_features vs text（你原来版本的主干）
            img_f = F.normalize(image_features, dim=-1)
            img_logits_n_all = (img_f @ t_n.t()) / args.clip_temp
            img_logits_a_all = (img_f @ t_abn_all.t()) / args.clip_temp

            img_score_n = torch.logsumexp(img_logits_n_all, dim=-1) - math.log(img_logits_n_all.size(-1))
            img_score_a = torch.logsumexp(img_logits_a_all, dim=-1) - math.log(img_logits_a_all.size(-1))
            img_combined_logits = torch.stack([img_score_n, img_score_a], dim=1)
            loss_img_ce = F.cross_entropy(img_combined_logits, label)

            # MIL: 多层平均 patch_diff -> top-k pooling -> two-class CE
            patch_diff_mean = patch_diff_sum / len(patch_features)
            k_mil = max(1, int(mil_topk_ratio * patch_diff_mean.size(1)))
            k_mil = min(k_mil, patch_diff_mean.size(1))
            img_abn_score = patch_diff_mean.topk(k_mil, dim=1).values.mean(dim=1)  # [B]
            mil_logits = torch.stack([-img_abn_score, img_abn_score], dim=1)
            loss_img_mil = F.cross_entropy(mil_logits, label)

            loss_img = loss_img_ce + mil_img_weight * loss_img_mil

            # MIL score separation log
            if is_pos_label.any():
                stat_mil_pos_sum += float(img_abn_score[is_pos_label].sum().item())
                stat_mil_pos_cnt += int(is_pos_label.sum().item())
            if (~is_pos_label).any():
                stat_mil_neg_sum += float(img_abn_score[~is_pos_label].sum().item())
                stat_mil_neg_cnt += int((~is_pos_label).sum().item())

            # --- G) cross / orth（保持不变） ---
            loss_cross = 0.0
            for i in range(num_layers):
                P_n = F.normalize(swav_list[i].prototypes, dim=1)
                P_a = F.normalize(swav_abn_list[i].prototypes, dim=1)
                loss_cross += cross_sep_loss(P_n, P_a, topk=10, margin=0.3)
            loss_cross /= num_layers

            Pn_all = torch.cat([m.prototypes for m in swav_list], dim=0)
            Pa_all = torch.cat([m.prototypes for m in swav_abn_list], dim=0)
            loss_orth = proto_orth_loss(Pn_all) + proto_orth_loss(Pa_all)
            # 在 forward 得到 loss_img / loss_pixel 后，backward 前：
            shared_params = []
            shared_params += list(prompt_learner.parameters())
            shared_params += list(identity_proj.parameters())
            shared_params += list(deviation_proj.parameters())
            
            gn_img = grad_norm(loss_img, shared_params)
            gn_pix = grad_norm(args.pixel_loss_weight * loss_pixel, shared_params)
            if torch.rand(1).item() < 0.02:
                logger.info(f"[GradNorm] img={gn_img:.4e} pix={gn_pix:.4e} ratio(pix/img)={gn_pix/(gn_img+1e-12):.2f}")

            proto_params = []
            for m in itertools.chain(swav_list, swav_abn_list):
                proto_params.append(m.prototypes)
            
            gn_img_p = grad_norm(loss_img, proto_params)
            gn_pix_p = grad_norm(args.pixel_loss_weight * loss_pixel, proto_params)
            if torch.rand(1).item() < 0.02:
                logger.info(f"[GradNorm-Proto] img={gn_img_p:.3e} pix={gn_pix_p:.3e} ratio={gn_pix_p/(gn_img_p+1e-12):.2f}")

            
            # --- H) Backward ---
            total_loss = (
                loss_img
                + args.swav_loss_weight * loss_proto
                + args.cross_loss_weight * loss_cross
                + args.pixel_loss_weight * loss_pixel
                + args.orth_loss_weight * loss_orth
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 原型归一化约束
            with torch.no_grad():
                for m in itertools.chain(swav_list, swav_abn_list):
                    m.prototypes.data = F.normalize(m.prototypes.data, dim=1, eps=1e-6)

            # 记录 batch loss
            epoch_losses['total'].append(total_loss.item())
            epoch_losses['img'].append(loss_img.item())
            epoch_losses['img_ce'].append(loss_img_ce.item())
            epoch_losses['img_mil'].append(loss_img_mil.item())
            epoch_losses['pixel'].append(loss_pixel.item())
            epoch_losses['proto'].append(loss_proto.item())
            epoch_losses['orth'].append(loss_orth.item())
            epoch_losses['cross'].append(loss_cross.item())
            epoch_losses['bce_pixel'].append(loss_bce_rec.item())
            epoch_losses['dice_abn'].append(loss_dice_rec.item())
            epoch_losses['bg_topk'].append(loss_bg_topk_rec.item())
            epoch_losses['bg_mean'].append(loss_bg_mean_rec.item())

            pbar.set_postfix({
                "Loss": f"{total_loss.item():.3f}",
                "ImgCE": f"{loss_img_ce.item():.3f}",
                "ImgMIL": f"{loss_img_mil.item():.3f}",
                "Pix": f"{loss_pixel.item():.3f}",
                "BCE": f"{loss_bce_rec.item():.3f}",
                "Dice": f"{loss_dice_rec.item():.3f}",
                "BG": f"{loss_bg_topk_rec.item():.3f}",
            })

        # ===== epoch 汇总日志 =====
        avg_loss = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        logger.info(f"--- Epoch {epoch+1} Report ---")
        logger.info(
            f"Total: {avg_loss['total']:.4f} | "
            f"Img: {avg_loss['img']:.4f} (CE:{avg_loss['img_ce']:.4f} + {mil_img_weight}*MIL:{avg_loss['img_mil']:.4f}) | "
            f"Pixel: {avg_loss['pixel']:.4f} (BCE:{avg_loss['bce_pixel']:.4f}, Dice:{avg_loss['dice_abn']:.4f}, BG_topk:{avg_loss['bg_topk']:.6f}, BG_mean:{avg_loss['bg_mean']:.6f}) | "
            f"Proto: {avg_loss['proto']:.4f} | Orth: {avg_loss['orth']:.6f} | Cross: {avg_loss['cross']:.6f}"
        )

        # 关键验证日志：label/mask一致性 + patch监督密度 + MIL 分离度
        pos_ratio_mask = stat_pos_img_by_mask / max(1, stat_total_img)
        pos_ratio_label = stat_pos_img_by_label / max(1, stat_total_img)

        valid_ratio = stat_valid_patches / max(1, stat_total_patches)
        pos_patch_ratio_all = stat_pos_patches / max(1, stat_total_patches)
        nor_patch_ratio_all = stat_nor_patches / max(1, stat_total_patches)
        boundary_ratio = stat_boundary_patches / max(1, stat_total_patches)
        pos_in_valid = stat_pos_patches / max(1, stat_valid_patches)

        mil_pos_mean = stat_mil_pos_sum / max(1, stat_mil_pos_cnt)
        mil_neg_mean = stat_mil_neg_sum / max(1, stat_mil_neg_cnt)

        logger.info(
            f"[Verify] pos_img_ratio(mask)={pos_ratio_mask:.4f} | pos_img_ratio(label)={pos_ratio_label:.4f} | "
            f"mismatch(pos_label & empty_mask)={stat_mismatch_pos_label_emptymask} | "
            f"mismatch(neg_label & nonempty_mask)={stat_mismatch_neg_label_nonemptymask} | "
            f"dice_used_img_cnt={stat_dice_used_img}"
        )
        logger.info(
            f"[Verify] patches: valid_ratio={valid_ratio:.4f} | boundary_ratio={boundary_ratio:.4f} | "
            f"pos_patch_ratio_all={pos_patch_ratio_all:.6f} | nor_patch_ratio_all={nor_patch_ratio_all:.6f} | "
            f"pos_in_valid={pos_in_valid:.6f}"
        )
        logger.info(
            f"[Verify] MIL score mean: pos={mil_pos_mean:.4f} (n={stat_mil_pos_cnt}) | "
            f"neg={mil_neg_mean:.4f} (n={stat_mil_neg_cnt})"
        )
        logger.info(
            f"[WeightedTerms] Img={avg_loss['img']:.4f} | "
            f"Pixel={args.pixel_loss_weight * avg_loss['pixel']:.4f} | "
            f"Proto={args.swav_loss_weight * avg_loss['proto']:.4f} | "
            f"Orth={args.orth_loss_weight * avg_loss['orth']:.6f} | "
            f"Cross={args.cross_loss_weight * avg_loss['cross']:.6f}"
        )

        # checkpoint 保存保持不变
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_path, f"checkpoint_epoch_{epoch+1}.pth")
            save_dict = {
                "epoch": epoch + 1,
                "prompt_learner": prompt_learner.state_dict(),
                "identity_proj": identity_proj.state_dict(),
                "deviation_proj": deviation_proj.state_dict(),
                "swav_list": swav_list.state_dict(),
                "swav_abn_list": swav_abn_list.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args)
            }
            torch.save(save_dict, save_path)
            logger.info(f"Model checkpoint saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP Full Training (Route1 vMF Mixture Prototypes)")

    # data / io
    parser.add_argument("--train_data_path", type=str, default="./data/visa")
    parser.add_argument("--save_path", type=str, default='./checkpoints/experiment_1')
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--num_workers", type=int, default=4)

    # prompt learner
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)

    # prototypes
    parser.add_argument("--num_prototypes", type=int, default=64)
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])

    # training
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.0004)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--image_size", type=int, default=518)

    # weights
    parser.add_argument("--swav_loss_weight", type=float, default=0.01)
    parser.add_argument("--pixel_loss_weight", type=float, default=1)
    parser.add_argument("--orth_loss_weight", type=float, default=1)

    # CLIP temperature
    parser.add_argument("--clip_temp", type=float, default=0.07)

    # Route1 vMF hyperparams
    parser.add_argument("--vmf_kappa_n", type=float, default=30.0)
    parser.add_argument("--vmf_tau_q_n", type=float, default=1.0)
    parser.add_argument("--vmf_usage_reg_n", type=float, default=0.01)
    parser.add_argument("--warmup_epoch", type=int, default=8)

    parser.add_argument("--vmf_kappa_a", type=float, default=20.0)
    parser.add_argument("--vmf_tau_q_a", type=float, default=1.2)
    parser.add_argument("--vmf_usage_reg_a", type=float, default=0.001)

    parser.add_argument("--cross_loss_weight", type=float, default=1)
    parser.add_argument("--dice_abn", type=float, default=1)
    parser.add_argument("--focal_1", type=float, default=1)
    parser.add_argument("--focal_2", type=float, default=1)
    parser.add_argument("--focal_gamma", type=float, default=1)
    parser.add_argument("--prompt_num", type=int, default=8)

    # proto sampling
    parser.add_argument("--proto_max_samples_n", type=int, default=8192)
    parser.add_argument("--proto_max_samples_a", type=int, default=2048)

    # ===================== Loss Optimization Args (NEW) =====================
    # 1) dice_abn 只在异常图像上计算（推荐开）
    parser.add_argument("--dice_abn_pos_only", type=int, default=1,
                        help="1: dice_abn only on anomalous images; 0: dice_abn on all images")

    # 2) 正常图像背景正则（patch-level topk）
    parser.add_argument("--bg_loss_weight", type=float, default=0.3,
                        help="weight of background regularization on normal images (0 disables)")
    parser.add_argument("--bg_topk_ratio", type=float, default=0.20,
                        help="top-k ratio for bg regularization on normal images (patch-level)")

    # 3) MIL image loss（把分类与定位绑定）
    parser.add_argument("--mil_img_loss_weight", type=float, default=0.1,
                        help="weight of MIL image loss (0 disables)")
    parser.add_argument("--mil_topk_ratio", type=float, default=0.05,
                        help="top-k ratio for MIL pooling (patch-level)")
    parser.add_argument(
        "--aug_rate",
        type=float,
        default=-1.0,
        help="mosaic augmentation probability for training; -1 means dataset default (mvtec=0.2, others=0.0)",
    )

    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
