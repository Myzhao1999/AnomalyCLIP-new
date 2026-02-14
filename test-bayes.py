
import os
import random
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as cc_label
from tqdm import tqdm
from tabulate import tabulate
import time
import concurrent.futures
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
# 自定义库导入
import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from logger import get_logger
from utils import get_transform
# 假设 metrics.py 存在于同级目录
from metrics import image_level_metrics, pixel_level_metrics


# =================================================================================
# 0. Bayes-PFL 基准数据与映射 (Baseline Data)
# =================================================================================

BAYES_PFL_RESULTS = {
    # Industrial (Both Metrics)
    "MVTec-AD":      {"img_auc": 92.3, "img_f1": 93.1, "img_ap": 96.7, "px_auc": 91.8, "px_pro": 87.4, "px_ap": 48.3},
    "VisA":          {"img_auc": 87.0, "img_f1": 84.1, "img_ap": 89.2, "px_auc": 95.6, "px_pro": 88.9, "px_ap": 29.8},
    "BTAD":          {"img_auc": 93.2, "img_f1": 91.9, "img_ap": 96.5, "px_auc": 93.9, "px_pro": 76.6, "px_ap": 47.1},
    "KSDD2":         {"img_auc": 97.3, "img_f1": 92.3, "img_ap": 97.9, "px_auc": 99.6, "px_pro": 97.6, "px_ap": 73.7},
    "RSDD":          {"img_auc": 94.1, "img_f1": 89.6, "img_ap": 92.3, "px_auc": 99.6, "px_pro": 98.0, "px_ap": 39.1},
    "DAGM":          {"img_auc": 97.7, "img_f1": 95.7, "img_ap": 97.0, "px_auc": 99.3, "px_pro": 98.0, "px_ap": 43.1},
    "DTD-Synthetic": {"img_auc": 95.1, "img_f1": 95.1, "img_ap": 98.4, "px_auc": 97.8, "px_pro": 94.3, "px_ap": 69.9},
    "mpdd":          {"img_auc": 81.8, "img_f1": 84.6, "img_ap": 85.5, "px_auc": 97.7, "px_pro": 92.2, "px_ap": 32.7},
    # Medical (Image-level only in Bayes Table)
    "HeadCT":        {"img_auc": 96.5, "img_f1": 92.9, "img_ap": 95.5, "px_auc": None, "px_pro": None, "px_ap": None},
    "BrainMRI":      {"img_auc": 96.2, "img_f1": 92.8, "img_ap": 92.4, "px_auc": None, "px_pro": None, "px_ap": None},
    "Br35H":         {"img_auc": 97.8, "img_f1": 93.6, "img_ap": 96.2, "px_auc": None, "px_pro": None, "px_ap": None},
    # Medical (Pixel-level only in Bayes Table)
    "ISIC":          {"img_auc": None, "img_f1": None, "img_ap": None, "px_auc": 92.2, "px_pro": 87.6, "px_ap": 84.6},
    "CVC-ColonDB":   {"img_auc": None, "img_f1": None, "img_ap": None, "px_auc": 82.1, "px_pro": 76.1, "px_ap": 31.9},
    "CVC-ClinicDB":  {"img_auc": None, "img_f1": None, "img_ap": None, "px_auc": 89.6, "px_pro": 78.4, "px_ap": 53.2},
    "Endo":          {"img_auc": None, "img_f1": None, "img_ap": None, "px_auc": 89.2, "px_pro": 74.8, "px_ap": 58.6},
    "Kvasir":        {"img_auc": None, "img_f1": None, "img_ap": None, "px_auc": 85.4, "px_pro": 63.9, "px_ap": 54.2},
    "TN3K":          {"img_auc": None, "img_f1": None, "img_ap": None, "px_auc": 79.4, "px_pro": 52.1, "px_ap": 40.6},
    "SDD":           {"img_auc": 93.1, "img_f1": 76.0, "img_ap": 66.4, "px_auc": 97.9, "px_pro": 94.4, "px_ap": 19.0},
}


def get_bayes_comparison_key(arg_dataset_name):
    arg_lower = arg_dataset_name.lower()
    mapping = {
        'mvtec': 'MVTec-AD',
        'visa': 'VisA',
        'btad': 'BTAD',
        'ksdd2': 'KSDD2',
        'rsdd': 'RSDD',
        'dagm': 'DAGM',
        'mpdd': 'mpdd',
        'dtd': 'DTD-Synthetic',
        'headct': 'HeadCT',
        'brainmri': 'BrainMRI',
        'br35h': 'Br35H',
        'isbi': 'ISIC',
        'colon': 'CVC-ColonDB',
        'clinic': 'CVC-ClinicDB',
        'endo': 'Endo',
        'kvasir': 'Kvasir',
        'sdd': 'SDD',
        'tn3k': 'TN3K',
    }
    if arg_lower in mapping:
        return mapping[arg_lower]
    return None


def format_score_with_diff(current_val, baseline_dict, metric_key):
    """
    Format score as '95.0 (+1.2)' if baseline exists, else '95.0'.
    Returns '-' if current_val is None (indicating skipped metric).
    """
    if current_val is None:
        return "-"

    current_percent = current_val * 100
    if baseline_dict and metric_key in baseline_dict and baseline_dict[metric_key] is not None:
        baseline = baseline_dict[metric_key]
        diff = current_percent - baseline
        return f"{current_percent:.1f} ({diff:+.1f})"
    else:
        return f"{current_percent:.1f}"


# =================================================================================
# 核心逻辑封装：严格保留原代码所有处理细节
# 仅【新增】Image AUROC 的多策略统计：Global-only / Patch-only / Fusion / Heuristic
# =================================================================================

def process_object_metrics(obj_name, obj_data, is_image_only, is_pixel_only):
    """
    进程内逻辑：
    1. 计算 Heuristic (串行，因为很快)
    2. 并行计算各个 Metrics (使用线程池，共享内存，避免复制大数组)
    """
    # --- 1. Heuristic 计算 (保持不变) ---
    special_objs = ["capsules", "macaroni1", "macaroni2", "pipe_fryum", "screw", "cashew", "chewinggum"]
    can_k = -20 if obj_name in special_objs else -2000
    v1_norm = np.array(obj_data['pr_sp_img_lse'])
    topk_scores = []

    for amap in obj_data['anomaly_maps_lse']:
        flat = amap.reshape(-1)
        curr_k = can_k if flat.size >= abs(can_k) else -flat.size
        val = 0.0 if curr_k == 0 else np.mean(np.partition(flat, kth=curr_k)[curr_k:])
        topk_scores.append(val)
    topk_norm = np.array(topk_scores)

    obj_data['pr_sp_heuristic'] = topk_norm.tolist()
    obj_data['pr_sp_heuristic_fusion'] = (0.5 * v1_norm + 0.5 * topk_norm).tolist()
    obj_data['anomaly_maps'] = obj_data['anomaly_maps_lse']

    # --- 2. 并行指标计算 (新增逻辑：仅新增 image-auroc 的多策略输出，不改原输出列) ---
    res_stats = {
        'px_lse': None, 'pro_lse': None, 'px_ap_lse': None,
        'img_auc_fus': None, 'img_auc_heu': None, 'f1_fus': None, 'ap_fus': None,
        # 【新增】诊断用 image AUROC（不影响原表格）
        'img_auc_global': None,          # 仅 scores_img_lse
        'img_auc_patch_topk': None,      # 仅 pixel_topk_mean_lse（推理时已存 pr_sp_patch_topk_lse）
    }

    # 构造临时 wrapper
    temp_results_wrapper = {obj_name: obj_data}

    # 定义具体的任务函数 (利用闭包特性访问 obj_data)
    def calc_px_auroc():
        if is_image_only:
            return None
        return pixel_level_metrics(temp_results_wrapper, obj_name, "pixel-auroc")

    def calc_px_pro():
        if is_image_only:
            return None
        gt_masks = obj_data['imgs_masks']
        lse_maps = obj_data['anomaly_maps_lse']
        # PRO 是最慢的，这里是并行的关键收益点
        return pro_auc_np(gt_masks, lse_maps, max_fpr=0.3, num_th=200)

    def calc_px_ap():
        if is_image_only:
            return None
        gt_masks = obj_data['imgs_masks']
        lse_maps = obj_data['anomaly_maps_lse']
        y_true_px = np.concatenate([m.reshape(-1) for m in gt_masks], axis=0).astype(np.uint8)
        y_score_px = np.concatenate([m.reshape(-1) for m in lse_maps], axis=0).astype(np.float32)
        return average_precision_score(y_true_px, y_score_px)

    def calc_img_metrics():
        if is_pixel_only:
            return (None, None, None, None, None, None)

        # 注意：这里我们不需要深拷贝巨大的 image/map 数据，只拷贝引用结构
        local_wrapper = {obj_name: {
            'pr_sp': None,
            'gt_sp': obj_data['gt_sp']  # 引用，不复制内存
        }}

        # 【新增 0】Image AUROC (Global only: scores_img_lse)
        local_wrapper[obj_name]['pr_sp'] = obj_data['pr_sp_img_lse']
        auc_global = image_level_metrics(local_wrapper, obj_name, "image-auroc")

        # 【新增 1】Image AUROC (Patch only: pixel_topk_mean_lse saved as pr_sp_patch_topk_lse)
        if 'pr_sp_patch_topk_lse' in obj_data and obj_data['pr_sp_patch_topk_lse'] is not None and len(obj_data['pr_sp_patch_topk_lse']) > 0:
            local_wrapper[obj_name]['pr_sp'] = obj_data['pr_sp_patch_topk_lse']
            auc_patch = image_level_metrics(local_wrapper, obj_name, "image-auroc")
        else:
            auc_patch = None

        # 1. Image AUROC (Fusion) - 保持不变
        local_wrapper[obj_name]['pr_sp'] = obj_data['pr_sp_fusion_lse']
        auc_fus = image_level_metrics(local_wrapper, obj_name, "image-auroc")

        # 2. Image AUROC (Heuristic) - 保持不变
        local_wrapper[obj_name]['pr_sp'] = obj_data['pr_sp_heuristic_fusion']
        auc_heu = image_level_metrics(local_wrapper, obj_name, "image-auroc")

        # 3. F1 & AP - 保持不变（Fusion 分数）
        y_true = obj_data['gt_sp']
        y_score = obj_data['pr_sp_fusion_lse']
        f1 = f1_max_np(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        return (auc_fus, auc_heu, f1, ap, auc_global, auc_patch)

    # 使用线程池并行执行这 4 个任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as thread_executor:
        future_px_auc = thread_executor.submit(calc_px_auroc)
        future_px_pro = thread_executor.submit(calc_px_pro)
        future_px_ap = thread_executor.submit(calc_px_ap)
        future_img = thread_executor.submit(calc_img_metrics)

        # 获取结果 (会阻塞直到计算完成)
        res_stats['px_lse'] = future_px_auc.result()
        res_stats['pro_lse'] = future_px_pro.result()
        res_stats['px_ap_lse'] = future_px_ap.result()

        img_res = future_img.result()
        res_stats['img_auc_fus'] = img_res[0]
        res_stats['img_auc_heu'] = img_res[1]
        res_stats['f1_fus'] = img_res[2]
        res_stats['ap_fus'] = img_res[3]
        # 【新增】诊断指标
        res_stats['img_auc_global'] = img_res[4]
        res_stats['img_auc_patch_topk'] = img_res[5]

    # --- 3. 生成 Row (保持不变：不改原表格列) ---
    row = [
        obj_name,
        f"{res_stats['px_lse'] * 100:.1f}" if res_stats['px_lse'] is not None else "-",
        f"{res_stats['pro_lse'] * 100:.1f}" if res_stats['pro_lse'] is not None else "-",
        f"{res_stats['px_ap_lse'] * 100:.1f}" if res_stats['px_ap_lse'] is not None else "-",
        f"{res_stats['img_auc_fus'] * 100:.1f}" if res_stats['img_auc_fus'] is not None else "-",
        f"{res_stats['img_auc_heu'] * 100:.1f}" if res_stats['img_auc_heu'] is not None else "-",
        f"{res_stats['f1_fus'] * 100:.1f}" if res_stats['f1_fus'] is not None else "-",
        f"{res_stats['ap_fus'] * 100:.1f}" if res_stats['ap_fus'] is not None else "-"
    ]

    return obj_name, res_stats, row


# =================================================================================
# 1. 辅助度量函数 (Metrics Helpers)
# =================================================================================

"""
def f1_max_np(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    if y_true.size == 0:
        return 0.0
    pos = int(y_true.sum(0))
    if pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    fn = pos - tp
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(tp + fn, 1e-12)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(np.max(f1)) if f1.size else 0.0

"""

def f1_max_np(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    precisions, recalls, thresholds = precision_recall_curve(y_true , y_score )
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    best_threshold_cls = thresholds[np.argmax(f1_scores)]
    f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
    return float(f1_sp)



def pro_auc_np(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    masks = np.asarray(masks)
    amaps = np.asarray(amaps, dtype=np.float32)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


# =================================================================================
# 2. 模型组件
# =================================================================================

class ResidualPerFeatureMLP(nn.Module):
    def __init__(self, d_model=768, expansion=2, dropout=0.0):
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

    def forward(self, x):
        if x.numel() == 0:
            return x
        return self.ln2(self.ffn(self.ln1(x)) + x)


class VMFMixturePrototypes(nn.Module):
    def __init__(self, num_prototypes: int, dim: int = 768, kappa: float = 30.0, tau_q: float = 1.0, learn_pi: bool = True,
                 usage_reg: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.K = int(num_prototypes)
        self.dim = int(dim)
        self.kappa = float(kappa)
        self.tau_q = float(tau_q)
        self.usage_reg = float(usage_reg)
        self.eps = float(eps)
        self.prototypes = nn.Parameter(F.normalize(torch.randn(self.K, self.dim), dim=1))
        if learn_pi:
            self.logits_pi = nn.Parameter(torch.zeros(self.K))
        else:
            self.register_buffer("logits_pi", torch.zeros(self.K))

    def forward(self, feats: torch.Tensor):
        if feats.numel() == 0:
            return feats.new_tensor(0.0), F.normalize(self.prototypes, dim=1)
        x = F.normalize(feats, dim=1)
        P = F.normalize(self.prototypes, dim=1)
        log_pi = F.log_softmax(self.logits_pi, dim=0)
        e = self.kappa * (x @ P.t()) + log_pi.unsqueeze(0)
        q = F.softmax(e / self.tau_q, dim=1)
        elbo = (q * (e - torch.log(q + self.eps))).sum(dim=1)
        loss = -elbo.mean()
        if self.usage_reg > 0:
            mean_q = q.mean(dim=0)
            uniform = torch.full_like(mean_q, 1.0 / self.K)
            kl = (mean_q * (torch.log(mean_q + self.eps) - torch.log(uniform + self.eps))).sum()
            loss = loss + self.usage_reg * kl
        return loss, P

    @torch.no_grad()
    def get_prototypes(self):
        return F.normalize(self.prototypes, dim=1)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =================================================================================
# 3. 主测试逻辑
# =================================================================================

def test(args):
    os.makedirs(args.save_path, exist_ok=True)
    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    params = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device="cpu", design_details=params)
    model.eval()
    num_layers = len(args.features_list)
    prompt_learner = AnomalyCLIP_PromptLearner(model, params, n_prompts=args.prompt_num, num_layers=num_layers)

    vmf_list = nn.ModuleList([
        VMFMixturePrototypes(
            num_prototypes=args.num_prototypes,
            dim=768,
            kappa=args.vmf_kappa_n,
            tau_q=args.vmf_tau_q_n,
            learn_pi=True,
            usage_reg=args.vmf_usage_reg_n
        ) for _ in range(num_layers)
    ])
    vmf_abn_list = nn.ModuleList([
        VMFMixturePrototypes(
            num_prototypes=max(1, args.num_prototypes // 4),
            dim=768,
            kappa=args.vmf_kappa_a,
            tau_q=args.vmf_tau_q_a,
            learn_pi=True,
            usage_reg=args.vmf_usage_reg_a
        ) for _ in range(num_layers)
    ])

    identity_proj = ResidualPerFeatureMLP()
    deviation_proj = ResidualPerFeatureMLP()

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    prompt_learner.load_state_dict(ckpt["prompt_learner"])

    print("\n" + "=" * 40)
    print("Checking Gating Parameters (Prototype Influence):")

    with torch.no_grad():
        if hasattr(prompt_learner, 'gate_pos'):
            gpos_raw = prompt_learner.gate_pos.detach().cpu()  # [M,n_ctx,1]
            gpos = torch.sigmoid(prompt_learner.gate_pos).detach().cpu()  # [M,n_ctx,1]
            print(f" -> gate_pos raw: mean={gpos_raw.mean().item():.4f}, min={gpos_raw.min().item():.4f}, max={gpos_raw.max().item():.4f}")
            print(" -> gate_pos sigmoid per-prompt mean (over tokens):", gpos.mean(dim=1).squeeze(-1).numpy())
        else:
            print(" -> gate_pos not found")

        if hasattr(prompt_learner, 'gate_neg'):
            gneg_raw = prompt_learner.gate_neg.detach().cpu()
            gneg = torch.sigmoid(prompt_learner.gate_neg).detach().cpu()
            print(f" -> gate_neg raw: mean={gneg_raw.mean().item():.4f}, min={gneg_raw.min().item():.4f}, max={gneg_raw.max().item():.4f}")
            print(" -> gate_neg sigmoid per-prompt mean (over tokens):", gneg.mean(dim=1).squeeze(-1).numpy())
        else:
            print(" -> gate_neg not found")

    print("=" * 40 + "\n")

    identity_proj.load_state_dict(ckpt["identity_proj"])
    deviation_proj.load_state_dict(ckpt["deviation_proj"])

    if ("swav_list" in ckpt) and ("swav_abn_list" in ckpt):
        vmf_list.load_state_dict(ckpt["swav_list"])
        vmf_abn_list.load_state_dict(ckpt["swav_abn_list"])
        logger.info("Loaded Route1-vMF checkpoint.")
    else:
        print('=====load error=====')

    model.to(device)
    prompt_learner.to(device)
    vmf_list.to(device)
    vmf_abn_list.to(device)
    identity_proj.to(device)
    deviation_proj.to(device)

    model.visual.DAPM_replace(DPAM_layer=None)

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    obj_list = test_data.obj_list

    # 【新增】results 中增加 pr_sp_patch_topk_lse，用于 image-auroc 诊断（不影响原逻辑）
    results = {
        obj: {
            k: [] for k in [
                'gt_sp',
                'pr_sp_img_lse',
                'pr_sp_fusion_lse',
                'pr_sp_patch_topk_lse',   # <<< 新增：patch-only image score
                'imgs_masks',
                'anomaly_maps',
                'anomaly_maps_lse',
                'pr_sp_heuristic',
                'pr_sp_heuristic_fusion'
            ]
        } for obj in obj_list
    }

    with torch.inference_mode():
        combined_P_n = torch.cat([vmf_list[i].get_prototypes() for i in range(num_layers)], dim=0)
        combined_P_a = torch.cat([vmf_abn_list[i].get_prototypes() for i in range(num_layers)], dim=0)

        feat_n_mapped = identity_proj(combined_P_n)
        feat_a_mapped = identity_proj(combined_P_a)
        feat_imp_mapped = deviation_proj(feat_n_mapped)
        t_n = F.normalize(model.encode_text_learn(*prompt_learner(normal_items=feat_n_mapped)), dim=-1)
        t_exp = F.normalize(model.encode_text_learn(*prompt_learner(abnormal_items=feat_a_mapped)), dim=-1)
        t_imp = F.normalize(model.encode_text_learn(*prompt_learner(abnormal_items=feat_imp_mapped)), dim=-1)
        t_abn_all = torch.cat([t_exp, t_imp], dim=0)

        for items in tqdm(test_dataloader, desc="Batch Inference"):
            image = items['img'].to(device, non_blocking=True)
            batch_cls = items['cls_name']
            batch_anom = items['anomaly']
            gt_mask = items['img_mask']
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=None)
            img_f = F.normalize(image_features, dim=-1)
            logits_n = img_f @ t_n.t() / args.clip_temp
            logits_abn = img_f @ t_abn_all.t() / args.clip_temp
            sn_lse = torch.logsumexp(logits_n, dim=-1) - math.log(logits_n.shape[-1])
            sa_lse = torch.logsumexp(logits_abn, dim=-1) - math.log(logits_abn.shape[-1])
            scores_img_lse = torch.stack([sn_lse, sa_lse], dim=1).softmax(dim=-1)[:, 1]

            batch_size = image.shape[0]
            ph = int(np.sqrt(patch_features[0].shape[1] - 1))
            sum_maps_lse_small = 0
            layer_topk_scores_lse = []

            for pf in patch_features:
                pf = F.normalize(pf[:, 1:, :], dim=-1)
                p_logits_n = (pf @ t_n.t()) / args.clip_temp
                p_logits_a = (pf @ t_abn_all.t()) / args.clip_temp
                p_score_n = torch.logsumexp(p_logits_n, dim=-1) - math.log(p_logits_n.shape[-1])
                p_score_a = torch.logsumexp(p_logits_a, dim=-1) - math.log(p_logits_a.shape[-1])
                a_maps_lse = torch.stack([p_score_n, p_score_a], dim=1).softmax(dim=1)[:, 1, :]

                tk_val_lse, _ = torch.topk(a_maps_lse, k=min(args.topk, a_maps_lse.shape[1]), dim=-1)
                layer_topk_scores_lse.append(tk_val_lse.mean(dim=-1))

                sum_maps_lse_small += a_maps_lse.view(batch_size, ph, ph)

            sum_maps_lse_small /= num_layers
            batch_anomaly_maps_lse = F.interpolate(
                sum_maps_lse_small.unsqueeze(1),
                size=args.image_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            batch_maps_lse_np = batch_anomaly_maps_lse.cpu().numpy()
            smoothed_maps_lse_batch = gaussian_filter(batch_maps_lse_np, sigma=(0, args.sigma, args.sigma))

            # patch-only image score（与你的 fusion 中同一个分量），【新增存储】用于 image-auroc 诊断
            pixel_topk_mean_lse = torch.stack(layer_topk_scores_lse, dim=0).mean(dim=0)

            # 原有 fusion（保持不变）
            fusion_scores_lse = scores_img_lse + pixel_topk_mean_lse

            for b in range(batch_size):
                c = batch_cls[b]
                results[c]['pr_sp_img_lse'].append(scores_img_lse[b].item())
                results[c]['pr_sp_patch_topk_lse'].append(pixel_topk_mean_lse[b].item())  # <<< 新增
                results[c]['pr_sp_fusion_lse'].append(fusion_scores_lse[b].item())
                results[c]['gt_sp'].append(batch_anom[b].item())
                results[c]['anomaly_maps_lse'].append(smoothed_maps_lse_batch[b])
                results[c]['imgs_masks'].append(gt_mask[b].squeeze().cpu().numpy())

    # =================================================================================
    # Pixel Ratio Statistics (Dataset Level)
    # =================================================================================

    logger.info("Calculating Normal / Abnormal Pixel Ratio ...")

    dataset_normal_pixels = 0
    dataset_abnormal_pixels = 0

    pixel_ratio_table = []

    for obj in obj_list:
        masks = results[obj]['imgs_masks']

        obj_normal = 0
        obj_abnormal = 0

        for m in masks:
            m = (m > 0.5).astype(np.uint8)
            obj_abnormal += int(m.sum())
            obj_normal += int((m == 0).sum())

        dataset_normal_pixels += obj_normal
        dataset_abnormal_pixels += obj_abnormal

        total = obj_normal + obj_abnormal
        abn_ratio = obj_abnormal / total if total > 0 else 0

        pixel_ratio_table.append([
            obj,
            obj_normal,
            obj_abnormal,
            f"{abn_ratio * 100:.4f}%"
        ])

    dataset_total = dataset_normal_pixels + dataset_abnormal_pixels
    dataset_ratio = dataset_abnormal_pixels / dataset_total if dataset_total > 0 else 0

    pixel_ratio_table.append([
        "Dataset-Total",
        dataset_normal_pixels,
        dataset_abnormal_pixels,
        f"{dataset_ratio * 100:.4f}%"
    ])

    ratio_table_str = tabulate(
        pixel_ratio_table,
        headers=["Object", "Normal Pixels", "Abnormal Pixels", "Abnormal Ratio"],
        tablefmt="pipe"
    )

    logger.info("\n" + ratio_table_str)

    # =================================================================================
    # 4. 指标计算与报告 (Metrics Reporting) - 多进程并行加速版
    # =================================================================================

    pixel_only_datasets = ["ISBI", "colon", "clinic", "endo", "Kvasir", "tn3k"]  # Medical Pixel
    image_only_datasets = ["headct", "BrainMRI", "br35h"]  # Medical Image

    is_pixel_only = args.dataset in pixel_only_datasets
    is_image_only = args.dataset in image_only_datasets

    headers = [
        "Object",
        "Pixel AUROC-LSE", "Pixel PRO-LSE", "Pixel AP-LSE",
        "Fusion-AUROC-LSE", "Fusion-Heuristic", "F1-Max(Fusion)", "AP(Fusion)"
    ]

    summary = {
        'px_lse': [], 'pro_lse': [], 'px_ap_lse': [],
        'img_auc_fus': [], 'img_auc_heu': [], 'f1_fus': [], 'ap_fus': []
    }
    table_ls = []

    # 【新增】image-auroc 多策略统计（不影响原表格）
    img_auc_strategy_rows = []
    img_auc_strategy_summary = {
        'img_auc_global': [],
        'img_auc_patch_topk': [],
        'img_auc_fus': [],
        'img_auc_heu': [],
    }

    logger.info("Starting Parallel Metric Calculation...")
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_obj = {
            executor.submit(process_object_metrics, obj, results[obj], is_image_only, is_pixel_only): obj
            for obj in obj_list
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_obj), total=len(obj_list), desc="Calculating Metrics"):
            obj_name, res_stats, row = future.result()

            table_ls.append(row)

            if res_stats['px_lse'] is not None:
                summary['px_lse'].append(res_stats['px_lse'])
            if res_stats['pro_lse'] is not None:
                summary['pro_lse'].append(res_stats['pro_lse'])
            if res_stats['px_ap_lse'] is not None:
                summary['px_ap_lse'].append(res_stats['px_ap_lse'])
            if res_stats['img_auc_fus'] is not None:
                summary['img_auc_fus'].append(res_stats['img_auc_fus'])
            if res_stats['img_auc_heu'] is not None:
                summary['img_auc_heu'].append(res_stats['img_auc_heu'])
            if res_stats['f1_fus'] is not None:
                summary['f1_fus'].append(res_stats['f1_fus'])
            if res_stats['ap_fus'] is not None:
                summary['ap_fus'].append(res_stats['ap_fus'])

            # 【新增】收集 image AUROC 策略诊断数据
            if (not is_pixel_only) and (res_stats.get('img_auc_fus', None) is not None):
                img_auc_strategy_rows.append([
                    obj_name,
                    f"{res_stats['img_auc_global'] * 100:.1f}" if res_stats['img_auc_global'] is not None else "-",
                    f"{res_stats['img_auc_patch_topk'] * 100:.1f}" if res_stats['img_auc_patch_topk'] is not None else "-",
                    f"{res_stats['img_auc_fus'] * 100:.1f}" if res_stats['img_auc_fus'] is not None else "-",
                    f"{res_stats['img_auc_heu'] * 100:.1f}" if res_stats['img_auc_heu'] is not None else "-",
                ])

                if res_stats['img_auc_global'] is not None:
                    img_auc_strategy_summary['img_auc_global'].append(res_stats['img_auc_global'])
                if res_stats['img_auc_patch_topk'] is not None:
                    img_auc_strategy_summary['img_auc_patch_topk'].append(res_stats['img_auc_patch_topk'])
                if res_stats['img_auc_fus'] is not None:
                    img_auc_strategy_summary['img_auc_fus'].append(res_stats['img_auc_fus'])
                if res_stats['img_auc_heu'] is not None:
                    img_auc_strategy_summary['img_auc_heu'].append(res_stats['img_auc_heu'])

    end_time = time.time()
    logger.info(f"Metrics calculation finished in {end_time - start_time:.2f} seconds.")

    table_ls.sort(key=lambda x: x[0])

    # --- Mean Row with Bayes Comparison (保持不变) ---
    bayes_key = get_bayes_comparison_key(args.dataset)
    baseline_stats = BAYES_PFL_RESULTS.get(bayes_key, {}) if bayes_key else {}

    mean_row = ["Mean (Diff)"]

    # Pixel Mean
    if (not is_image_only) and summary['px_lse']:
        mean_row.append(format_score_with_diff(np.mean(summary['px_lse']), baseline_stats, "px_auc"))
        mean_row.append(format_score_with_diff(np.mean(summary['pro_lse']), baseline_stats, "px_pro"))
        mean_row.append(format_score_with_diff(np.mean(summary['px_ap_lse']), baseline_stats, "px_ap"))
    else:
        mean_row.extend(["-", "-", "-"])

    # Image Mean
    if (not is_pixel_only) and summary['img_auc_fus']:
        mean_row.append(format_score_with_diff(np.mean(summary['img_auc_fus']), baseline_stats, "img_auc"))
        mean_row.append(format_score_with_diff(np.mean(summary['img_auc_heu']), baseline_stats, "img_auc"))
        mean_row.append(format_score_with_diff(np.mean(summary['f1_fus']), baseline_stats, "img_f1"))
        mean_row.append(format_score_with_diff(np.mean(summary['ap_fus']), baseline_stats, "img_ap"))
    else:
        mean_row.extend(["-", "-", "-", "-"])

    table_ls.append(mean_row)

    results_table = tabulate(table_ls, headers=headers, tablefmt="pipe")

    # 【新增】Image AUROC 多策略对比表（仅日志输出，不影响原表格）
    img_strategy_table_str = None
    if not is_pixel_only:
        img_auc_strategy_rows.sort(key=lambda x: x[0])

        img_strategy_headers = [
            "Object",
            "Img-AUROC(GlobalOnly)",
            "Img-AUROC(PatchTopKOnly)",
            "Img-AUROC(Fusion)",
            "Img-AUROC(HeuristicFusion)",
        ]

        # Mean 行（可选：对比 Bayes 的 img_auc，帮助定位 global/patch 哪个拖累）
        if img_auc_strategy_summary['img_auc_fus']:
            mean_strategy_row = ["Mean (Diff vs Bayes img_auc)"]
            mean_strategy_row.append(format_score_with_diff(np.mean(img_auc_strategy_summary['img_auc_global']), baseline_stats, "img_auc")
                                     if img_auc_strategy_summary['img_auc_global'] else "-")
            mean_strategy_row.append(format_score_with_diff(np.mean(img_auc_strategy_summary['img_auc_patch_topk']), baseline_stats, "img_auc")
                                     if img_auc_strategy_summary['img_auc_patch_topk'] else "-")
            mean_strategy_row.append(format_score_with_diff(np.mean(img_auc_strategy_summary['img_auc_fus']), baseline_stats, "img_auc")
                                     if img_auc_strategy_summary['img_auc_fus'] else "-")
            mean_strategy_row.append(format_score_with_diff(np.mean(img_auc_strategy_summary['img_auc_heu']), baseline_stats, "img_auc")
                                     if img_auc_strategy_summary['img_auc_heu'] else "-")

            img_auc_strategy_rows.append(mean_strategy_row)

        img_strategy_table_str = tabulate(img_auc_strategy_rows, headers=img_strategy_headers, tablefmt="pipe")

    final_log_path = os.path.join(args.save_path, args.result_log if args.result_log.endswith('.log') else 'results.log')
    with open(final_log_path, "a") as f:
        f.write("\n" + "=" * 30 + " PIXEL RATIO STATISTICS " + "=" * 30 + "\n")
        f.write(ratio_table_str + "\n")
        f.write("=" * 80 + "\n")

        # 【新增】把多策略 Image AUROC 也写入同一个 log 文件（只添加，不改原逻辑）
        if img_strategy_table_str is not None:
            f.write("\n" + "=" * 26 + " IMAGE AUROC STRATEGY DIAGNOSTIC " + "=" * 26 + "\n")
            f.write(f"Dataset: {args.dataset} (Mapped: {bayes_key})\n")
            f.write(img_strategy_table_str + "\n")
            f.write("=" * 80 + "\n")

    logger.info(f"\n{results_table}")

    # 【新增】输出到 logger（从而进入你的日志系统）
    if img_strategy_table_str is not None:
        logger.info("\n" + "=" * 26 + " IMAGE AUROC STRATEGY DIAGNOSTIC " + "=" * 26)
        logger.info(f"Dataset: {args.dataset} (Mapped: {bayes_key})")
        logger.info("\n" + img_strategy_table_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP-Optimized-Test")
    parser.add_argument("--data_path", type=str, default="./data/visa")
    parser.add_argument("--save_path", type=str, default='./results/unified_logs')
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--num_prototypes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--sigma", type=int, default=8)
    parser.add_argument("--result_log", type=str, default='total_test_results.log')
    parser.add_argument("--clip_temp", type=float, default=0.07)
    parser.add_argument("--vmf_kappa_n", type=float, default=30.0)
    parser.add_argument("--vmf_tau_q_n", type=float, default=1.0)
    parser.add_argument("--vmf_usage_reg_n", type=float, default=0.01)
    parser.add_argument("--vmf_kappa_a", type=float, default=20.0)
    parser.add_argument("--vmf_tau_q_a", type=float, default=1.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--vmf_usage_reg_a", type=float, default=0.0)
    parser.add_argument("--prompt_num", type=int, default=10)
    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)

