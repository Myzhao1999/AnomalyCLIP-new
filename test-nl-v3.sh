#!/usr/bin/env bash
set -euo pipefail

# ================= 配置区域 =================

# 1. 硬件与 Python 脚本配置
DEVICE_ID=7
PYTHON_SCRIPT="test-bayes2.py"

# 2. 模型核心参数
DEPTH=9
N_CTX=12
T_N_CTX=4
NUM_PROTOTYPES=64
FEATURES_LIST=(9 12 15 18 21 24)
IMAGE_SIZE=518
SIGMA=8
topk=7
prompt_num=12
# 3. 路径配置  (9 12 15 18 21 24)
CKPT_ROOT="save_ckpt"
RESULTS_ROOT="save_results"
BASE_DIR="64_preby_mvtec-clinic-12prompt-newloss-v3"

# 4. 需要测试的 Epoch 列表
EPOCHS=(50 52 54 56 48 40 44)

# 5. 数据集列表   'SDD:data/SDD'      'tn3k:data/tn3k'
DATASETS_CONFIG=(
    "RSDD:data/RSDD"
    "DAGM:data/DAGM"
    "KSDD2:data/KSDD2"
    "visa:data/Visa"
    "mpdd:data/MPDD"
    "DTD:data/DTD-Synthetic"
    "btad:data/BTech_Dataset_transformed"
    "headct:data/headct"
    "BrainMRI:data/BrainMRI"
    "br35h:data/br35h"
    "ISBI:data/ISBI"
    "colon:data/CVC-ColonDB"
    "endo:data/EndoTect_2020_Segmentation_Test_Dataset"
)

# ================= 执行逻辑 =================

# 1. 外层循环：Epoch (因为你想把同一个Epoch的结果放在一起)
for epoch in "${EPOCHS[@]}"; do
    
    # 定义该 Epoch 的 checkpoint 路径
    CKPT_PATH="${CKPT_ROOT}/${BASE_DIR}/checkpoint_epoch_${epoch}.pth"
    
    if [ ! -f "$CKPT_PATH" ]; then
        echo "Error: Checkpoint not found at $CKPT_PATH"
        continue
    fi

    # 定义该 Epoch 的总结果保存目录 (例如 save_results/base_dir/epoch_47_results)
    # 这样该 epoch 的所有日志都在这个文件夹下
    EPOCH_SAVE_DIR="${RESULTS_ROOT}/${BASE_DIR}/epoch_${epoch}_results"
    mkdir -p "${EPOCH_SAVE_DIR}"

    # 定义汇总指标日志文件名 (该 epoch 下所有数据集的结果都会写进这一个文件)
    # 例如: save_results/.../epoch_47_results/TOTAL_METRICS_epoch_47_sig10.log
    METRIC_LOG_FILE="TOTAL_METRICS_epoch_${epoch}_sig${SIGMA}_top${topk}-newlossv2-bayespro.log"
    
    # 清空或初始化该日志文件，加上表头说明
    echo "========================================================" > "${EPOCH_SAVE_DIR}/${METRIC_LOG_FILE}"
    echo "Test Results for Epoch ${epoch} (Sigma=${SIGMA})" >> "${EPOCH_SAVE_DIR}/${METRIC_LOG_FILE}"
    echo "Checkpoint: ${CKPT_PATH}" >> "${EPOCH_SAVE_DIR}/${METRIC_LOG_FILE}"
    echo "========================================================" >> "${EPOCH_SAVE_DIR}/${METRIC_LOG_FILE}"

    # 2. 内层循环：Dataset
    for dataset_info in "${DATASETS_CONFIG[@]}"; do
        DATASET_NAME="${dataset_info%%:*}"
        DATA_PATH="${dataset_info#*:}"
        
        # 定义该数据集的详细运行日志 (debug用)
        RUN_LOG="${EPOCH_SAVE_DIR}/run_log_${DATASET_NAME}.log"

        echo "--------------------------------------------------------"
        echo "Running: Epoch ${epoch} | Dataset: ${DATASET_NAME}"
        echo "Result File: ${EPOCH_SAVE_DIR}/${METRIC_LOG_FILE}"
        echo "--------------------------------------------------------"

        CUDA_VISIBLE_DEVICES=${DEVICE_ID} \
        python ${PYTHON_SCRIPT} \
          --dataset "${DATASET_NAME}" \
          --data_path "${DATA_PATH}" \
          --save_path "${EPOCH_SAVE_DIR}" \
          --checkpoint_path "${CKPT_PATH}" \
          --result_log "${METRIC_LOG_FILE}" \
          --features_list "${FEATURES_LIST[@]}" \
          --image_size "${IMAGE_SIZE}" \
          --depth "${DEPTH}" \
          --n_ctx "${N_CTX}" \
          --t_n_ctx "${T_N_CTX}" \
          --topk "${topk}" \
          --sigma "${SIGMA}" \
          --num_prototypes "${NUM_PROTOTYPES}" \
          --prompt_num "${prompt_num}" \
          2>&1 | tee "${RUN_LOG}"
          
    done
done

echo "Done. Results structure:"
echo "${RESULTS_ROOT}/${BASE_DIR}/epoch_<N>_results/TOTAL_METRICS_epoch_<N>_sig${SIGMA}_top${topk}.log"