#!/bin/bash
# 光照鲁棒性测试脚本
# 逐渐降低光照亮度，评估检测性能

export CUDA_HOME=/usr/local/cuda-11.1/

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate monodetr

cd /data3/mushiyi/smb9_msy/02-Code/05-stereo-detr/DDStereo-CVPR

# ======== 配置选项 ========
CONFIG="configs/101-OoD-arind-train-ood-val.yaml"
CHECKPOINT="outputs/101-OoD-arind-train-ood-val/lightstereoOoDdetr/checkpoint_best.pth"

# 测试模式: gamma / brightness / both
# brightness: factor*pixel, factor<1 变暗, 最直观
# gamma: pixel^gamma, gamma>1 变暗
ADJUST_MODE="brightness"

# Gamma 测试参数 (gamma>1 变暗)
GAMMA_MIN=1.0
GAMMA_MAX=5.0
GAMMA_STEP=0.2

# 亮度测试参数 (factor 从 1.0->0.1 逐渐变暗)
BRIGHTNESS_MIN=0.1
BRIGHTNESS_MAX=1.0
BRIGHTNESS_STEP=0.2

# 输出目录
OUTPUT_DIR="outputs/illumination_robustness"

# ======== 运行测试 ========
echo "============================================"
echo "Starting Illumination Robustness Test"
echo "Mode: ${ADJUST_MODE}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python tools/test_illumination_robustness.py \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --adjust_mode ${ADJUST_MODE} \
    --gamma_min ${GAMMA_MIN} \
    --gamma_max ${GAMMA_MAX} \
    --gamma_step ${GAMMA_STEP} \
    --brightness_min ${BRIGHTNESS_MIN} \
    --brightness_max ${BRIGHTNESS_MAX} \
    --brightness_step ${BRIGHTNESS_STEP} \
    --output_dir ${OUTPUT_DIR}

echo "============================================"
echo "Test completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================"
