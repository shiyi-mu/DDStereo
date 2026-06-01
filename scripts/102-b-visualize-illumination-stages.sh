#!/bin/bash
# 可视化光照各阶段的输入图像

export CUDA_HOME=/usr/local/cuda-11.1/
source $(conda info --base)/etc/profile.d/conda.sh
conda activate monodetr

cd /data3/mushiyi/smb9_msy/02-Code/05-stereo-detr/DDStereo-CVPR

CONFIG="configs/101-OoD-arind-train-ood-val.yaml"
OUTPUT_DIR="outputs/illumination_visualization"

# 默认随机采样6张图像，展示 brightness 从1.0到0.01的变化
CUDA_VISIBLE_DEVICES=0 python tools/visualize_illumination_stages.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --adjust_mode brightness \
    --num_samples 6

echo "============================================"
echo "Visualization completed!"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
