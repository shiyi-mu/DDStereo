export CUDA_HOME=/usr/local/cuda-11.1/

CUDA_VISIBLE_DEVICES=3 python tools/train_val.py --config configs/301-OoD-arind-train-ood-val-align.yaml
