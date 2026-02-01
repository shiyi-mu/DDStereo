export CUDA_HOME=/usr/local/cuda-11.1/

CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config configs/201-a-closeset-DDStereo_for-argo-s19-50HD.yaml