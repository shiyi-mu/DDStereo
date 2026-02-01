export CUDA_HOME=/usr/local/cuda-11.1/

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_val.py --config configs/201-b-closeset-DDStereo_for-argo-s19-50HD-s19.yaml