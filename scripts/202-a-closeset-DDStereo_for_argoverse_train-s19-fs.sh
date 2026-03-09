export CUDA_HOME=/usr/local/cuda-11.1/

CUDA_VISIBLE_DEVICES=1,2 python tools/train_val.py --config configs/202-a-closeset-DDStereo_for-argo-s19-50HD-s19-foundationstereo.yaml