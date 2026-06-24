# DDStereo
DDStereo: Efficient Dual Decoder Transformers for Stereo 3D Road Anomaly Detection

Arxiv：http://arxiv.org/abs/2606.24805

Final version was accepted by ECCV26

## Environment Configuration
We refer to the environment of `monodetr`.
```bash
conda activate monodetr
```

## Data Preparation
### Generate Disparity Label
```bash
python tools/get_disp_label.py
```

## Training
```bash
bash scripts/101-a-openset-DDStereo_for_kittiAR_train.sh
```

## Evaluation
```bash
bash scripts/101-a-openset-DDStereo_for_kittiAR_val.sh
```
