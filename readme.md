# DDStereo
DDStereo: Efficient Dual Decoder Transformers for Open-set Stereo 3D Object Detection

# eval
```
conda activate monodetr
bash scripts/001-a-closeset-DDStereo_for_valsplit_BM_val.sh
```

# train
## argoverse
### generate disp label
```bash
python tool/get_disp_label.py
```