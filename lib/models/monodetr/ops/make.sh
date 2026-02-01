#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
# 1. 强制将 PATH 指向 CUDA 11.1
export PATH=/usr/local/cuda-11.1/bin:$PATH

# 2. 强制库路径指向 CUDA 11.1
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH

# 3. 再次确认 CUDA_HOME
export CUDA_HOME=/usr/local/cuda-11.1

python setup.py build install