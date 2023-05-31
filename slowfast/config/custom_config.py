#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    # (added 3/30): specific gradcam method to use for visualization.
    _C.TENSORBOARD.MODEL_VIS.GRAD_CAM.METHOD = "grad_cam"
