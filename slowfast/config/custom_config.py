#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    # (added 3/30): specific gradcam method to use for visualization.
    _C.TENSORBOARD.MODEL_VIS.GRAD_CAM.METHOD = "grad_cam"
    # (added 6/5): whether to save input video to model
    _C.TRAIN.SAVE_INPUT_VIDEO = False
    _C.TEST.SAVE_INPUT_VIDEO = False

    # (added 6/5): directory to save inputs right before they get passed in
    # to the model
    _C.VIS_MODEL_INPUT_DIR = "/research/cwloka/projects/nikki_sandbox/action_attention/vis_model_input"

