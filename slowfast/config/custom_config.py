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

    # -----------------------------------------------------------------------------
    # Options for saving dataloader outputs
    # -----------------------------------------------------------------------------
    # (added 7/5):
    # _C.DATA_LOADER.INSPECT = CfgNode()
    # TODO: figure out why these defaults aren't working
    # whether to save sampled videos as frames
    _C.DATA_LOADER.INSPECT.SAVE_FRAMES: False
    # whether to save sampled videos as videos
    _C.DATA_LOADER.INSPECT.SAVE_VIDEO: False
    # how many sampled videos from a dataloader to save
    _C.DATA_LOADER.INSPECT.SAVE_SEQ_COUNT: 0
    # whether to shuffle the dataloader before iterating over it
    _C.DATA_LOADER.INSPECT.SHUFFLE: True
