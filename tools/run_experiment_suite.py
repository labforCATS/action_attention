# this script is to run through all of our training and visualization pipelines
# for all experiments, models, and visualization parameters

from slowfast.config.defaults import get_cfg
import slowfast.utils.checkpoint as cu
from slowfast.utils.misc import launch_job
from test_net import test
from train_net import train
from visualization import visualize

import os
from datetime import datetime
import json
import numpy as np


def run_experiment(cfg):
    """
    Main function to spawn the train and test process. Copied and modified from
    run_net.main()
    """
    default_init_method = "tcp://localhost:9999"

    # We added the line below
    cfg.NUM_GPUS = 1

    # Perform training.
    if cfg.TRAIN.ENABLE:
        print("Starting training")
        start = datetime.now()
        launch_job(cfg=cfg, init_method=default_init_method, func=train)
        print("Training complete, runtime was", datetime.now() - start)
    else:
        print("No training")

    # Perform testing.
    if cfg.TEST.ENABLE:
        print("Starting testing")
        start = datetime.now()
        launch_job(cfg=cfg, init_method=default_init_method, func=test)
        print("Testing complete, runtime was", datetime.now() - start)
    else:
        print("no test")

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        print("Starting visualization")
        start = datetime.now()
        launch_job(cfg=cfg, init_method=default_init_method, func=visualize)
        print("Visualization complete, runtime was", datetime.now() - start)
    else:
        print("no visualize")


def run_all_experiments(server):
    """Runs training and visualization for all permutations of dataset, models,
    and visualization parameters

    params to loop over:
        training:
            experiment
            model

        visualization:
            experiment
            model
            gradcam variant
            pre/post softmax
    """
    pre_trained_slowfast = "/research/cwloka/data/action_attn/synthetic_motion_experiments/pretrained_weights/SLOWFAST_8x8_R50.pkl"
    pre_trained_i3d = "/research/cwloka/data/action_attn/synthetic_motion_experiments/pretrained_weights/I3D_8x8_R50.pkl"
    pre_trained_i3d_nln = "/research/cwloka/data/action_attn/synthetic_motion_experiments/pretrained_weights/I3D_NLN_8x8_R50.pkl"

    experiments = ["1", "2", "3", "4", "5", "5b"]
    num_classes = {"1": 7, "2": 7, "3": 7, "4": 9, "5": 9, "5b": 9}
    vis_techniques = ["gradcam", "gradcam_plusplus", "eigen_cam"]
    post_softmax = [False, True]

    model_dicts = {
        "slowfast": {
            "pre_trained_weights_paths": pre_trained_slowfast,
            "arch": "slowfast",
            "input_channel_num": [3, 3],
            "num_block_temp_kernel": [[3, 3], [4, 4], [6, 6], [3, 3]],
            "spatial_strides": [[1, 1], [2, 2], [2, 2], [2, 2]],
            "spatial_dilations": [[1, 1], [1, 1], [1, 1], [1, 1]],
            "nonlocal_location": [[[], []], [[], []], [[], []], [[], []]],
            "nonlocal_group": [[1, 1], [1, 1], [1, 1], [1, 1]],
            "nonlocal_instantiation": "dot_product",
            "model_name": "SlowFast",
        },
        "i3d": {
            "pre_trained_weights_paths": pre_trained_i3d,
            "arch": "i3d",
            "input_channel_num": [3],
            "num_block_temp_kernel": [[3], [4], [6], [3]],
            "spatial_strides": [[1], [2], [2], [2]],
            "spatial_dilations": [[1], [1], [1], [1]],
            "nonlocal_location": [[[]], [[]], [[]], [[]]],
            "nonlocal_group": [[1], [1], [1], [1]],
            "nonlocal_instantiation": "softmax",
            "model_name": "ResNet",
        },
        "i3d_nln": {
            "pre_trained_weights_paths": pre_trained_i3d_nln,
            "arch": "i3d",
            "input_channel_num": [3],
            "num_block_temp_kernel": [[3], [4], [6], [3]],
            "spatial_strides": [[1], [2], [2], [2]],
            "spatial_dilations": [[1], [1], [1], [1]],
            "nonlocal_location": [[[]], [[1, 3]], [[1, 3, 5]], [[]]],
            "nonlocal_group": [[1], [1], [1], [1]],
            "nonlocal_instantiation": "softmax",
            "model_name": "ResNet",
        },
    }

    if server == "shadowfax":
        model_dicts = {k: model_dicts[k] for k in ["slowfast", "i3d_nln"]}
        batch_size = 16
    elif server == "shuffler":
        model_dicts = {k: model_dicts[k] for k in ["i3d"]}
        batch_size = 10
    else:
        raise NotImplementedError

    ######## TRAINING ########
    # iterate over model
    for model, model_params in model_dicts.items():
        # iterate over experiment
        for exp in experiments:
            # get default config
            cfg = get_cfg()
            # update config with our desired parameters
            cfg.TRAIN.ENABLE = True
            cfg.TRAIN.DATASET = "SyntheticMotion"
            cfg.TRAIN.BATCH_SIZE = (
                batch_size if model == "slowfast" else batch_size / 2
            )  # TODO: check if this fits on shuffler
            cfg.TRAIN.EVAL_PERIOD = 1
            cfg.TRAIN.CHECKPOINT_PERIOD = 1
            cfg.TRAIN.RESUME_FROM_CHECKPOINT = True
            cfg.TRAIN.CHECKPOINT_FILE_PATH = model_params[
                "pre_trained_weights_paths"
            ]
            cfg.TRAIN.CHECKPOINT_TYPE = "caffe2"

            cfg.TENSORBOARD.ENABLE = False
            cfg.TEST.ENABLE = False

            data_dir = f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}"
            cfg.DATA.PATH_TO_DATA_DIR = data_dir
            cfg.DATA.NUM_FRAMES: 32
            cfg.DATA.TRAIN_JITTER_SCALES: [256, 320]
            cfg.DATA.TRAIN_CROP_SIZE: 224
            cfg.DATA.TEST_CROP_SIZE: 256
            cfg.DATA.INPUT_CHANNEL_NUM = model_params["input_channel_num"]

            if model == "slowfast":
                cfg.SLOWFAST.ALPHA = 4
                cfg.SLOWFAST.BETA_INV = 8
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2
                cfg.SLOWFAST.FUSION_KERNEL_SZ = 7
            else:  # model == "i3d" or model == "i3d_nln"
                pass

            cfg.RESNET.ZERO_INIT_FINAL_BN = True
            cfg.RESNET.WIDTH_PER_GROUP = 64
            cfg.RESNET.NUM_GROUPS = 1
            cfg.RESNET.DEPTH = 50
            cfg.RESNET.TRANS_FUNC = "bottleneck_transform"
            cfg.RESNET.STRIDE_1X1 = False
            cfg.RESNET.NUM_BLOCK_TEMP_KERNEL = model_params[
                "num_block_temp_kernel"
            ]
            cfg.RESNET.SPATIAL_STRIDES = model_params["spatial_strides"]
            cfg.RESNET.SPATIAL_DILATIONS = model_params["spatial_dilations"]

            cfg.NONLOCAL.LOCATION = model_params["nonlocal_location"]
            cfg.NONLOCAL.GROUP = model_params["nonlocal_group"]
            cfg.NONLOCAL.INSTANTIATION = model_params["nonlocal_instantiation"]

            cfg.BN.USE_PRECISE_STATS = True
            cfg.BN.NUM_BATCHES_PRECISE = 200

            # TODO: if training does not go well, double check the
            # solver params, since the i3d vs i3d_in1k configs for
            # kinetics had different setups
            cfg.SOLVER.BASE_LR = 0.1
            cfg.SOLVER.LR_POLICY = "cosine"
            cfg.SOLVER.MAX_EPOCH = 100
            cfg.SOLVER.MOMENTUM = 0.9
            cfg.SOLVER.WEIGHT_DECAY = 1e-4
            cfg.SOLVER.WARMUP_EPOCHS = 34.0
            cfg.SOLVER.WARMUP_START_LR = 0.01
            cfg.SOLVER.OPTIMIZING_METHOD = "sgd"

            cfg.MODEL.NUM_CLASSES = num_classes[exp]
            cfg.MODEL.ARCH = model_params["arch"]
            cfg.MODEL.MODEL_NAME = model_params["model_name"]
            cfg.MODEL.LOSS_FUNC = "cross_entropy"
            cfg.MODEL.DROPOUT_RATE = 0.5

            cfg.DATA_LOADER.NUM_WORKERS = 8
            cfg.DATA_LOADER.PIN_MEMORY = True
            cfg.DATA_LOADER.INSPECT.SAVE_SEQ_COUNT = 30
            cfg.DATA_LOADER.INSPECT.SAVE_FRAMES = True
            cfg.DATA_LOADER.INSPECT.SAVE_VIDEO = True
            cfg.DATA_LOADER.INSPECT.SHUFFLE = True
            cfg.DATA_LOADER.NUM_GPUS = 8
            cfg.DATA_LOADER.NUM_SHARDS = 1
            cfg.DATA_LOADER.RNG_SEED = 0

            output_dir = os.path.join(data_dir, f"{model}_output")
            cfg.OUTPUT_DIR = output_dir

            # run training
            run_experiment(cfg)

    ######## VISUALIZATION ########

    # iterate over model
    for model, model_params in model_dicts.items():
        # iterate over experiment
        for exp in experiments:
            # iterate over gradcam variant
            for gradcam_variant in vis_techniques:
                # iterate over pre/post softmax
                for softmax_option in post_softmax:
                    # get default config
                    cfg = get_cfg()
                    # update config with our desired parameters
                    cfg.TRAIN.ENABLE = False

                    cfg.TENSORBOARD.ENABLE = True
                    cfg.TENSORBOARD.CLASS_NAMES_PATH = f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}/synthetic_motion_labels.json"
                    cfg.TENSORBOARD.MODEL_VIS.ENABLE = True
                    cfg.TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS = (
                        False  # Set to True to visualize model weights.
                    )
                    cfg.TENSORBOARD.MODEL_VIS.ACTIVATIONS = (
                        False  # Set to True to visualize feature maps.
                    )
                    cfg.TENSORBOARD.MODEL_VIS.INPUT_VIDEO = False  # Set to True to visualize the input video(s) for the corresponding feature maps.
                    cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE = True
                    cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST = [
                        "s5/pathway1_res2",
                        "s5/pathway0_res2",
                    ]  # List of CNN layers to use for Grad-CAM visualization method.
                    # The number of layer must be equal to the number of pathway(s).
                    cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.USE_TRUE_LABEL = False
                    cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.METHOD = gradcam_variant
                    cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.POST_SOFTMAX = (
                        softmax_option
                    )
                    cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.SOFTMAX_LAYER = (
                        "head/act"
                    )
                    cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.SAVE_OVERLAY_VIDEO = (
                        False
                    )

                    output_dir = os.path.join(data_dir, f"{model}_output")
                    cfg.OUTPUT_DIR = output_dir

                    best_epoch = get_best_epoch(
                        output_dir=output_dir, epochs=100, eval_period=1
                    )

                    cfg.TEST.ENABLE = False
                    cfg.TEST.DATASET = "SyntheticMotion"
                    cfg.TEST.BATCH_SIZE = 1
                    cfg.TEST.CHECKPOINT_FILE_PATH = f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_5/slowfast_outputs/checkpoints/checkpoint_epoch_{best_epoch:05d}.pyth"
                    cfg.TEST.CHECKPOINT_TYPE = "pytorch"
                    # remove extra cropping from testing data
                    cfg.TEST.NUM_ENSEMBLE_VIEWS = 1  # Number of clips to sample from a video uniformly for aggregating the prediction results.
                    cfg.TEST.NUM_SPATIAL_CROPS = 1  # Number of crops to sample from a frame spatially for aggregating the prediction results.

                    data_dir = f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}"
                    cfg.DATA.PATH_TO_DATA_DIR = data_dir
                    cfg.DATA.NUM_FRAMES: 32
                    cfg.DATA.TRAIN_JITTER_SCALES: [256, 320]
                    cfg.DATA.TRAIN_CROP_SIZE: 224
                    cfg.DATA.TEST_CROP_SIZE: 256
                    cfg.DATA.INPUT_CHANNEL_NUM = model_params[
                        "input_channel_num"
                    ]

                    if model == "slowfast":
                        cfg.SLOWFAST.ALPHA = 4
                        cfg.SLOWFAST.BETA_INV = 8
                        cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2
                        cfg.SLOWFAST.FUSION_KERNEL_SZ = 7
                    else:  # model == "i3d" or model == "i3d_nln"
                        pass

                    cfg.RESNET.ZERO_INIT_FINAL_BN = True
                    cfg.RESNET.WIDTH_PER_GROUP = 64
                    cfg.RESNET.NUM_GROUPS = 1
                    cfg.RESNET.DEPTH = 50
                    cfg.RESNET.TRANS_FUNC = "bottleneck_transform"
                    cfg.RESNET.STRIDE_1X1 = False
                    cfg.RESNET.NUM_BLOCK_TEMP_KERNEL = model_params[
                        "num_block_temp_kernel"
                    ]
                    cfg.RESNET.SPATIAL_STRIDES = model_params["spatial_strides"]
                    cfg.RESNET.SPATIAL_DILATIONS = model_params[
                        "spatial_dilations"
                    ]

                    cfg.NONLOCAL.LOCATION = model_params["nonlocal_location"]
                    cfg.NONLOCAL.GROUP = model_params["nonlocal_group"]
                    cfg.NONLOCAL.INSTANTIATION = model_params[
                        "nonlocal_instantiation"
                    ]

                    cfg.BN.USE_PRECISE_STATS = True
                    cfg.BN.NUM_BATCHES_PRECISE = 200

                    cfg.MODEL.NUM_CLASSES = num_classes[exp]
                    cfg.MODEL.ARCH = model_params["arch"]
                    cfg.MODEL.MODEL_NAME = model_params["model_name"]
                    cfg.MODEL.LOSS_FUNC = "cross_entropy"
                    cfg.MODEL.DROPOUT_RATE = 0.5

                    cfg.DATA_LOADER.NUM_WORKERS = 8
                    cfg.DATA_LOADER.PIN_MEMORY = True
                    cfg.DATA_LOADER.INSPECT.SAVE_SEQ_COUNT = 30
                    cfg.DATA_LOADER.INSPECT.SAVE_FRAMES = True
                    cfg.DATA_LOADER.INSPECT.SAVE_VIDEO = True
                    cfg.DATA_LOADER.INSPECT.SHUFFLE = True
                    cfg.DATA_LOADER.NUM_GPUS = 8
                    cfg.DATA_LOADER.NUM_SHARDS = 1
                    cfg.DATA_LOADER.RNG_SEED = 0

                    # run visualization
                    run_experiment(cfg)


def get_best_epoch(output_dir, epochs, eval_period):
    """identify the epoch from training w the best validation accuracy"""

    # verify that output log of training stats exists
    stats_path = os.path.join(output_dir, "json_stats.log")
    assert os.path.exists(
        stats_path
    ), f"training did not start properly, no stats file found"

    # compile list of validation accuracies for each epoch
    val_accs = []

    with open(stats_path, "r") as f:
        for line in f.readlines():
            dict_str = line.rstrip().removeprefix("json_stats: ")
            stats_dict = json.loads(dict_str)
            mode = stats_dict["_type"].removesuffix("_epoch")
            loss = float(stats_dict["loss"])
            acc = 1.0 - float(stats_dict["top1_err"])

            if mode == "val":
                val_accs.append(acc)

    # verify that training is complete, i.e. that the number of validation
    # epochs matches the number of epochs declared in the config
    assert len(val_accs) == epochs // eval_period, f"training was incomplete"

    # retrieve the best epoch
    best_epoch = np.argmax(val_accs)

    return best_epoch


if __name__ == "__main__":
    run_all_experiments(server="shadowfax")
