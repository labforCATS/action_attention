# this script generates config files for training, testing, and visualizing
# each of our experiments

# to rerun the metrics pipeline with new metrics, look at lines 575, 599, 686 to change pathing and METRIC FUNCS
# the new metrics themselves are calculated in metrics.py
# as of 7/25/24, old metric CSVs (before calculating precision and recall) are in data -> backup_synthetic_metric_results

import yaml
import os
import pdb
import copy
import json
import numpy as np
import argparse


def get_best_epoch(output_dir, epochs, eval_period):
    """identify the epoch from training w the best validation accuracy.

    NOTE: epochs are 1-indexed.
    """

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
    print(len(val_accs))
    print(epochs // eval_period)
    assert len(val_accs) == epochs // eval_period, f"training was incomplete"

    # retrieve the best epoch
    best_epoch = np.argmax(val_accs)

    # checkpoint epoch labels are 1-indexed
    return best_epoch + 1


def get_nonzero_epoch(model_name: str, experiment_num: int) -> int:
    """Get nonzero epoch to pull weights from for visualization

    Args:
        model_name (str): name of the model, one of "slowfast", "i3d", or "i3d_nln"
        experiment_num (int): number of experiment, 1-6. Note experiment 5b is experiment 6

    Returns:
        int: predetermined epoch to pull weights from
    """
    assert model_name in ["slowfast", "i3d", "i3d_nln"]
    assert experiment_num >= 1 and experiment_num <= 6

    specific_epochs = {
            1 : {"slowfast": 20, "i3d": 50, "i3d_nln": 50},
            2 : {"slowfast": 10, "i3d": 20, "i3d_nln": 30}, # nln is a mess here, so may need to change
            3 : {"slowfast": 25, "i3d": 30, "i3d_nln": 50},
            4 : {"slowfast": 75, "i3d": 80, "i3d_nln": 90},
            5 : {"slowfast": 80, "i3d": 70, "i3d_nln": 1}, # TODO: nln is a mess here, pick a real epoch
            6 : {"slowfast": 70, "i3d": 80, "i3d_nln": 90},
        }
    
    return specific_epochs[experiment_num][model_name]

def generate_all_configs(use_specific_epoch: bool = True):
    """
    Generates config files for training, testing, and visualizing for each
    of our experiments. Adds a new subfolder to each of the experiments folders
    containing the config files
    """
    use_specific_epoch = True
    # naming convention for the config files:
    #   train/test/vis, exp num, architecture, grad cam variant, pre-post softmax
    #   exp[num]_arch_gcv_prepost.yaml <-- sample

    # training: 6 experiments, 3 model architectures
    # testing: 6 experiments, 3 model architectures
    # visualization: 6 experiments, 3 model architectures, two
    #   options for when to take the gradients, 3 GradCAM variants

    model_names = ["slowfast", "i3d", "i3d_nln"]
    experiments = ["1", "2", "3", "4", "5", "5b"]
    num_classes = {"1": 7, "2": 7, "3": 7, "4": 7, "5": 7, "5b": 9}
    gradcam_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
    post_softmax = [False, True]

    # location of pre-trained weights for each network
    pre_trained_slowfast = "/research/cwloka/data/action_attn/synthetic_motion_experiments/pretrained_weights/SLOWFAST_8x8_R50.pkl"
    pre_trained_i3d = "/research/cwloka/data/action_attn/synthetic_motion_experiments/pretrained_weights/I3D_8x8_R50.pkl"
    pre_trained_i3d_nln = "/research/cwloka/data/action_attn/synthetic_motion_experiments/pretrained_weights/I3D_NLN_8x8_R50.pkl"


    # dictionary of config file information for each of the networks
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
            "gradcam_layer_list": ["s5/pathway1_res2", "s5/pathway0_res2"],
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
            "gradcam_layer_list": ["s5/pathway0_res2"],
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
            "gradcam_layer_list": ["s5/pathway0_res2"],
        },
    }

    # define parameters that are constant throughout all config files

    # slowfast params
    slowfast_params = {
        "ALPHA": 4,
        "BETA_INV": 8,
        "FUSION_CONV_CHANNEL_RATIO": 2,
        "FUSION_KERNEL_SZ": 7,
    }

    # solver params
    solver_params = {
        "BASE_LR": 0.1,
        "LR_POLICY": "cosine",
        "MAX_EPOCH": 100,
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 1e-4,
        "WARMUP_EPOCHS": 34.0,
        "WARMUP_START_LR": 0.01,
        "OPTIMIZING_METHOD": "sgd",
    }

    # batchnorm params
    batchnorm_params = {
        "USE_PRECISE_STATS": True,
        "NUM_BATCHES_PRECISE": 200,
    }

    # dataloader params
    dataloader_params = {
        "NUM_WORKERS": 8,
        "PIN_MEMORY": True,
        "INSPECT": {
            "SAVE_SEQ_COUNT": 30,
            "SAVE_FRAMES": True,
            "SAVE_VIDEO": True,
            "SHUFFLE": True,
        },
    }

    ### commented out training/vis on 7/22/24 for metrics rerun

    # # generate training config
    # for model, model_params_original in model_dicts.items():
    #     # iterate over experiment
    #     for exp in experiments:
    #         # there is an unresolved issue where the entries of
    #         # model params kept getting deleted, so solution
    #         # was to use a deepcopy
    #         model_params = copy.deepcopy(model_params_original)

    #         # directory to where the data is located
    #         data_dir = f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}"
    #         output_dir = os.path.join(data_dir, f"{model}_output")

    #         train_params = {
    #             "ENABLE": True,
    #             "DATASET": "SyntheticMotion",
    #             "BATCH_SIZE": 7,
    #             "EVAL_PERIOD": 1,
    #             "CHECKPOINT_PERIOD": 1,
    #             "RESUME_FROM_CHECKPOINT": True,
    #             "CHECKPOINT_FILE_PATH": model_params["pre_trained_weights_paths"],
    #             "CHECKPOINT_TYPE": "caffe2",
    #         }

    #         test_params = {"ENABLE": False}

    #         data_params = {
    #             "PATH_TO_DATA_DIR": data_dir,
    #             "NUM_FRAMES": 32,
    #             "TRAIN_JITTER_SCALES": [256, 320],
    #             "TRAIN_CROP_SIZE": 224,
    #             "TEST_CROP_SIZE": 256,
    #             "INPUT_CHANNEL_NUM": model_params["input_channel_num"],
    #         }
    #         tensorboard_params = {
    #             "ENABLE": True,
    #             "CLASS_NAMES_PATH": f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}/synthetic_motion_labels.json",
    #             "MODEL_VIS": {
    #                 "ENABLE": False,
    #                 "MODEL_WEIGHTS": False,
    #                 "ACTIVATIONS": False,
    #                 "INPUT_VIDEO": False,
    #                 "GRAD_CAM": {
    #                     "ENABLE": False,
    #                     "LAYER_LIST": model_params["gradcam_layer_list"],
    #                     "SOFTMAX_LAYER": "head/act",
    #                     "SAVE_OVERLAY_VIDEO": False,
    #                 },
    #             },
    #             "GRAD_CAM": {
    #                 "USE_TRUE_LABEL": False,
    #                 # "METHOD": gradcam_variant,
    #                 # "POST_SOFTMAX": softmax_option,
    #             },
    #         }

    #         resnet_params = {
    #             "ZERO_INIT_FINAL_BN": True,
    #             "WIDTH_PER_GROUP": 64,
    #             "NUM_GROUPS": 1,
    #             "DEPTH": 50,
    #             "TRANS_FUNC": "bottleneck_transform",
    #             "STRIDE_1X1": False,
    #             "NUM_BLOCK_TEMP_KERNEL": model_params["num_block_temp_kernel"],
    #             "SPATIAL_STRIDES": model_params["spatial_strides"],
    #             "SPATIAL_DILATIONS": model_params["spatial_dilations"],
    #         }

    #         nonlocal_params = {
    #             "LOCATION": model_params["nonlocal_location"],
    #             "GROUP": model_params["nonlocal_group"],
    #             "INSTANTIATION": model_params["nonlocal_instantiation"],
    #         }

    #         model_params = {
    #             "NUM_CLASSES": num_classes[exp],
    #             "ARCH": model_params["arch"],
    #             "MODEL_NAME": model_params["model_name"],
    #             "LOSS_FUNC": "cross_entropy",
    #             "DROPOUT_RATE": 0.5,
    #         }

    #         # combine all the params in a single dictionary
    #         cfg_dict = {
    #             "TRAIN": train_params,
    #             "TEST": test_params,
    #             "TENSORBOARD": tensorboard_params,
    #             "DATA": data_params,
    #             "SLOWFAST": slowfast_params,
    #             "RESNET": resnet_params,
    #             "NONLOCAL": nonlocal_params,
    #             "BN": batchnorm_params,
    #             "MODEL": model_params,
    #             "DATA_LOADER": dataloader_params,
    #             "SOLVER": solver_params,
    #             "NUM_GPUS": 8,
    #             "NUM_SHARDS": 1,
    #             "RNG_SEED": 0,
    #             "OUTPUT_DIR": output_dir,
    #         }

    #         # set up folder to save config files to
    #         config_dir = os.path.join(data_dir, "configs")
    #         if not os.path.isdir(config_dir):
    #             os.makedirs(config_dir)

    #         # turn the dictionary into a config file and save it
    #         config_filename = os.path.join(config_dir, f"train_exp{exp}_{model}.yaml")
    #         with open(config_filename, "w") as f:
    #             yaml.dump(cfg_dict, f)

    # # generate visualization configs
    # for model, model_params_original in model_dicts.items():
    #     # iterate over experiment
    #     for exp in experiments:
    #         data_dir = f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}"
    #         output_dir = os.path.join(data_dir, f"{model}_output")
    #         print("model:", model)
    #         print("experiment:", exp)

    #         if use_specific_epoch:
    #             exp_num = 6 if exp=="5b" else int(exp)
    #             epoch = get_nonzero_epoch(model_name=model, experiment_num=exp_num)
    #         else:
    #             epoch = get_best_epoch(
    #                 output_dir=output_dir, epochs=100, eval_period=1
    #             )

    #         epoch_path = os.path.join(
    #             output_dir,
    #             f"checkpoints/checkpoint_epoch_{epoch:05d}.pyth",
    #         )

    #         # iterate over gradcam variant
    #         for gradcam_variant in gradcam_variants:
    #             # iterate over pre/post softmax
    #             for softmax_option in post_softmax:
    #                 model_params = copy.deepcopy(model_params_original)

    #                 train_params = {
    #                     "ENABLE": False,
    #                     "DATASET": "SyntheticMotion",
    #                     "BATCH_SIZE": 7,
    #                     "EVAL_PERIOD": 1,
    #                     "CHECKPOINT_PERIOD": 1,
    #                     "RESUME_FROM_CHECKPOINT": True,
    #                     "CHECKPOINT_FILE_PATH": model_params[
    #                         "pre_trained_weights_paths"
    #                     ],
    #                     "CHECKPOINT_TYPE": "caffe2",
    #                 }

    #                 test_params = {
    #                     "ENABLE": False,
    #                     "DATASET": "SyntheticMotion",
    #                     "BATCH_SIZE": 1,
    #                     "NUM_ENSEMBLE_VIEWS": 1,
    #                     "NUM_SPATIAL_CROPS": 1,
    #                     "CHECKPOINT_FILE_PATH": epoch_path,
    #                 }

    #                 data_params = {
    #                     "PATH_TO_DATA_DIR": data_dir,
    #                     "NUM_FRAMES": 32,
    #                     "TRAIN_JITTER_SCALES": [256, 320],
    #                     "TRAIN_CROP_SIZE": 224,
    #                     "TEST_CROP_SIZE": 256,
    #                     "INPUT_CHANNEL_NUM": model_params["input_channel_num"],
    #                 }
    #                 tensorboard_params = {
    #                     "ENABLE": True,
    #                     "CLASS_NAMES_PATH": f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}/synthetic_motion_labels.json",
    #                     "MODEL_VIS": {
    #                         "ENABLE": True,
    #                         "MODEL_WEIGHTS": False,
    #                         "ACTIVATIONS": False,
    #                         "INPUT_VIDEO": False,
    #                         "GRAD_CAM": {
    #                             "ENABLE": True,
    #                             "LAYER_LIST": model_params["gradcam_layer_list"],
    #                             "POST_SOFTMAX": softmax_option,
    #                             "METHOD": gradcam_variant,
    #                             "SOFTMAX_LAYER": "head/act",
    #                             "SAVE_OVERLAY_VIDEO": False,
    #                         },
    #                     },
    #                 }

    #                 resnet_params = {
    #                     "ZERO_INIT_FINAL_BN": True,
    #                     "WIDTH_PER_GROUP": 64,
    #                     "NUM_GROUPS": 1,
    #                     "DEPTH": 50,
    #                     "TRANS_FUNC": "bottleneck_transform",
    #                     "STRIDE_1X1": False,
    #                     "NUM_BLOCK_TEMP_KERNEL": model_params["num_block_temp_kernel"],
    #                     "SPATIAL_STRIDES": model_params["spatial_strides"],
    #                     "SPATIAL_DILATIONS": model_params["spatial_dilations"],
    #                 }

    #                 nonlocal_params = {
    #                     "LOCATION": model_params["nonlocal_location"],
    #                     "GROUP": model_params["nonlocal_group"],
    #                     "INSTANTIATION": model_params["nonlocal_instantiation"],
    #                 }

    #                 model_params = {
    #                     "NUM_CLASSES": num_classes[exp],
    #                     "ARCH": model_params["arch"],
    #                     "MODEL_NAME": model_params["model_name"],
    #                     "LOSS_FUNC": "cross_entropy",
    #                     "DROPOUT_RATE": 0.5,
    #                 }
    #                 metric_params = {
    #                     "FUNCS": ["kl_div", "mse", "covariance", "pearson", "iou"],
    #                     "ENABLE": False,
    #                 }

    #                 # combine all the params in a single dictionary
    #                 cfg_dict = {
    #                     "TRAIN": train_params,
    #                     "TEST": test_params,
    #                     "TENSORBOARD": tensorboard_params,
    #                     "DATA": data_params,
    #                     "SLOWFAST": slowfast_params,
    #                     "RESNET": resnet_params,
    #                     "NONLOCAL": nonlocal_params,
    #                     "BN": batchnorm_params,
    #                     "MODEL": model_params,
    #                     "DATA_LOADER": dataloader_params,
    #                     "SOLVER": solver_params,
    #                     "NUM_GPUS": 8,
    #                     "NUM_SHARDS": 1,
    #                     "RNG_SEED": 0,
    #                     "OUTPUT_DIR": output_dir,
    #                     "METRICS": metric_params,
    #                 }
    #                 # set up folder to save config files to
    #                 config_dir = os.path.join(data_dir, "configs")
    #                 if not os.path.isdir(config_dir):
    #                     os.makedirs(config_dir)

    #                 # turn the dictionary into a config file and save it
    #                 pre_post_softmax = "post"
    #                 if softmax_option == False:
    #                     pre_post_softmax = "pre"

    #                 config_filename = os.path.join(
    #                     config_dir,
    #                     f"vis_exp{exp}_{model}_{gradcam_variant}_{pre_post_softmax}softmax.yaml",
    #                 )
    #                 with open(config_filename, "w") as f:
    #                     yaml.dump(cfg_dict, f)
    #                 # train/test/vis, exp num, architecture, grad cam variant, pre-post softmax

    # # generate testing configs
    # for model, model_params_original in model_dicts.items():
    #     # iterate over experiment
    #     for exp in experiments:
    #         data_dir = f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}"
    #         output_dir = os.path.join(data_dir, f"{model}_output")
    #         print("model:", model)
    #         print("experiment:", exp)
    #         best_epoch = get_best_epoch(
    #             output_dir=output_dir, epochs=100, eval_period=1
    #         )
    #         best_epoch_path = os.path.join(
    #             output_dir,
    #             f"checkpoints/checkpoint_epoch_{best_epoch:05d}.pyth",
    #         )
    #         model_params = copy.deepcopy(model_params_original)

    #         train_params = {
    #             "ENABLE": False,
    #             "DATASET": "SyntheticMotion",
    #             "BATCH_SIZE": 7,
    #             "EVAL_PERIOD": 1,
    #             "CHECKPOINT_PERIOD": 1,
    #             "RESUME_FROM_CHECKPOINT": True,
    #             "CHECKPOINT_FILE_PATH": model_params["pre_trained_weights_paths"],
    #             "CHECKPOINT_TYPE": "caffe2",
    #         }

    #         test_params = {
    #             "ENABLE": True,
    #             "DATASET": "SyntheticMotion",
    #             "BATCH_SIZE": 1,
    #             "NUM_ENSEMBLE_VIEWS": 1,
    #             "NUM_SPATIAL_CROPS": 1,
    #             "SAVE_RESULTS_PATH": "test_results.pkl",
    #         }

    #         data_params = {
    #             "PATH_TO_DATA_DIR": data_dir,
    #             "NUM_FRAMES": 32,
    #             "TRAIN_JITTER_SCALES": [256, 320],
    #             "TRAIN_CROP_SIZE": 224,
    #             "TEST_CROP_SIZE": 256,
    #             "INPUT_CHANNEL_NUM": model_params["input_channel_num"],
    #         }
    #         tensorboard_params = {
    #             "ENABLE": True,
    #             "CLASS_NAMES_PATH": f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}/synthetic_motion_labels.json",
    #             "MODEL_VIS": {
    #                 "ENABLE": False,
    #                 "MODEL_WEIGHTS": False,
    #                 "ACTIVATIONS": False,
    #                 "INPUT_VIDEO": False,
    #                 "GRAD_CAM": {
    #                     "ENABLE": False,
    #                     "LAYER_LIST": model_params["gradcam_layer_list"],
    #                     "SOFTMAX_LAYER": "head/act",
    #                     "SAVE_OVERLAY_VIDEO": False,
    #                 },
    #             },
    #             "GRAD_CAM": {
    #                 "USE_TRUE_LABEL": False,
    #                 # "METHOD": gradcam_variant,
    #                 # "POST_SOFTMAX": softmax_option,
    #             },
    #         }

    #         resnet_params = {
    #             "ZERO_INIT_FINAL_BN": True,
    #             "WIDTH_PER_GROUP": 64,
    #             "NUM_GROUPS": 1,
    #             "DEPTH": 50,
    #             "TRANS_FUNC": "bottleneck_transform",
    #             "STRIDE_1X1": False,
    #             "NUM_BLOCK_TEMP_KERNEL": model_params["num_block_temp_kernel"],
    #             "SPATIAL_STRIDES": model_params["spatial_strides"],
    #             "SPATIAL_DILATIONS": model_params["spatial_dilations"],
    #         }

    #         nonlocal_params = {
    #             "LOCATION": model_params["nonlocal_location"],
    #             "GROUP": model_params["nonlocal_group"],
    #             "INSTANTIATION": model_params["nonlocal_instantiation"],
    #         }

    #         model_params = {
    #             "NUM_CLASSES": num_classes[exp],
    #             "ARCH": model_params["arch"],
    #             "MODEL_NAME": model_params["model_name"],
    #             "LOSS_FUNC": "cross_entropy",
    #             "DROPOUT_RATE": 0.5,
    #         }

    #         # combine all the params in a single dictionary
    #         cfg_dict = {
    #             "TRAIN": train_params,
    #             "TEST": test_params,
    #             "TENSORBOARD": tensorboard_params,
    #             "DATA": data_params,
    #             "SLOWFAST": slowfast_params,
    #             "RESNET": resnet_params,
    #             "NONLOCAL": nonlocal_params,
    #             "BN": batchnorm_params,
    #             "MODEL": model_params,
    #             "DATA_LOADER": dataloader_params,
    #             "SOLVER": solver_params,
    #             "NUM_GPUS": 8,
    #             "NUM_SHARDS": 1,
    #             "RNG_SEED": 0,
    #             "OUTPUT_DIR": output_dir,
    #         }

    #         # set up folder to save config files to
    #         config_dir = os.path.join(data_dir, "configs")
    #         if not os.path.isdir(config_dir):
    #             os.makedirs(config_dir)

    #         # turn the dictionary into a config file and save it
    #         config_filename = os.path.join(config_dir, f"test_exp{exp}_{model}.yaml")
    #         with open(config_filename, "w") as f:
    #             yaml.dump(cfg_dict, f)

    # metric generation
    for model, model_params_original in model_dicts.items():
        # iterate over experiment
        for exp in experiments:
            data_dir = f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}"
            config_root_dir = f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}"
            # ^^ change here to avoid rewriting config files if rerunning metrics pipeline
            output_dir = os.path.join(data_dir, f"{model}_output")
            print("model:", model, "experiment: ", exp)
            if use_specific_epoch:
                exp_num = 6 if exp=="5b" else int(exp)
                epoch = get_nonzero_epoch(model_name=model, experiment_num=exp_num)
                print("using specific epoch ", epoch)
            else:
                epoch = get_best_epoch(
                    output_dir=output_dir, epochs=100, eval_period=1
                )

            epoch_path = os.path.join(
                output_dir,
                f"checkpoints/checkpoint_epoch_{epoch:05d}.pyth",
            )

            # iterate over gradcam variant
            for gradcam_variant in gradcam_variants:
                # iterate over pre/post softmax
                for softmax_option in post_softmax:
                    
                    model_params = copy.deepcopy(model_params_original)
                    csv_output_folder = f"/research/cwloka/data/action_attn/synthetic_motion_experiments/metric_results/experiment_{exp}/{model}/{gradcam_variant}"
                    # ^^ change here to avoid overwriting CSV files if rerunning metrics pipeline
                    if not os.path.exists(csv_output_folder):
                        os.makedirs(csv_output_folder)
                    pre_post_softmax = "post"
                    if softmax_option == False:
                        pre_post_softmax = "pre"
                    csv_output_path = os.path.join(csv_output_folder, f"exp_{exp}_{model}_{pre_post_softmax}_softmax.csv")
                
                    
                    train_params = {
                        "ENABLE": False,
                        "DATASET": "SyntheticMotion",
                        "BATCH_SIZE": 7,
                        "EVAL_PERIOD": 1,
                        "CHECKPOINT_PERIOD": 1,
                        "RESUME_FROM_CHECKPOINT": True,
                        "CHECKPOINT_FILE_PATH": model_params[
                            "pre_trained_weights_paths"
                        ],
                        "CHECKPOINT_TYPE": "caffe2",
                    }

                    test_params = {
                        "ENABLE": False,
                        "DATASET": "SyntheticMotion",
                        "BATCH_SIZE": 20,
                        "NUM_ENSEMBLE_VIEWS": 1,
                        "NUM_SPATIAL_CROPS": 1,
                        "SAVE_RESULTS_PATH": "test_results.pkl",
                        "CHECKPOINT_FILE_PATH": epoch_path,
                    }

                    data_params = {
                        "PATH_TO_DATA_DIR": data_dir,
                        "NUM_FRAMES": 32,
                        "TRAIN_JITTER_SCALES": [256, 320],
                        "TRAIN_CROP_SIZE": 224,
                        "TEST_CROP_SIZE": 256,
                        "INPUT_CHANNEL_NUM": model_params["input_channel_num"],
                    }
                    tensorboard_params = {
                        "ENABLE": False,
                        "CLASS_NAMES_PATH": f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}/synthetic_motion_labels.json",
                        "MODEL_VIS": {
                            "ENABLE": False,
                            "MODEL_WEIGHTS": False,
                            "ACTIVATIONS": False,
                            "INPUT_VIDEO": False,
                            "GRAD_CAM": {
                                "ENABLE": False,
                                "LAYER_LIST": model_params["gradcam_layer_list"],
                                "POST_SOFTMAX": softmax_option,
                                "METHOD": gradcam_variant,
                                "SOFTMAX_LAYER": "head/act",
                                "SAVE_OVERLAY_VIDEO": False,
                            },
                        },
                    }

                    resnet_params = {
                        "ZERO_INIT_FINAL_BN": True,
                        "WIDTH_PER_GROUP": 64,
                        "NUM_GROUPS": 1,
                        "DEPTH": 50,
                        "TRANS_FUNC": "bottleneck_transform",
                        "STRIDE_1X1": False,
                        "NUM_BLOCK_TEMP_KERNEL": model_params["num_block_temp_kernel"],
                        "SPATIAL_STRIDES": model_params["spatial_strides"],
                        "SPATIAL_DILATIONS": model_params["spatial_dilations"],
                    }

                    nonlocal_params = {
                        "LOCATION": model_params["nonlocal_location"],
                        "GROUP": model_params["nonlocal_group"],
                        "INSTANTIATION": model_params["nonlocal_instantiation"],
                    }

                    model_params = {
                        "NUM_CLASSES": num_classes[exp],
                        "ARCH": model_params["arch"],
                        "MODEL_NAME": model_params["model_name"],
                        "LOSS_FUNC": "cross_entropy",
                        "DROPOUT_RATE": 0.5,
                    }

                    metric_params = {
                        "FUNCS": ["kl_div", "iou", "pearson", "mse", "covariance", "precision", "recall"], # change here if changing metrics to run!
                        "ENABLE": True,
                        "CSV_PATH": csv_output_path
                    }

                    # combine all the params in a single dictionary
                    cfg_dict = {
                        "TRAIN": train_params,
                        "TEST": test_params,
                        "TENSORBOARD": tensorboard_params,
                        "DATA": data_params,
                        "SLOWFAST": slowfast_params,
                        "RESNET": resnet_params,
                        "NONLOCAL": nonlocal_params,
                        "BN": batchnorm_params,
                        "MODEL": model_params,
                        "DATA_LOADER": dataloader_params,
                        "SOLVER": solver_params,
                        "NUM_GPUS": 8,
                        "NUM_SHARDS": 1,
                        "RNG_SEED": 0,
                        "OUTPUT_DIR": output_dir,
                        "METRICS": metric_params,
                    }
                    # set up folder to save config files to
                    config_dir = os.path.join(config_root_dir, "configs")
                    if not os.path.isdir(config_dir):
                        os.makedirs(config_dir)

                    # turn the dictionary into a config file and save it
                    pre_post_softmax = "post"
                    if softmax_option == False:
                        pre_post_softmax = "pre"

                    config_filename = os.path.join(
                        config_dir,
                        f"metrics_exp{exp}_{model}_{gradcam_variant}_{pre_post_softmax}softmax.yaml",
                    )
                    with open(config_filename, "w") as f:
                        yaml.dump(cfg_dict, f)
                    # train/test/vis, exp num, architecture, grad cam variant, pre-post softmax

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_specific_epochs", help="enable using specific epochs instead of the best performing epoch", action="store_true")
    args = parser.parse_args()

    generate_all_configs(use_specific_epoch=args.use_specific_epochs)

