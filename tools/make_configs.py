# this script generates config files for training, testing, and visualizing
# each of our experiments

import yaml
import os
import pdb
import copy


def generate_all_configs():
    """
    Generates config files for training, testing, and visualizing for each
    of our experiments. Adds a new subfolder to each of the experiments folders
    containing the config files
    """
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

    # generate training config
    for model, model_params_original in model_dicts.items():
        # iterate over experiment
        for exp in experiments:
            # there is an unresolved issue where the entries of 
            # model params kept getting deleted, so solution
            # was to use a deepcopy
            model_params = copy.deepcopy(model_params_original)

            # directory to where the data is located
            data_dir = f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}"
            output_dir = os.path.join(data_dir, f"{model}_output")

            
            train_params = {
                "ENABLE": True,
                "DATASET": "SyntheticMotion",
                # TODO: perhaps loop over this variable if we want to also generate configs for shuffler
                "BATCH_SIZE": 10,
                "EVAL_PERIOD": 1,
                "CHECKPOINT_PERIOD": 1,
                "RESUME_FROM_CHECKPOINT": True,
                "CHECKPOINT_FILE_PATH": model_params["pre_trained_weights_paths"],
                "CHECKPOINT_TYPE": "caffe2",
            }

            test_params = {"ENABLE": False}

            data_params = {
                "PATH_TO_DATA_DIR": data_dir,
                "NUM_FRAMES": 32,
                "TRAIN_JITTER_SCALES": [256, 320],
                "TRAIN_CROP_SIZE": 224,
                "TEST_CROP_SIZE": 256,
                "INPUT_CHANNEL_NUM": model_params["input_channel_num"],
            }
            tensorboard_params = {
                "ENABLE": True,
                "CLASS_NAMES_PATH": f"/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_{exp}/synthetic_motion_labels.json",
                "MODEL_VIS": {
                    "ENABLE": False,
                    "MODEL_WEIGHTS": False,
                    "ACTIVATIONS": False,
                    "INPUT_VIDEO": False,
                    "GRAD_CAM": {
                        "ENABLE": False,
                        "LAYER_LIST": model_params["gradcam_layer_list"],
                        "SOFTMAX_LAYER": "head/act",
                        "SAVE_OVERLAY_VIDEO": False,
                    },
                },
                "GRAD_CAM": {
                    "USE_TRUE_LABEL": False,
                    # "METHOD": gradcam_variant,
                    # "POST_SOFTMAX": softmax_option,
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
            }

            # set up folder to save config files to
            config_dir = os.path.join(data_dir, "configs")
            if not os.path.isdir(config_dir):
                os.makedirs(config_dir)

            # turn the dictionary into a config file and save it
            config_filename = os.path.join(config_dir, f"train_exp{exp}_{model}.yaml")
            with open(config_filename, "w") as f:
                yaml.dump(cfg_dict, f)

    # generate visualization configs
    for model, model_params in model_dicts.items():
        # iterate over experiment
        for exp in experiments:
            # iterate over gradcam variant
            for gradcam_variant in gradcam_variants:
                # iterate over pre/post softmax
                for softmax_option in post_softmax:
                    # generate visualization config
                    # save visualization config
                    # this is where the yaml dump goes

                    pass

    pass
