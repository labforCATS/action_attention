import os
from datetime import datetime
import json
import numpy as np
import pdb  
    
def generate_commands(server, run_train=True, run_vis=True):
    experiments = ["1", "2", "3", "4", "5", "5b"]
    num_classes = {"1": 7, "2": 7, "3": 7, "4": 7, "5": 7, "5b": 9}
    vis_techniques = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
    post_softmax = [False, True]

    commands = "python setup.py build develop; "

    data_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
    for experiment in experiments:
        exp_folder = os.path.join(data_dir, f"experiment_{experiment}/configs")
        configs = os.listdir(exp_folder)
        if run_train:
            train_configs = [config for config in configs if "train" in config]
            if server == "shadowfax":
                i3d_train_configs = [config for config in train_configs if "i3d_nln" in config]
                for c in i3d_train_configs:
                    file_path = os.path.join(exp_folder, c)
                    commands += f"python3 tools/run_net.py --cfg {file_path}; "
            elif server == "shuffler":
                slowfast_train_configs = [config for config in train_configs if "slowfast" in config]
                for c in slowfast_train_configs:
                    file_path = os.path.join(exp_folder, c)
                    commands += f"python3 tools/run_net.py --cfg {file_path}; "
    return commands[:-2]