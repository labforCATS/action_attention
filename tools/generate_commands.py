import os
from datetime import datetime
import json
import numpy as np
import pdb  
    
def generate_commands(server, run_train=False, run_vis=False, run_metrics=False):
    """
    generate_commands takes in information about the server and processes
    that are to be spawned and returns a string consisting of bash commands
    to run the desired processes

    Parameters:
        server: string, either shadowfax or shuffler
        run_train: boolean signifying whether training commands should
            be generated, default False
        run_vis: boolean signifying whether visualization commands should
            be generated, default False
        run_metrics: boolean signifying whether metrics should be run, default
            True
    Returns:
        string; bash commands that can be entered into a terminal
    """
    experiments = ["1", "2", "3", "4", "5", "5b"]
    num_classes = {"1": 7, "2": 7, "3": 7, "4": 7, "5": 7, "5b": 9}
    vis_techniques = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
    post_softmax = [False, True]

    # data_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
    data_dir = "/research/cwloka/data/action_attn/diane_synthetic"
    commands = "python setup.py build develop; "

    # iterate through each experiment
    for experiment in experiments:
        # retrieve config folder for each experiment
        exp_folder = os.path.join(data_dir, f"experiment_{experiment}/configs")
        configs = os.listdir(exp_folder)
        # if run_train:
        #     # filter out non-training configs
        #     train_configs = [config for config in configs if "train" in config]
        #     if server == "shadowfax":
        #         # add both i3d network training configs to the list of commands
        #         i3d_train_configs = [config for config in train_configs if "i3d" in config]
        #         for c in i3d_train_configs:
        #             file_path = os.path.join(exp_folder, c)
        #             commands += f"python3 tools/run_net.py --cfg {file_path}; "
        #     elif server == "shuffler":
        #         # add slowfast training configs to the list of commands
        #         slowfast_train_configs = [config for config in train_configs if "slowfast" in config]
        #         for c in slowfast_train_configs:
        #             file_path = os.path.join(exp_folder, c)
        #             commands += f"python3 tools/run_net.py --cfg {file_path}; "
        #     else:
        #         raise NotImplementedError("Logic for this server needs to be added in")
        
        # if run_vis:
        #     # filter out non-visualization config files
        #     vis_configs = [config for config in configs if "vis" in config]
        #     if server == "shadowfax":
        #         # add both i3d networks' visualization configs to list of commands
        #         i3d_vis_configs = [config for config in vis_configs if "i3d" in config]
        #         for c in i3d_vis_configs:
        #             file_path = os.path.join(exp_folder, c)
        #             commands += f"python3 tools/run_net.py --cfg {file_path}; "
        #     elif server == "shuffler":
        #         # add slowfast networks' visualization configs to list of commands
        #         slowfast_vis_configs = [config for config in vis_configs if "slowfast" in config]
        #         for c in slowfast_vis_configs:
        #             file_path = os.path.join(exp_folder, c)
        #             commands += f"python3 tools/run_net.py --cfg {file_path}; "
        #     else:
        #         raise NotImplementedError("Logic for this server needs to be added in")
        
        if run_metrics:
            # filter out non-metric config files
            metric_configs = [config for config in configs if "metric" in config]
            if server == "shadowfax":
                # add both i3d networks' metrics configs to list of commands
                i3d_metric_configs = [config for config in metric_configs if "i3d" in config]
                for c in i3d_metric_configs:
                    file_path = os.path.join(exp_folder, c)
                    commands += f"python3 tools/run_net.py --cfg {file_path}; "
            elif server == "shuffler":
                # add slowfast networks' metrics configs to list of commands
                slowfast_metric_configs = [config for config in metric_configs if "slowfast" in config]
                for c in slowfast_metric_configs:
                    file_path = os.path.join(exp_folder, c)
                    commands += f"python3 tools/run_net.py --cfg {file_path}; "
            else:
                raise NotImplementedError("Logic for this server needs to be added in")
    
    return commands[:-2]