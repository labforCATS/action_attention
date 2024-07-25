from slowfast.utils.parser import load_config

import os
import json
import shutil
import pdb
import traceback

old_metric_results_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments/metric_results"
preserve = "/research/cwloka/data/action_attn"

dir_to_change = "/synthetic_motion_experiments"
new_dir = "/backup_synthetic_metric_results"

def backup_metrics_csvs():
    count = 0
    for root, _, filenames in os.walk(old_metric_results_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                old_file_path = os.path.join(root, filename)

                start_root = root[:len(preserve)] 
                end_root = root[len(preserve) + len(dir_to_change):]

                new_root = start_root + new_dir + end_root
                if not os.path.exists(new_root):
                    os.makedirs(new_root)
                new_file_path = os.path.join(new_root, filename)
                
                shutil.move(old_file_path, new_file_path)

                count+=1

    print("moved ", count, " files to new directory ", new_dir)
