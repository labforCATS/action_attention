import os
import pdb
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings
from PIL import Image
from slowfast.visualization.connected_components_utils import load_heatmaps
from sklearn.metrics import r2_score


# target_data_dir = "/research/cwloka/data/action_attn/diane_synthetic/metric_results"
bbox_data_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments_bbox/metric_results"
target_data_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments_rot_bbox/metric_results" # for rotated bbox
output_base_folder = "/research/cwloka/data/action_attn/bbox_reg_vs_rotated"

### global variables ###
# experiments = [1, 2, 3, 4, 5, "5b"]
experiments = [1, 4]
# architectures = ["slowfast", "i3d", "i3d_nln"]
architectures = ["slowfast", "i3d"]
gc_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
softmax_status = ["pre_softmax", "post_softmax"]
metrics = ["kl_div", "iou", "pearson", "mse", "covariance", "precision", "recall"]
# exp_comparisons = [[1, 4], [1, 3, 4], [1, 2], [4, 5], [4, "5b"]]
exp_comparisons = [[1, 4]]

def plot_target_vs_bbox(exp, arch, vis_technique, softmax):
    if arch == "slowfast":
                channels = ["slow", "fast"]
    elif arch in ["i3d", "i3d_nln"]:
            channels = ["rgb"]
    else:
            raise NotImplementedError("Add in logic for handling channels")
    for channel in channels:
        # output_folder = os.path.join(
        #                     output_base_folder,
        #                     f"{exp}_{arch}_{vis_technique}_{softmax}_{channel}"
        #                 )

        output_folder = os.path.join(
                            output_base_folder, f"{exp}", arch, vis_technique, softmax, channel
                        )

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # target_csv = os.path.join(target_data_dir, f"experiment_{exp}", arch, vis_technique, f"exp_{exp}_{arch}_{softmax}_frames.csv") #for REGULAR TARGETS
        target_csv = os.path.join(target_data_dir, f"experiment_{exp}", arch, vis_technique, f"rot_bbox", f"exp_{exp}_bbox_{arch}_{softmax}_frames.csv") # for rotated bbox with weird pathing
        target_df = pd.read_csv(target_csv)
        target_df["unique_id"] = target_df["input_vid_idx"].astype(str) + target_df["label_numeric"].astype(str)
        target_df = target_df.loc[target_df["channel"] == channel]
        target_df.drop_duplicates(inplace = True)
        target_df = target_df.loc[target_df["correct"] == True]


        bbox_csv = os.path.join(bbox_data_dir, f"experiment_{exp}", arch, vis_technique, f"exp_{exp}_bbox_{arch}_{softmax}_frames.csv")
        bbox_df = pd.read_csv(bbox_csv)
        bbox_df["unique_id"] = bbox_df["input_vid_idx"].astype(str) + bbox_df["label_numeric"].astype(str)
        bbox_df = bbox_df.loc[bbox_df["channel"] == channel]
        bbox_df.drop_duplicates(inplace = True)
        bbox_df = bbox_df.loc[bbox_df["correct"] == True]

        for metric in metrics:
            fig, ax = plt.subplots()

            metric_target_df = target_df[["unique_id", "frame_id", metric]].copy().set_index(["unique_id", "frame_id"])
            metric_target_df.rename(columns={metric:f"target_{metric}"}, inplace=True)
            metric_bbox_df = bbox_df[["unique_id", "frame_id", metric]].copy().set_index(["unique_id", "frame_id"])
            metric_bbox_df.rename(columns={metric:f"bbox_{metric}"}, inplace=True)

            try: 
                df = metric_target_df.join(metric_bbox_df)
                df = df.dropna()
            except:
                print('error occurred with metric df', str(target_csv), "and framewise df", str(bbox_csv))
                pdb.set_trace()
 
            x=df[f"target_{metric}"]
            y=df[f"bbox_{metric}"]
            plt.scatter(x,y, s=2)

            # ax.set_xlabel(f"target_{metric}")
            # ax.set_ylabel(f"bbox_{metric}")
            ax.set_xlabel(f"rotated_bbox_{metric}")
            ax.set_ylabel(f"bbox_{metric}")

            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x,p(x),"r--")
            
            # https://www.statology.org/r-squared-in-python/
            #initiate linear regression model
            ax.annotate(text=("r-squared = {:.3f}".format(r2_score(y, p(x)))), xy=(0.2,0.8),xycoords='figure fraction')

            file_path = os.path.join(output_folder, f"{metric}reg_vs_rotated_bbox.png")
            plt.savefig(file_path)

            plt.cla()
            plt.clf()

        plt.close()

    
def gen_all_target_vs_bbox():
    for exp in experiments:
        for arch in architectures:
            for vis_technique in gc_variants:
                for softmax in softmax_status:
                    plot_target_vs_bbox(exp, arch, vis_technique, softmax)
                    print("finished plots for ", exp, arch, vis_technique, softmax)
           
                