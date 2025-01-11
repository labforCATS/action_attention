"""
Creates all frame-vs-metric and frame-vs-activation plots from metric results CSVs.
Plots are created with only data from videos/frames that were correctly classified.
Plots should generate within a couple minutes. 
"""

import os
import pdb
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
from PIL import Image
from slowfast.visualization.connected_components_utils import load_heatmaps
from plot_dictionaries import *

#### GLOBAL VARIABLES #####
experiments = [1, 2, 3, 4, 5, "5b"]
architectures = ["slowfast", "i3d", "i3d_nln"]
gc_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
softmax_status = ["pre_softmax", "post_softmax"]
metrics = ["activation", "kl_div", "iou", "pearson", "mse", "covariance", "precision", "recall"]
# exp_comparisons = [[1, 4], [1, 3, 4], [1, 2], [4, 5], [4, "5b"]]
exp_comparisons = [[1, 4]]

#### FOR REGULAR SYNTHETIC MOTION #####
# base_data_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
# results_dir = "/research/cwloka/data/action_attn/diane_synthetic/metric_results"
# output_base_folder = "/research/cwloka/data/action_attn/alex_synthetic"
##############################

#### FOR SYNTHETIC BOUNDING BOXES #####
# base_data_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
# results_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments_bbox/metric_results"
# output_base_folder = "/research/cwloka/data/action_attn/bbox_plots"
##############################

#### FOR SYNTHETIC ROTATED BOUNDING BOXES #####
base_data_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
results_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments_rot_bbox/metric_results"
output_base_folder = "/research/cwloka/data/action_attn/rotated_bbox_plots"

##############################


######################################################################################################
# HELPER FUNCTIONS #
######################################################################################################

def merge_metric_and_activation_csv(arch, vis_technique, softmax, channel, exp):
    ######################################
    # for regular synthetic motion
    # frame_metric_csv = os.path.join(
    #     results_dir, f"experiment_{exp}", arch, vis_technique, f"exp_{exp}_{arch}_{softmax}_frames.csv")

    # for synthetic bounding boxes
    # frame_metric_csv = os.path.join(
    #     results_dir, f"experiment_{exp}", arch, vis_technique, f"exp_{exp}_bbox_{arch}_{softmax}_frames.csv")

    # for synthetic rotated bounding boxes
    frame_metric_csv = os.path.join(
        results_dir, f"experiment_{exp}", arch, vis_technique, f"rot_bbox", f"exp_{exp}_bbox_{arch}_{softmax}_frames.csv")
    ######################################

    df = pd.read_csv(frame_metric_csv)
    df = df.loc[df["channel"] == channel]
    df.drop_duplicates(inplace = True)

    # get framewise activation dataframe
    framewise_root_dir = os.path.join(
        base_data_dir, f"experiment_{exp}", f"{arch}_output",
    )

    heatmap_folder = ""
    for entry in os.listdir(framewise_root_dir):
        if "heatmaps_epoch_" in entry:
            heatmap_folder = entry

    framewise_csv_path = os.path.join(
        framewise_root_dir, heatmap_folder, vis_technique, softmax, f"{channel}_framewise_activations.csv"
    )
    
    framewisedf = pd.read_csv(framewise_csv_path)

    # merge the two
    try:
        df["mean_activations"] = framewisedf["mean_activations"].values
        pd.testing.assert_series_equal(df["input_vid_idx"], framewisedf["input_vid_idx"], check_index=False)
        pd.testing.assert_series_equal((df["frame_id"] + 1), framewisedf["frame_id"], check_index=False) 
        # frames are 1-indexed in metrics CSV, 0-indexed in framewise activations CSV
    except:
        print('error occurred with metric df', str(frame_metric_csv), "and framewise df", str(framewise_csv_path ))
        pdb.set_trace()

    if channel == "slow":
        df["frame_id"] *= 4 # slow channel has 1/4 the frame rate of all other channels
    df = df.loc[df["correct"] == True] # only use data from correctly-classified videos
    df = df.rename(columns={"mean_activations": "activation"})

    return df

def find_metric_maximums():
    metric_dict = {}
    for exp in experiments:
        for arch in architectures:
            for vis_technique in gc_variants:
                for softmax in softmax_status:
                    if arch == "slowfast":
                            channels = ["slow", "fast"]
                    elif arch in ["i3d", "i3d_nln"]:
                            channels = ["rgb"]
                    else:
                            raise NotImplementedError("Add in logic for handling channels")
                    for channel in channels:
                        df = merge_metric_and_activation_csv(arch, vis_technique, softmax, channel, exp)

                        for metric in metrics:

                            metric_df = df[["input_vid_idx", "label","label_numeric", "frame_id", metric]].copy()
                            maxlist = []

                            for class_id in metric_df["label_numeric"].unique():
                                plot_df = metric_df[metric_df["label_numeric"] == class_id]
                                mean_df = plot_df.groupby(["frame_id"]).mean(numeric_only = True)

                                maxlist.append(mean_df[metric].max())

                            key = f"{metric}_{exp}_{softmax}"
                            metric_dict.setdefault(key, [])
                            metric_dict[f"{metric}_{exp}_{softmax}"].append(max(maxlist))

                        activation_df = df[["input_vid_idx", "label","label_numeric", "frame_id", "mean_activations"]].copy()
                        maxlist = []

                        for class_id in activation_df["label_numeric"].unique():
                            plot_df = activation_df[metric_df["label_numeric"] == class_id]
                            mean_df = plot_df.groupby(["frame_id"]).mean(numeric_only = True)
                            maxlist.append(mean_df["mean_activations"].max())

                        key = f"activation_{exp}_{softmax}"
                        metric_dict.setdefault(key, [])
                        metric_dict[f"activation_{exp}_{softmax}"].append(max(maxlist))

    for exp in experiments:
        for key, value in metric_dict.items():
                cleaned_metric_dict[key] = max(value)

######################################################################################################
# PLOTTING FUNCTIONS #
######################################################################################################

def rescaled_multi_experiment_frames_vs_metric_plots(
    experiment_subset_list,
    vis_technique,
    softmax,
):
    plt.rcdefaults()
    warnings.filterwarnings("ignore") # avoid spam of warnings that these lines are not on legend

    for arch in architectures:
        if arch == "slowfast":
                channels = ["slow", "fast"]
        elif arch in ["i3d", "i3d_nln"]:
                channels = ["rgb"]
        else:
                raise NotImplementedError("Add in logic for handling channels")

        for channel in channels:
            output_folder = os.path.join(
                                output_base_folder,
                                f"multi_experiment_{experiment_subset_list}", arch, vis_technique, softmax, channel
                            )

            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

            output_grid_folder = os.path.join(
                                output_base_folder,
                                f"multi_experiment_{experiment_subset_list}", f"grid", arch, vis_technique, softmax, channel
                            )

            if not os.path.exists(output_grid_folder):
                                os.makedirs(output_grid_folder)

            dataframe_list = []

            for i in range(len(experiment_subset_list)):
                exp = experiment_subset_list[i]
                df = merge_metric_and_activation_csv(arch, vis_technique, softmax, channel, exp)
                dataframe_list.append(df)

            for metric in metrics:
                fig, ax = plt.subplots()

                pivot_list = []

                for i in range(len(dataframe_list)):

                    metric_df = dataframe_list[i][["input_vid_idx", "label","label_numeric", "frame_id", metric]].copy()

                    for class_id in metric_df["label_numeric"].unique():
                        plot_df = metric_df[metric_df["label_numeric"] == class_id]
                        mean_df = plot_df.groupby(["frame_id"]).mean()

                        classlabel = plot_df["label"].iloc[0]

                        mean_df.plot(ax=ax, y=metric, color=pastel_experiment_color_dict[experiment_subset_list[i]], label="_nolegend_")

                        mean_df['frameindex'] = mean_df.index

                        ax.scatter(x=mean_df['frameindex'][::8], y=mean_df[metric][::8], color=vivid_experiment_color_dict[experiment_subset_list[i]], marker = label_marker_dict[classlabel], label=classlabel)
                    
                    mean_df = metric_df.groupby(["frame_id"]).mean()
                    stimuluslabel = experiment_stimulus_name_dict[experiment_subset_list[i]]
                    mean_df.plot(ax=ax, y=metric, color=vivid_experiment_color_dict[experiment_subset_list[i]], label=f"{stimuluslabel} mean")
                    
                ax.legend(bbox_to_anchor=(1.1, 1.05))

                ax.set_xlabel("Frame Index")
                ax.set_ylabel({metric})

                file_path = os.path.join(output_folder, f"frames_vs_{metric}.png")
                plt.savefig(file_path, bbox_inches='tight')

                y_limits = []
                for i in experiment_subset_list:
                    config = metric + "_" + str(i) + "_" + softmax
                    y_limits.append(cleaned_metric_dict[config])
                ax.set_ylim([0, max(y_limits)])

                ax.get_legend().remove() # no need for legend on grid plots

                grid_path = os.path.join(output_grid_folder, f"rescaled_frames_vs_{metric}.png")
                plt.savefig(grid_path)

                plt.cla()
                plt.clf()
            
            print("plotted for", arch, channel)

    plt.close()

def new_grid(
    rows = ["eigen_cam", "grad_cam","grad_cam_plusplus"],
    cols = ["i3d", "i3d_nln", "fast", "slow"], 
    experiment_subset_list = [1, 4], 
    metric = "iou", 
    softmax = "post_softmax"
    ):
    # rows = ["eigen_cam","grad_cam", "grad_cam_plusplus"],

    row = 0
    col = 0

    plt.rcdefaults()
    fig,axs = plt.subplots(len(rows),len(cols), figsize=(40,20))
    
    for vis_technique in rows:
        col = 0
        for arch in cols:
            if arch in ["fast", "slow"]:
                imgpath = os.path.join(
                                    output_base_folder,
                                    f"multi_experiment_{experiment_subset_list}", f"grid",
                                    f"slowfast", vis_technique, softmax, arch, 
                                    f"rescaled_frames_vs_{metric}.png"
                                )
            elif arch in ["i3d", "i3d_nln"]:
                    imgpath = os.path.join(
                                    output_base_folder,
                                    f"multi_experiment_{experiment_subset_list}", f"grid",
                                    arch, vis_technique, softmax, f"rgb", 
                                    f"rescaled_frames_vs_{metric}.png"
                                )
            else:
                    raise NotImplementedError("Add in logic for handling channels")

            img = cv2.imread(imgpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[row, col].imshow(img)
            axs[row, col].set_xlabel(arch)
            axs[row, col].set_ylabel(vis_technique)

            col += 1
        row += 1
        
    for a in axs.flat:
        a.set_xticks([])
        a.set_yticks([])
        a.xaxis.label.set_fontsize(30)
        a.yaxis.label.set_fontsize(30)
        a.label_outer()

    plt.tight_layout()

    out_path = os.path.join(output_base_folder,
                                    f"multi_experiment_{experiment_subset_list}", f"GRID_frames_vs_{metric}_{softmax}.png")
    plt.savefig(out_path, dpi=500)
    plt.close()


######################################################################################################
# "generate all" functions to generate one kind of plot for all applicable configurations
######################################################################################################

def gen_all_rescaled_multi_experiment_plots():
    for subset in exp_comparisons:
        for vis_technique in gc_variants:
            for softmax in softmax_status:
                print("starting for", subset, vis_technique, softmax)
                rescaled_multi_experiment_frames_vs_metric_plots(
                    subset,
                    vis_technique,
                    softmax,
                )
                
                rescaled_multi_experiment_frames_vs_activation_plots(
                    subset,
                    vis_technique,
                    softmax,
                )
                print("finished plots for ", subset, vis_technique, softmax)


def gen_all_rescaled_grid_frames_vs_metric_plots():
    for subset in exp_comparisons:
        for metric in metrics:
            for softmax in softmax_status:
                
                rescaled_arch_model_grid_metric(experiment_subset_list = subset, metric=metric, softmax=softmax)

                print("grid metric plots for ", subset, metric, softmax)

def gen_all_rescaled_grid_frames_vs_activation_plots():
    for subset in exp_comparisons:
        for softmax in softmax_status:
        
            rescaled_arch_model_grid_activation(experiment_subset_list = subset, softmax=softmax)

            print("grid activation plots for ", subset, softmax)