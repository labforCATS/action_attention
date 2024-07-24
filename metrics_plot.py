"""
Contains functionality to create various plots for metric results.
Should be merged with Nikki and Diane's equivalent files when complete. 

All plots are created with only data from videos/frames that were correctly classified.
Metric values are very similar for correctly vs incorrectly classified videos, but 
we exclude data from incorrectly-classified videos on the assumption that we don't know
*why* they were incorrectly classified (and therefore the heatmaps are not useful).

This is a stupidly long file; couldn't figure out what would be a better way to organize it.
Plots should generate quickly (on the order of seconds to a few minutes); if they're taking longer than that, 
something might be wrong! 

As of 7/23/24, implemented plots are:
    single_model_frames_vs_metric(): # with subset plotting available
    single_model_frames_vs_activation():  # with subset plotting available
    single_model_metric_vs_activation(): 

    multi_model_frames_vs_metric_():
    multi_model_frames_vs_activation():

    multi_experiment_frames_vs_metric():
    multi_experiment_frames_vs_activation():

    arch_model_grid_metric():
    arch_model_grid_activation():

    gen_all_single_model_plots():
    gen_all_subset_single_model_plots():
    gen_all_multi_model_plots():
    gen_all_multi_experiment_plots()
    gen_all_grid_plots_frames_vs_metric_plots(): # probably the most useful function run!
    gen_all_grid_plots_frames_vs_activation_plots():
"""

import os
import pdb
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings
from PIL import Image
from slowfast.visualization.connected_components_utils import load_heatmaps

### global variables ###
experiments = [1, 2, 3, 4, 5, "5b"]
architectures = ["slowfast", "i3d", "i3d_nln"]
gc_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
softmax_status = ["pre_softmax", "post_softmax"]
metrics = ["kl_div", "iou", "pearson", "mse", "covariance"]
# metrics = ["kl_div", "iou", "pearson", "mse", "covariance", "precision", "recall"]

exp_comparisons = [[1, 4], [1, 3, 4], [1, 2], [4, 5], [4, "5b"]]

base_data_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
output_base_folder = "/research/cwloka/data/action_attn/alex_synthetic"
results_dir = os.path.join(base_data_dir, "metric_results")
# results_dir = os.path.join("/research/cwloka/data/action_attn/diane_synthetic", "metric_results")

# for subset plotting only
video_id_to_plot = [0, 1, 2]

experiment_stimulus_name_dict = {
    1 : "Motion",
    2 : "Discr. Motion",
    3 : "Bijection",
    4 : "Appearance", 
    5 : "Static Targets",
    "5b" : "Solo Targets"
}

model_cam_name_dict = {
    "eigen_cam" : "EigenCAM",
    "grad_cam" : "GradCAM",
    "grad_cam_plusplus" : "GradCAM++",
    "cam_plusplus" : "GradCAM++", 
    "i3d_rgb": "I3D",
    "i3dFalsergb": "I3D",
    "i3d_nln" : "NLN",
    "i3dTruergb": "NLN",
    "slowfast_fast" : "SlowFast Fast",
    "slowfastFalsefast": "SlowFast Fast",
    "slowfast_slow" : "SlowFast Slow",
    "slowfastFalseslow": "SlowFast Slow",
}

# for multi model plotting only
# hardcoded in RGB values, heavily modified from IBM design library palette
multi_model_color_dict = {
    "i3dFalsergb" : "#AC5EF0", # I3d is purple
    "i3dTruergb" : "#BD075F", # NLN is pink
    "slowfastFalsefast" : "#FBB705", # Fast is yellow
    "slowfastFalseslow" : "#02A1A5",  # Slow is teal
}

# for multi experiment plotting only
# hardcoded in RGB values, heavily modified from Okabe-Ito palette 
vivid_experiment_color_dict = {
    1 : "#0072B2", # dark blue
    2 : "#12C34F", #green
    3 : "#F3BB0B", # yellow
    4 : "#E62B62",  # fuschia
    5 : "#94027A", # dark purple
    "5b" : "#56C8E9" # light blue
}

# RBGA hex values use alpha = 0.3
pastel_experiment_color_dict = {
    1 : ("#62B6E44d"), # dark blue
    2 : ("#85E8984d"), # green
    3 : ("#F7F1A74d"), # yellow
    4 : ("#FFBBCC4d"), # pink
    5 : ("#C5C1F34d"), # light purple
    "5b" : ("#94D8EC4d")# light blue
}

######################################################################################################
# HELPER FUNCTIONS #
######################################################################################################

def single_model_frames_vs_metric(
    exp,
    vis_technique,
    softmax,
    plot_subset = False,
):
     ########################## shared lines for all single model plots ############################
     for arch in architectures:
        if arch == "slowfast":
                channels = ["slow", "fast"]
        elif arch in ["i3d", "i3d_nln"]:
                channels = ["rgb"]
        else:
                raise NotImplementedError("Add in logic for handling channels")
        for channel in channels:
            output_folder = os.path.join(
                                output_base_folder, f"experiment_{exp}", arch, vis_technique, softmax, channel
                            )

            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

            data_folder_path = os.path.join(
                results_dir, f"experiment_{exp}", arch, vis_technique
            )

            frame_metric_csv = os.path.join(
                data_folder_path, 
                f"exp_{exp}_{arch}_{softmax}_frames.csv"
            )

            df = pd.read_csv(frame_metric_csv)
            df = df.loc[df["channel"] == channel]

            # get frame activations
            framewise_root_dir = os.path.join(
                base_data_dir,
                f"experiment_{exp}",
                f"{arch}_output",
            )

            heatmap_folder = ""
            for entry in os.listdir(framewise_root_dir):
                if "heatmaps_epoch_" in entry:
                    heatmap_folder = entry

            framewise_csv_path = os.path.join(
                framewise_root_dir, heatmap_folder, vis_technique, softmax, f"{channel}_framewise_activations.csv"
            )
            framewisedf = pd.read_csv(framewise_csv_path)

            df["mean_activations"] = framewisedf["mean_activations"].values
            pd.testing.assert_series_equal(df["input_vid_idx"], framewisedf["input_vid_idx"], check_index=False)
            pd.testing.assert_series_equal((df["frame_id"] + 1), framewisedf["frame_id"], check_index=False) 
            # frames are 1-indexed in metrics CSV, 0-indexed in framewise activations CSV
            
            # done getting frame activations

            df = df.loc[df["correct"] == True] # only use data from correctly-classified videos
            if channel == "slow":
                    df["frame_id"] *= 4 # slow channel has 1/4 the frame rate of all other channels

            if plot_subset:
                df = df[df["input_vid_idx"].isin(video_id_to_plot)] 

    ########################## shared lines for all single model plots ############################

            for metric in metrics:
                fig, ax = plt.subplots()

                s = df.pivot_table(index="frame_id", columns="input_vid_idx", values=metric, aggfunc="mean")

                if plot_subset:
                    show_legend = True
                    ax = s.plot(label="input_vid_idx")
                    file_path = os.path.join(output_folder, f"subset_frames_vs_{metric}.png")
                else:
                    show_legend = False
                    ax = s.plot(color="gray", label="input_vid_idx")
                    s.mean(1).plot(ax=ax, color='b', linestyle='--', label='Mean')
                    file_path = os.path.join(output_folder, f"frames_vs_{metric}.png")
                
                ax.legend()
                if show_legend == False:
                    ax.get_legend().remove()

                ax.set_xlabel("frame_id")
                ax.set_ylabel({metric})
                plt.savefig(file_path)
                plt.cla()
                plt.clf()
     plt.close()
            
def single_model_frames_vs_activation(
    exp,
    vis_technique,
    softmax,
    plot_subset = False,
):
     ########################## shared lines for all single model plots ############################
     for arch in architectures:
        if arch == "slowfast":
                channels = ["slow", "fast"]
        elif arch in ["i3d", "i3d_nln"]:
                channels = ["rgb"]
        else:
                raise NotImplementedError("Add in logic for handling channels")
        for channel in channels:
            output_folder = os.path.join(
                                output_base_folder, f"experiment_{exp}", arch, vis_technique, softmax, channel
                            )

            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

            data_folder_path = os.path.join(
                results_dir, f"experiment_{exp}", arch, vis_technique
            )

            frame_metric_csv = os.path.join(
                data_folder_path, 
                f"exp_{exp}_{arch}_{softmax}_frames.csv"
            )

            df = pd.read_csv(frame_metric_csv)
            df = df.loc[df["channel"] == channel]

            # get frame activations
            framewise_root_dir = os.path.join(
                base_data_dir,
                f"experiment_{exp}",
                f"{arch}_output",
            )

            heatmap_folder = ""
            for entry in os.listdir(framewise_root_dir):
                if "heatmaps_epoch_" in entry:
                    heatmap_folder = entry

            framewise_csv_path = os.path.join(
                framewise_root_dir, heatmap_folder, vis_technique, softmax, f"{channel}_framewise_activations.csv"
            )
            framewisedf = pd.read_csv(framewise_csv_path)

            df["mean_activations"] = framewisedf["mean_activations"].values
            pd.testing.assert_series_equal(df["input_vid_idx"], framewisedf["input_vid_idx"], check_index=False)
            pd.testing.assert_series_equal((df["frame_id"] + 1), framewisedf["frame_id"], check_index=False) 
            # frames are 1-indexed in metrics CSV, 0-indexed in framewise activations CSV
            
            # done getting frame activations

            df = df.loc[df["correct"] == True] # only use data from correctly-classified videos
            if channel == "slow":
                    df["frame_id"] *= 4 # slow channel has 1/4 the frame rate of all other channels

            if plot_subset:
                df = df[df["input_vid_idx"].isin(video_id_to_plot)] 

    ########################## shared lines for all single model plots ############################

            s = df.pivot_table(index="frame_id", columns="input_vid_idx", values="mean_activations", aggfunc="mean")

            if plot_subset:
                show_legend = True
                ax = s.plot(label="input_vid_idx")
                file_path = os.path.join(output_folder, f"subset_frames_vs_activation.png")
            else:
                show_legend = False
                ax = s.plot(color="gray",label="input_vid_idx")
                s.mean(1).plot(ax=ax, color='r', linestyle='--', label='Mean')
                file_path = os.path.join(output_folder, f"frames_vs_activation.png")

            ax.legend()
            if show_legend == False:
                ax.get_legend().remove() 

            ax.set_xlabel("frame_id")
            ax.set_ylabel("activation")
            plt.savefig(file_path)
            plt.cla()
            plt.clf()
     plt.close() 

def single_model_metric_vs_activation(
    exp,
    vis_technique,
    softmax,
    plot_subset = False,
):
     ########################## shared lines for all single model plots ############################
     for arch in architectures:
        if arch == "slowfast":
                channels = ["slow", "fast"]
        elif arch in ["i3d", "i3d_nln"]:
                channels = ["rgb"]
        else:
                raise NotImplementedError("Add in logic for handling channels")
        for channel in channels:
            output_folder = os.path.join(
                                output_base_folder, f"experiment_{exp}", arch, vis_technique, softmax, channel
                            )

            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

            data_folder_path = os.path.join(
                results_dir, f"experiment_{exp}", arch, vis_technique
            )

            frame_metric_csv = os.path.join(
                data_folder_path, 
                f"exp_{exp}_{arch}_{softmax}_frames.csv"
            )

            df = pd.read_csv(frame_metric_csv)
            df = df.loc[df["channel"] == channel]

            # get frame activations
            framewise_root_dir = os.path.join(
                base_data_dir,
                f"experiment_{exp}",
                f"{arch}_output",
            )

            heatmap_folder = ""
            for entry in os.listdir(framewise_root_dir):
                if "heatmaps_epoch_" in entry:
                    heatmap_folder = entry

            framewise_csv_path = os.path.join(
                framewise_root_dir, heatmap_folder, vis_technique, softmax, f"{channel}_framewise_activations.csv"
            )
            framewisedf = pd.read_csv(framewise_csv_path)

            df["mean_activations"] = framewisedf["mean_activations"].values
            pd.testing.assert_series_equal(df["input_vid_idx"], framewisedf["input_vid_idx"], check_index=False)
            pd.testing.assert_series_equal((df["frame_id"] + 1), framewisedf["frame_id"], check_index=False) 
            # frames are 1-indexed in metrics CSV, 0-indexed in framewise activations CSV
            
            # done getting frame activations

            df = df.loc[df["correct"] == True] # only use data from correctly-classified videos
            if channel == "slow":
                    df["frame_id"] *= 4 # slow channel has 1/4 the frame rate of all other channels

            if plot_subset:
                df = df[df["input_vid_idx"].isin(video_id_to_plot)] 

    ########################## shared lines for all single model plots ############################

            show_legend = False

            for metric in metrics:
                    fig, ax = plt.subplots()

                    if plot_subset:
                        for i in range(len(video_id_to_plot)):
                            vid_id_df = df[df["input_vid_idx"] == video_id_to_plot[i]] # plot each vid
                            x = np.isfinite(vid_id_df["mean_activations"])
                            y = np.isfinite(vid_id_df[metric])
                            ax.scatter(x, y, label=video_id_to_plot[i])
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            ax.plot(x, p(x))

                        file_path = os.path.join(output_folder, 
                        f"subset_activation_vs_{metric}.png")
                          
                    else:
                        x = df["mean_activations"]
                        y = df[metric]
                        ax.scatter(x,y, marker=".", s=1, color="gray")
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        ax.plot(x, p(x), color="g", label ="trendline")       
                        
                        file_path = os.path.join(output_folder, 
                        f"activation_vs_{metric}.png")

                    ax.legend()
                    if show_legend == False:
                        ax.get_legend().remove() 
                    ax.set_xlabel("activation")
                    ax.set_ylabel({metric})
                    plt.savefig(file_path)
                    plt.cla()
                    plt.clf()
     plt.close()  


def multi_model_frames_vs_metric_plot(
    exp,
    vis_technique,
    softmax,
):
    for metric in metrics:
        print("Creating multi-model frame-vs-metric plots for ", metric)

        fig, ax = plt.subplots()
        dataframe_list = []

        ########################## shared lines for all multi model plots ############################

        for arch in architectures:
            if arch == "slowfast":
                channels = ["slow", "fast"]
            elif arch in ["i3d", "i3d_nln"]:
                channels = ["rgb"]
            else:
                raise NotImplementedError("Add in logic for handling channels")
            for channel in channels:
                output_folder = os.path.join(
                                    output_base_folder, f"experiment_{exp}", "multi_model", vis_technique, softmax
                                )

                if not os.path.exists(output_folder):
                                    os.makedirs(output_folder)

                # retrieve and load csv for results
                data_folder_path = os.path.join(
                    results_dir, f"experiment_{exp}", arch, vis_technique
                )

                # retrieve and load csv with results
                csv_path = os.path.join(
                    data_folder_path, f"exp_{exp}_{arch}_{softmax}_frames.csv"
                )
                df = pd.read_csv(csv_path)
                df = df.loc[df["correct"] == True] # only use data from correctly-classified videos
                df = df.loc[df["channel"] == channel] # separate slow and fast channels if needed
                if channel == "slow":
                    df["frame_id"] *= 4 # slow channel has 1/4 the frame rate of all other channels

                dataframe_list.append(df)

        ########################## shared lines for all multi model plots ############################

        for i in range(len(dataframe_list)):
            s = (dataframe_list[i]).pivot_table(index="frame_id", columns="input_vid_idx", values=metric, aggfunc='mean')

            model_string_list = list(dataframe_list[i]["model"])
            nonlocal_string_list = list(dataframe_list[i]["nonlocal"])
            channel_string_list = list(dataframe_list[i]["channel"])

            model_channel = str(model_string_list[0]) + str(nonlocal_string_list[0]) + str(channel_string_list[0])

            s.mean(1).plot(ax=ax, color = multi_model_color_dict[model_channel], label=model_cam_name_dict[model_channel])

        ax.legend()
        ax.set_xlabel("frame id")
        ax.set_ylabel({metric})

        file_path = os.path.join(output_folder, f"multi_model_frames_vs_{metric}_.png")
        plt.savefig(file_path)
        plt.cla()
        plt.clf()
    plt.close()

def multi_model_frames_vs_activation_plot(
    exp,
    vis_technique,
    softmax,
):

    fig, ax = plt.subplots()
    dataframe_list = []

    ########################## shared lines for all multi model plots ############################
    for arch in architectures:
        if arch == "slowfast":
            channels = ["slow", "fast"]
        elif arch in ["i3d", "i3d_nln"]:
            channels = ["rgb"]
        else:
            raise NotImplementedError("Add in logic for handling channels")
        for channel in channels:
            output_folder = os.path.join(
                                    output_base_folder, f"experiment_{exp}", "multi_model", vis_technique, softmax
                                )

            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

            data_folder_path = os.path.join(
                    results_dir, f"experiment_{exp}", arch, vis_technique
                )

            # retrieve and load csv with results
            csv_path = os.path.join(
                data_folder_path, f"exp_{exp}_{arch}_{softmax}_frames.csv"
            )
            df = pd.read_csv(csv_path)
        
            df = df.loc[df["channel"] == channel] # separate slow and fast channels if needed

            framewise_root_dir = os.path.join(
                base_data_dir,
                f"experiment_{exp}",
                f"{arch}_output",
            )

            heatmap_folder = ""
            for entry in os.listdir(framewise_root_dir):
                if "heatmaps_epoch_" in entry:
                    heatmap_folder = entry

            framewise_csv_path = os.path.join(
                framewise_root_dir, heatmap_folder, vis_technique, softmax, f"{channel}_framewise_activations.csv"
            )
            framewisedf = pd.read_csv(framewise_csv_path)
            
            df["mean_activations"] = framewisedf["mean_activations"].values
            pd.testing.assert_series_equal(df["input_vid_idx"], framewisedf["input_vid_idx"], check_index=False)
            pd.testing.assert_series_equal((df["frame_id"] + 1), framewisedf["frame_id"], check_index=False) 
            # frames are 1-indexed in metrics CSV, 0-indexed in framewise activations csv

            df = df.loc[df["correct"] == True] # only use data from correctly-classified videos
            if channel == "slow":
                df["frame_id"] *= 4 # slow channel has 1/4 the frame rate of all other channels

            dataframe_list.append(df)
    ########################## shared lines for all multi model plots ############################

    for i in range(len(dataframe_list)):
            s = (dataframe_list[i]).pivot_table(index="frame_id", columns="input_vid_idx", values="mean_activations", aggfunc='mean')

            model_string_list = list(dataframe_list[i]["model"])
            nonlocal_string_list = list(dataframe_list[i]["nonlocal"])
            channel_string_list = list(dataframe_list[i]["channel"])

            model_channel = str(model_string_list[0]) + str(nonlocal_string_list[0]) + str(channel_string_list[0])

            s.mean(1).plot(ax=ax, color = multi_model_color_dict[model_channel], label=model_cam_name_dict[model_channel])

    ax.legend()
    ax.set_xlabel("frame id")
    ax.set_ylabel("mean activation")

    file_path = os.path.join(output_folder, f"multi_model_frames_vs_activations.png")
    plt.savefig(file_path)
    plt.cla()
    plt.clf()
    plt.close()

def multi_experiment_frames_vs_metric_plots(
    experiment_subset_list,
    vis_technique,
    softmax,
):
    ########################## shared lines for all multi experiment plots ############################
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

            arch_model_grid_folder = os.path.join(
                                output_base_folder,
                                f"multi_experiment_{experiment_subset_list}", "arch_model_grid", softmax)

            if not os.path.exists(arch_model_grid_folder):
                                os.makedirs(arch_model_grid_folder)

            dataframe_list = []

            for i in range(len(experiment_subset_list)):
                exp = experiment_subset_list[i]

                data_folder_path = os.path.join(
                    results_dir, f"experiment_{exp}", arch, vis_technique
                )

                frame_metric_csv = os.path.join(
                    data_folder_path, 
                    f"exp_{exp}_{arch}_{softmax}_frames.csv"
                )

                df = pd.read_csv(frame_metric_csv)
                df = df.loc[df["channel"] == channel]
                # separate slow and fast channels

                ###### get frame activations #######
                framewise_root_dir = os.path.join(
                    base_data_dir,
                    f"experiment_{exp}",
                    f"{arch}_output",
                )

                heatmap_folder = ""
                for entry in os.listdir(framewise_root_dir):
                    if "heatmaps_epoch_" in entry:
                        heatmap_folder = entry

                framewise_csv_path = os.path.join(
                    framewise_root_dir, heatmap_folder, vis_technique, softmax, f"{channel}_framewise_activations.csv"
                )
                framewisedf = pd.read_csv(framewise_csv_path)

                df["mean_activations"] = framewisedf["mean_activations"].values
                pd.testing.assert_series_equal(df["input_vid_idx"], framewisedf["input_vid_idx"], check_index=False)
                pd.testing.assert_series_equal((df["frame_id"] + 1), framewisedf["frame_id"], check_index=False) 
                # frames are 1-indexed in metrics CSV, 0-indexed in framewise activations CSV

                if channel == "slow":
                    df["frame_id"] *= 4 # slow channel has 1/4 the frame rate of all other channels
                
                df = df.loc[df["correct"] == True] # only use data from correctly-classified videos

                dataframe_list.append(df)

    ########################## shared lines for all multi experiment plots ############################

            for metric in metrics:
                fig, ax = plt.subplots()

                pivot_list = []

                for i in range(len(dataframe_list)):

                    s = (dataframe_list[i]).pivot_table(index="frame_id", columns="input_vid_idx", values=metric, aggfunc='mean')
                    s.rename(columns=lambda x: "_" + str(x), inplace=True)
                    # s.plot(ax=ax, color=pastel_color_list[i])
                    s.plot(ax=ax, color=pastel_experiment_color_dict[experiment_subset_list[i]])
                    pivot_list.append(s)

                for i in range(len(dataframe_list)):
                    # (pivot_list[i]).mean(1).plot(ax=ax, color=vivid_color_list[i], label=experiment_subset_list[i])
                    (pivot_list[i]).mean(1).plot(ax=ax, color=vivid_experiment_color_dict[experiment_subset_list[i]], label=experiment_subset_list[i])

                ax.legend()

                ax.set_xlabel("frame id")
                ax.set_ylabel({metric})

                file_path = os.path.join(output_folder, f"multi_exp_frames_vs_{metric}_.png")
                plt.savefig(file_path)

                arch_cam_grid_path = os.path.join(arch_model_grid_folder, f"frames_vs_{metric}_{arch}_{channel}_{vis_technique}.png")
                plt.savefig(arch_cam_grid_path)

                plt.cla()
                plt.clf()
            
            print("plotted for", arch, channel)

    plt.close()

def multi_experiment_frames_vs_activation_plots(
    experiment_subset_list,
    vis_technique,
    softmax,
):
    ########################## shared lines for all multi experiment plots ############################
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

            arch_model_grid_folder = os.path.join(
                                output_base_folder,
                                f"multi_experiment_{experiment_subset_list}", "arch_model_grid", softmax)

            if not os.path.exists(arch_model_grid_folder):
                                os.makedirs(arch_model_grid_folder)

            dataframe_list = []

            for i in range(len(experiment_subset_list)):
                exp = experiment_subset_list[i]

                data_folder_path = os.path.join(
                    results_dir, f"experiment_{exp}", arch, vis_technique
                )

                frame_metric_csv = os.path.join(
                    data_folder_path, 
                    f"exp_{exp}_{arch}_{softmax}_frames.csv"
                )

                df = pd.read_csv(frame_metric_csv)
                df = df.loc[df["channel"] == channel]
                # separate slow and fast channels

                ###### get frame activations #######
                framewise_root_dir = os.path.join(
                    base_data_dir,
                    f"experiment_{exp}",
                    f"{arch}_output",
                )

                heatmap_folder = ""
                for entry in os.listdir(framewise_root_dir):
                    if "heatmaps_epoch_" in entry:
                        heatmap_folder = entry

                framewise_csv_path = os.path.join(
                    framewise_root_dir, heatmap_folder, vis_technique, softmax, f"{channel}_framewise_activations.csv"
                )
                framewisedf = pd.read_csv(framewise_csv_path)

                df["mean_activations"] = framewisedf["mean_activations"].values
                pd.testing.assert_series_equal(df["input_vid_idx"], framewisedf["input_vid_idx"], check_index=False)
                pd.testing.assert_series_equal((df["frame_id"] + 1), framewisedf["frame_id"], check_index=False) 
                # frames are 1-indexed in metrics CSV, 0-indexed in framewise activations CSV

                if channel == "slow":
                    df["frame_id"] *= 4 # slow channel has 1/4 the frame rate of all other channels
                
                df = df.loc[df["correct"] == True] # only use data from correctly-classified videos

                dataframe_list.append(df)

            ########################## shared lines for all multi experiment plots ############################

            fig, ax = plt.subplots()

            pivot_list = []

            for i in range(len(dataframe_list)):

                s = (dataframe_list[i]).pivot_table(index="frame_id", columns="input_vid_idx", values="mean_activations", aggfunc='mean')
                s.rename(columns=lambda x: "_" + str(x), inplace=True)
                s.plot(ax=ax, color=pastel_experiment_color_dict[experiment_subset_list[i]])
                pivot_list.append(s)

            for i in range(len(dataframe_list)):
                (pivot_list[i]).mean(1).plot(ax=ax, color=vivid_experiment_color_dict[experiment_subset_list[i]], label=experiment_subset_list[i])

            ax.legend()

            ax.set_xlabel("frame id")
            ax.set_ylabel("mean_activations")

            file_path = os.path.join(output_folder, f"multi_exp_frames_vs_activation_.png")
            plt.savefig(file_path)
            arch_cam_grid_path = os.path.join(arch_model_grid_folder, f"frames_vs_activation_{arch}_{channel}_{vis_technique}.png")
            plt.savefig(arch_cam_grid_path)
            plt.cla()
            plt.clf()
        
            print("plotted for", arch, channel)
    plt.close()

def arch_model_grid_metric(
    experiment_subset_list = [1, 4], metric = "iou", softmax = "pre_softmax"
    ):

    flattened_image_dir = os.path.join(output_base_folder, f"multi_experiment_{experiment_subset_list}", "arch_model_grid", softmax)

    image_name_list = []
    for __, __, files in os.walk(flattened_image_dir):
        for f in files:
            if f[10:(10+len(metric))] == metric:
                image_name_list.append(f)
    
    assert len(image_name_list) == 12 # 4 arch/channel by 3 cams 
    image_name_list.sort() 
    # places in order of nln eigen, nln gc, nln ++, i3d eigen, i3d gc, i3d ++,
    #                    fast eigen, fast gc, fast ++, slow eigen, slow gc, slow ++

    assert image_name_list[0][-21:] == "nln_rgb_eigen_cam.png" # check sorting is correct
    assert image_name_list[4][-20:] == "i3d_rgb_grad_cam.png" # check sorting is correct
    assert image_name_list[-1][-26:] == "slow_grad_cam_plusplus.png" # check sorting is correct

    fig,ax = plt.subplots(3,4)

    plt.subplots_adjust(left=0.1,
                    bottom=0.2, 
                    right=0.9, 
                    top=0.8, 
                    wspace=0.05, 
                    hspace=0.0001)

    for j in range(4): # columns of models
        for i in range(3): # rows of CAMs
            name = image_name_list[(3*j)+i]
            with open(flattened_image_dir + "/" + name, "rb") as f:
                image = Image.open(f)
                ax[i][j].imshow(image)

                if metric == "kl_div": # kl_div has a _ in the name, which disrupts later string handling
                    name = "_".join(name.split("_")[:2]) + "_kldiv_" + "_".join(name.split("_")[4:])

                model_name = name.split("_")[3:5] # i3d_nln, i3d_rgb, or slowfast_channel
                model_name = "_".join(model_name)
                model_name = model_cam_name_dict[model_name]
                cam_name = name.split("_")[-2:] # eigen_cam, grad_cam, or cam_plusplus
                cam_name = "_".join(cam_name)
                cam_name = cam_name[:-4] # remove png
                cam_name = model_cam_name_dict[cam_name]
                
                ax[i][j].set_xlabel(model_name)
                ax[i][j].set_ylabel(cam_name)

    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
        a.label_outer()

    stimulus_set_names = [experiment_stimulus_name_dict[x] for x in experiment_subset_list]

    fig.suptitle(f"{metric} for stimulus sets {stimulus_set_names} ({softmax})", x=0.5, y=0.85, fontsize=12)

    file_path = os.path.join(flattened_image_dir, f"GRID_frames_vs_{metric}_{softmax}.png")
    plt.savefig(file_path, dpi=500)
    plt.close()


def arch_model_grid_activation(
    experiment_subset_list = [1, 4], softmax = "pre_softmax"
    ):

    flattened_image_dir = os.path.join(output_base_folder, f"multi_experiment_{experiment_subset_list}", "arch_model_grid", softmax)

    image_name_list = []
    for __, __, files in os.walk(flattened_image_dir):
        for f in files:
            if f[10:(10+len("activation"))] == "activation": #frames_vs_{yvar}
                image_name_list.append(f)
    
    assert len(image_name_list) == 12 # 4 arch/channel by 3 cams 
    image_name_list.sort() 
    # places in order of nln eigen, nln gc, nln ++, i3d eigen, i3d gc, i3d ++,
    #                    fast eigen, fast gc, fast ++, slow eigen, slow gc, slow ++

    assert image_name_list[0][-21:] == "nln_rgb_eigen_cam.png" # check sorting is correct
    assert image_name_list[4][-20:] == "i3d_rgb_grad_cam.png" # check sorting is correct
    assert image_name_list[-1][-26:] == "slow_grad_cam_plusplus.png" # check sorting is correct

    fig,ax = plt.subplots(3,4)

    plt.subplots_adjust(left=0.1,
                    bottom=0.2, 
                    right=0.9, 
                    top=0.8, 
                    wspace=0.05, 
                    hspace=0.0001)

    for j in range(4): # columns of models
        for i in range(3): # rows of CAMs
            name = image_name_list[(3*j)+i]
            with open(flattened_image_dir + "/" + name, "rb") as f:
                image = Image.open(f)
                ax[i][j].imshow(image)

                model_name = name.split("_")[3:5] # i3d_nln, i3d_rgb, or slowfast_channel
                model_name = "_".join(model_name)
                model_name = model_cam_name_dict[model_name]
                cam_name = name.split("_")[-2:] # eigen_cam, grad_cam, or cam_plusplus
                cam_name = "_".join(cam_name)
                cam_name = cam_name[:-4] # remove png
                cam_name = model_cam_name_dict[cam_name]
                
                ax[i][j].set_xlabel(model_name)
                ax[i][j].set_ylabel(cam_name)

    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
        a.label_outer()

    stimulus_set_names = [experiment_stimulus_name_dict[x] for x in experiment_subset_list]

    fig.suptitle(f"activation for stimulus sets {stimulus_set_names} ({softmax})", x=0.5, y=0.85, fontsize=12)

    file_path = os.path.join(flattened_image_dir, f"GRID_frames_vs_activation_{softmax}.png")
    plt.savefig(file_path, dpi=500)
    plt.close()

######################################################################################################
# "generate all" functions to generate one kind of plot for all applicable configurations
######################################################################################################

def gen_all_single_model_plots():
    for exp in experiments:
        for vis_technique in gc_variants:
            for softmax in softmax_status:
                print("single model plots for ", exp, vis_technique, softmax)

                single_model_frames_vs_metric(
                    exp,
                    vis_technique,
                    softmax,
                )

                single_model_frames_vs_activation(
                    exp,
                    vis_technique,
                    softmax,
                )

                single_model_metric_vs_activation(
                    exp,
                    vis_technique,
                    softmax,
                )

                print("single model plots for ", exp, vis_technique, softmax)

def gen_all_subset_single_model_plots():
    for exp in experiments:
        for vis_technique in gc_variants:
            for softmax in softmax_status:

                single_model_frames_vs_metric(
                    exp,
                    vis_technique,
                    softmax,
                    plot_subset=True,
                )

                single_model_frames_vs_activation(
                    exp,
                    vis_technique,
                    softmax,
                    plot_subset=True,
                )

                print("subset single model plots for ", exp, vis_technique, softmax)

def gen_all_multi_model_plots():
      for exp in experiments:
        for vis_technique in gc_variants:
            for softmax in softmax_status:
                
                multi_model_frames_vs_metric_plot(
                    exp,
                    vis_technique,
                    softmax,
                )

                multi_model_frames_vs_activation_plot(
                    exp,
                    vis_technique,
                    softmax,
                )

                print("multi model plots for ", exp, vis_technique, softmax)


def gen_all_multi_experiment_plots():
    for subset in exp_comparisons:
        for vis_technique in gc_variants:
            for softmax in softmax_status:
                multi_experiment_frames_vs_metric_plots(
                    subset,
                    vis_technique,
                    softmax,
                )
                
                multi_experiment_frames_vs_activation_plots(
                    subset,
                    vis_technique,
                    softmax,
                )

                print("multi experiment plots for ", subset, vis_technique, softmax)

def gen_all_grid_frames_vs_metric_plots():
    for subset in exp_comparisons:
        for metric in metrics:
            for softmax in softmax_status:
                
                arch_model_grid_metric(experiment_subset_list = subset, metric=metric, softmax=softmax)

                print("grid metric plots for ", subset, metric, softmax)

def gen_all_grid_frames_vs_activation_plots():
    for subset in exp_comparisons:
        for metric in metrics:
            for softmax in softmax_status:
                
                arch_model_grid_activation(experiment_subset_list = subset, softmax=softmax)

                print("grid activation plots for ", subset, softmax)