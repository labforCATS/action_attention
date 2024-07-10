"""
Contains functionality to create various plots for metric results.
Should be merged with Nikki and Diane's equivalent files when complete. 

All plots are created with only data from videos/frames that were correctly classified.
Metric values are very similar for correctly vs incorrectly classified videos, but 
we exclude data from incorrectly-classified videos on the assumption that we don't know
*why* they were incorrectly classified (and therefore the heatmaps are not useful).
"""

import os
import pdb
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from slowfast.visualization.connected_components_utils import load_heatmaps

### global variables ###
experiments = [1, 2, 3, 4, 5, "5b"]
architectures = ["slowfast", "i3d", "i3d_nln"]
gc_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
softmax_status = ["pre_softmax", "post_softmax"]
metrics = ["kl_div", "iou", "pearson", "mse", "covariance"]

exp_base_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"

######################################################################################################
# Functions that we *are* using! (unnecessary but possibly useful-later functions are lower in file) #
######################################################################################################

def frame_vs_metric_plot(
    exp,
    vis_technique,
    softmax,
    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
):
    
    output_base_folder = "/research/cwloka/data/action_attn/alex_synthetic"
    base_dir = os.path.join(exp_base_dir, "metric_results")

    for arch in architectures:
        if arch == "slowfast":
                channels = ["slow", "fast"]
        elif arch in ["i3d", "i3d_nln"]:
                channels = ["rgb"]
        else:
                raise NotImplementedError("Add in logic for handling channels")
        for channel in channels:

            output_folder = os.path.join(
                                output_base_folder, f"experiment_{exp}", arch, vis_technique, channel
                            )

            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)


            # retrieve and load csv for results
            data_folder_path = os.path.join(
                base_dir, f"experiment_{exp}", arch, vis_technique
            )

            # retrieve and load csv with results
            csv_path = os.path.join(
                data_folder_path, f"exp_{exp}_{arch}_{softmax}_frames.csv"
            )
            df = pd.read_csv(csv_path)
            df = df.loc[df["channel"] == channel] # separate slow and fast channels if needed
            df = df.loc[df["correct"] == True] # only use data from correctly-classified videos

            for metric in metrics:
                # print("Creating frame vs metric plots for ", arch, " ", channel, " and ", metric)

                metric_val = df[metric]
                frame_id = df["frame_id"]
                video_name = df["input_vid_idx"]

                s = df.pivot_table(index="frame_id", columns="input_vid_idx", values=metric, aggfunc='mean')
                ax = s.plot(color='gray', label="_hidden")
                s.mean(1).plot(ax=ax, color='b', linestyle='--', label='Mean')
                ax.get_legend().remove() 
                ax.set_xlabel("frame id")
                ax.set_ylabel({metric})

                file_path = os.path.join(output_folder, f"frames_vs_{metric}_.png")
                plt.savefig(file_path)
                plt.close()    

def gen_all_single_model_frame_vs_metric_plots():  
    for exp in experiments:
        for vis_technique in gc_variants:
            for softmax in softmax_status:
                frame_vs_metric_plot(
                    exp,
                    vis_technique,
                    softmax,
                    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
                )

def multi_model_frame_vs_metric_plot(
    exp,
    vis_technique,
    softmax,
    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
):
    
    output_base_folder = "/research/cwloka/data/action_attn/alex_synthetic"
    base_dir = os.path.join(exp_base_dir, "metric_results")

    for metric in metrics:
        print("Creating multi-model frame-vs-metric plots for ", metric)

        pivot_list = []

        for arch in architectures:
            if arch == "slowfast":
                channels = ["slow", "fast"]
            elif arch in ["i3d", "i3d_nln"]:
                channels = ["rgb"]
            else:
                raise NotImplementedError("Add in logic for handling channels")
            for channel in channels:
                print("Plotting data for ", arch, channel)
                output_folder = os.path.join(
                                    output_base_folder, f"experiment_{exp}", "multi_model", vis_technique, softmax
                                )

                if not os.path.exists(output_folder):
                                    os.makedirs(output_folder)

                # retrieve and load csv for results
                data_folder_path = os.path.join(
                    base_dir, f"experiment_{exp}", arch, vis_technique
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
            
                metric_val = df[metric]
                frame_id = df["frame_id"]
                video_name = df["input_vid_idx"]

                s = df.pivot_table(index="frame_id", columns="input_vid_idx", values=metric, aggfunc='mean')
                pivot_list.append(s)

        # labels hardcoded from order of loop above
        ax = (pivot_list[0]).mean(1).plot(color='b', label='Slowfast SLOW')
        (pivot_list[1]).mean(1).plot(ax=ax, color = 'g', label='Slowfast FAST')
        (pivot_list[2]).mean(1).plot(ax=ax, color = 'r', label='I3D')
        (pivot_list[3]).mean(1).plot(ax=ax, color = 'y', label='I3D NLN')
        ax.legend()
        ax.set_xlabel("frame id")
        ax.set_ylabel({metric})

        file_path = os.path.join(output_folder, f"MULTI_MODEL_frames_vs_{metric}_.png")
        plt.savefig(file_path)
        plt.close()
      
def gen_all_multi_model_frame_vs_metric_plots():  
    for exp in experiments:
        for vis_technique in gc_variants:
            for softmax in softmax_status:
                multi_model_frame_vs_metric_plot(
                    exp,
                    vis_technique,
                    softmax,
                    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
                )

def frame_vs_activation_plot(
    exp,
    vis_technique,
    softmax,
    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
):
    
    output_base_folder = "/research/cwloka/data/action_attn/alex_synthetic"
    base_dir = os.path.join(exp_base_dir, "metric_results")

    for arch in architectures:
        if arch == "slowfast":
                channels = ["slow", "fast"]
        elif arch in ["i3d", "i3d_nln"]:
                channels = ["rgb"]
        else:
                raise NotImplementedError("Add in logic for handling channels")
        for channel in channels:

            output_folder = os.path.join(
                                output_base_folder, f"experiment_{exp}", arch, vis_technique, channel
                            )

            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

            # retrieve and load csv for results
            data_folder_path = os.path.join(
                base_dir, f"experiment_{exp}", arch, vis_technique
            )

            # retrieve and load csv with results
            csv_path = os.path.join(
                data_folder_path, f"exp_{exp}_{arch}_{softmax}_frames.csv"
            )
            df = pd.read_csv(csv_path)
            df.drop_duplicates(inplace=True)
            df = df.loc[df["channel"] == channel] # separate slow and fast channels if needed
            
            print("Creating frame vs activation plots for ", arch, " ", channel)

            framewise_root_dir = os.path.join(
                exp_base_dir,
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
            framewisedf.drop_duplicates(inplace=True)
            
            df["mean_activations"] = framewisedf["mean_activations"]
            pd.testing.assert_series_equal(df["input_vid_idx"], framewisedf["input_vid_idx"], check_index=False)
            pd.testing.assert_series_equal((df["frame_id"] + 1), framewisedf["frame_id"], check_index=False) 
            # frames are 1-indexed in metrics CSV, 0-indexed in framewise activations csv

            df = df.loc[df["correct"] == True] # only use data from correctly-classified videos

            activations_val = df["mean_activations"]
            frame_id = df["frame_id"]
            video_name = df["input_vid_idx"]

            s = df.pivot_table(index="frame_id", columns="input_vid_idx", values="mean_activations", aggfunc='mean')
            ax = s.plot(color='gray', label="_hidden")
            s.mean(1).plot(ax=ax, color='r', linestyle='--', label='Mean')
            ax.get_legend().remove() 
            ax.set_xlabel("frame id")
            ax.set_ylabel("mean activation")

            file_path = os.path.join(output_folder, f"frames_vs_mean_activation_.png")
            plt.savefig(file_path)
            plt.close()  

def gen_all_single_model_frame_vs_activation_plots():  
    for exp in experiments:
        for vis_technique in gc_variants:
            for softmax in softmax_status:
                frame_vs_activation_plot(
                    exp,
                    vis_technique,
                    softmax,
                    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
                )  

def multi_model_frame_vs_activation_plot(
    exp,
    vis_technique,
    softmax,
    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
):
    
    output_base_folder = "/research/cwloka/data/action_attn/alex_synthetic"
    base_dir = os.path.join(exp_base_dir, "metric_results")

    pivot_list = []

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
                base_dir, f"experiment_{exp}", arch, vis_technique
            )

            # retrieve and load csv with results
            csv_path = os.path.join(
                data_folder_path, f"exp_{exp}_{arch}_{softmax}_frames.csv"
            )
            df = pd.read_csv(csv_path)
            df.drop_duplicates(inplace=True)
            df = df.loc[df["channel"] == channel] # separate slow and fast channels if needed

            framewise_root_dir = os.path.join(
                exp_base_dir,
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
            framewisedf.drop_duplicates(inplace=True)
            
            df["mean_activations"] = framewisedf["mean_activations"]
            pd.testing.assert_series_equal(df["input_vid_idx"], framewisedf["input_vid_idx"], check_index=False)
            pd.testing.assert_series_equal((df["frame_id"] + 1), framewisedf["frame_id"], check_index=False) 
            # frames are 1-indexed in metrics CSV, 0-indexed in framewise activations csv

            df = df.loc[df["correct"] == True] # only use data from correctly-classified videos
            if channel == "slow":
                df["frame_id"] *= 4 # slow channel has 1/4 the frame rate of all other channels
        
            metric_val = df["mean_activations"]
            frame_id = df["frame_id"]
            video_name = df["input_vid_idx"]

            s = df.pivot_table(index="frame_id", columns="input_vid_idx", values="mean_activations", aggfunc='mean')
            pivot_list.append(s)

            print("Creating frame vs activation plots for ", arch, " ", channel)

    # labels hardcoded from order of loop above
    ax = (pivot_list[0]).mean(1).plot(color='b', label='Slowfast SLOW')
    (pivot_list[1]).mean(1).plot(ax=ax, color = 'g', label='Slowfast FAST')
    (pivot_list[2]).mean(1).plot(ax=ax, color = 'r', label='I3D')
    (pivot_list[3]).mean(1).plot(ax=ax, color = 'y', label='I3D NLN')
    ax.legend()
    ax.set_xlabel("frame id")
    ax.set_ylabel("mean activation")

    file_path = os.path.join(output_folder, f"frames_vs_activations_.png")
    plt.savefig(file_path)
    plt.close()

def gen_all_multi_model_frame_vs_activation_plots():  
    for exp in experiments:
        print("experiment", exp)
        for vis_technique in gc_variants:
            for softmax in softmax_status:
                multi_model_frame_vs_activation_plot(
                    exp,
                    vis_technique,
                    softmax,
                    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
                )

######################################################################################################
# Functions that we are *NOT* currently using! (don't want to delete yet in case useful later) #
######################################################################################################

def accuracy_divided_metrics(
    dataframe,
    experiment,
    architecture,
    gc_variant,
    softmax,
    channel,
    output_folder,
    motion_class=None,
    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
):

    for metric in metrics:
        # filter the dataframe by channel and metric
        metric_filtered = dataframe.loc[dataframe["channel"] == channel]
        metric_filtered = metric_filtered[metric]

        # split into subsets (correctly and incorrectly classified videos/frames)
        metric_accurate = dataframe.loc[dataframe["correct"] == True]
        metric_inaccurate = dataframe.loc[dataframe["correct"] == False]

        metric_stats_accurate = metric_accurate.describe().transpose()
        metric_stats_inaccurate = metric_inaccurate.describe().transpose()
        metric_stats_accurate = metric_stats_accurate.drop(["experiment", "input_vid_idx", "label_numeric", "pred_numeric", "frame_id"])
        metric_stats_inaccurate = metric_stats_inaccurate.drop(["experiment", "input_vid_idx", "label_numeric", "pred_numeric", "frame_id"])
    
        return metric_stats_accurate, metric_stats_inaccurate


def make_accuracy_divided():

    """ probably unnecessary function
    created 7/9/24 
    gets metrics stats for successfully-identified inputs vs incorrectly-identified inputs
    interestingly, are almost identical! 
    working under the assumption that incorrectly-identified inputs are not useful data 
    """
    #################################################
    #     booleans to control what is generated     #
    #################################################

    use_accuracy_divided_boxplot = True
    filter_high_activations = True

    # base directories
    exp_base_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
    output_base_folder = (
        "/research/cwloka/data/action_attn/alex_synthetic"
    )
    base_dir = os.path.join(exp_base_dir, "metric_results")

    for exp in experiments:
        for arch in architectures:
            if arch == "slowfast":
                channels = ["slow", "fast"]
            elif arch in ["i3d", "i3d_nln"]:
                channels = ["rgb"]
            else:
                raise NotImplementedError("Add in logic for handling channels")
            for vis_technique in gc_variants:
                for softmax in softmax_status:
                    for channel in channels:

                        # create the plot output folder
                        output_folder = os.path.join(
                            output_base_folder, f"experiment_{exp}", arch, vis_technique
                        )
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)

                        # retrieve and load csv for results
                        data_folder_path = os.path.join(
                            base_dir, f"experiment_{exp}", arch, vis_technique
                        )

                        # retrieve and load csv with results
                        csv_path = os.path.join(
                            # data_folder_path, f"exp_{exp}_{arch}_{softmax}.csv"
                            # TODO: switch back to use 3D volumes
                            data_folder_path, f"exp_{exp}_{arch}_{softmax}_frames.csv"
                        )
                        df = pd.read_csv(csv_path)
                        df.drop_duplicates(inplace=True)

                        framewise_csv_path = os.path.join(exp_base_dir,
                            f"experiment_{exp}/{arch}_output/heatmaps_epoch_00050/{vis_technique}/{softmax}/{channel}_framewise_activations.csv")

                        framewisedf = pd.read_csv(framewise_csv_path)
                        framewisedf.drop_duplicates(inplace=True)

                        df["mean_activations"] = framewisedf["mean_activations"]

                        print("made activations column!")
                        print(df.head())

                        pd.testing.assert_series_equal(df["input_vid_idx"], framewisedf["input_vid_idx"])
                        pd.testing.assert_series_equal((df["frame_id"] + 1), framewisedf["frame_id"]) # frames are 1-indexed in one CSV< 0-indexed in other

                        if filter_high_activations:
                            df = df.loc[df["mean_activations"] > 5]

                        print("df is dimensions ", df.shape)
                        
                        accurate_df, inaccurate_df = accuracy_divided_metrics(
                                df,
                                exp,
                                arch,
                                vis_technique,
                                softmax,
                                channel,
                                output_folder,
                                motion_class=None,
                            )

                        print("accurate-classification subset statistics below:")
                        print(accurate_df.head())
                        print()
                        print("INaccurate-classification subset statistics below:")
                        print(inaccurate_df.head())
######
