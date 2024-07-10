"""
Contains functionality to create plots for metric results
"""

import os
import pdb
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from slowfast.visualization.connected_components_utils import load_heatmaps

motion_classes = []

# def set_motion_classes(
#     exp,
#     exp_base_dir="/research/cwloka/data/action_attn/synthetic_motion_experiments",
#     label_json_name="synthetic_motion_labels.json",
# ):
#     """
#     Retrieves the motion classes for the specific experiment
#     Args:
#         exp: experiment number
#         exp_base_dir: base directory for the experimental results, defaults to
#             the Synthetic Motion directory
#         label_json_name: name for the json file containing the labels, defaults to
#             the labels for Synthetic Motion
#     Returns:
#         list consisting of the motion classes for the specific experiment
#     """
#     json_path = os.path.join(exp_base_dir, f"experiment_{exp}", label_json_name)
#     with open(json_path, "r") as f:
#         json_contents = json.load(f)
#     return list(json_contents.keys())


###### ADDED 7/9/24

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
        # filter the dataframe by channel, label, and metric
        metric_filtered = dataframe.loc[dataframe["channel"] == channel]
        if motion_class is not None:
            metric_filtered = dataframe.loc[dataframe["label"] == motion_class]
        metric_filtered = metric_filtered[metric]

        # split into subsets (correctly and incorrectly classified videos/frames)
        metric_accurate = dataframe.loc[dataframe["correct"] == True]
        metric_inaccurate = dataframe.loc[dataframe["correct"] == False]
        # print(metric_accurate.head())
        # print(metric_inaccurate.head())

        metric_stats_accurate = metric_accurate.describe().transpose()
        metric_stats_inaccurate = metric_inaccurate.describe().transpose()

        metric_stats_accurate = metric_stats_accurate.drop(["experiment", "input_vid_idx", "label_numeric", "pred_numeric", "frame_id"])
        metric_stats_inaccurate = metric_stats_inaccurate.drop(["experiment", "input_vid_idx", "label_numeric", "pred_numeric", "frame_id"])
    
        return metric_stats_accurate, metric_stats_inaccurate


def single_plot(
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
    """
    Generates a histogram and boxplot from the results of metric
    calculations for an experimental configuration.

    Args:
        dataframe: pandas dataframe consisting of experimental-config-wide
            results (must not be filtered by channel, label, or metric yet)
        experiment: number corresponding to the experiment
        architecture: current model architecture
        gc_variant: current GradCAM variant
        softmax: pre or post softmax
        channel: current input channel
        output_folder: base folder to save plots to
        motion_class: current motion class, defaults to None
        metrics: list metric results to analyze, defaults to
            ["kl_div", "iou", "pearson", "mse", "covariance"]
    """
    for metric in metrics:
        # filter the dataframe by channel, label, and metric
        metric_filtered = dataframe.loc[dataframe["channel"] == channel]
        if motion_class is not None:
            metric_filtered = dataframe.loc[dataframe["label"] == motion_class]
        metric_filtered = metric_filtered[metric]

        if motion_class is None:
            category = "all"
        else:
            category = motion_class

        print("Creating Histograms")
        fig = plt.hist(metric_filtered, bins=20, log=True)
        plt.xlabel(f"{metric} between GT and actual activation (log scale)")
        plt.ylabel("frequency")
        plt.title(f"Experiment {experiment}: Distribution of {metric} values")
        file_path = os.path.join(output_folder, f"histogram_{metric}_{category}.png")
        plt.savefig(file_path)
        plt.close()

        print("Creating Box and Whiskers")
        fig = plt.boxplot(metric_filtered, vert=False)
        plt.xlabel(f"{metric} between GT and actual activation (log scale)")
        plt.title(f"Experiment {experiment}: Distribution of {metric} values")
        file_path = os.path.join(
            output_folder, f"box_and_whisker_{metric}_{category}.png"
        )
        plt.savefig(file_path)
        plt.close()

def make_accuracy_divided():

    """ probably unnecessary function
    created 7/9/24 
    gets metrics stats for successfully-identified inputs vs incorrectly-identified inputs
    interestingly, are almost identical! 
    working under the assumption that incorrectly-identified inputs are not useful data 
    """

 # conditions to generate plots
    experiments = [1]
    architectures = ["i3d"]
    gc_variants = ["grad_cam"]
    softmax_status = ["pre_softmax", "post_softmax"]
    metrics = ["kl_div", "iou", "pearson", "mse", "covariance"]
    
    # experiments = [1, 2, 3, 4, 5, "5b"]
    # architectures = ["slowfast", "i3d", "i3d_nln"]
    # gc_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
    # softmax_status = ["pre_softmax", "post_softmax"]
    # metrics = ["kl_div", "iou", "pearson", "mse", "covariance"]

    #################################################
    #     booleans to control what is generated     #
    #################################################

    use_motion_classes = False  # whether all values should be done per motion class
    use_accuracy_divided_boxplot = True
    filter_high_activations = True

    # base directories
    exp_base_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
    output_base_folder = (
        "/research/cwloka/data/action_attn/alex_synthetic"
    )
    base_dir = os.path.join(exp_base_dir, "metric_results")

    for exp in experiments:
        if use_motion_classes:
            # set the global list of motion classes
            motion_classes = set_motion_classes(exp, exp_base_dir)
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
                             
                        if use_motion_classes:
                            for motion_class in motion_classes:
                                accurate_df, inaccurate_df = accuracy_divided_metrics(
                                    df,
                                    exp,
                                    arch,
                                    vis_technique,
                                    softmax,
                                    channel,
                                    output_folder,
                                    motion_class=motion_class,
                                )
                                
                        else:
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

def frame_vs_metric_plot(
    exp,
    vis_technique,
    softmax,
    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
):
    
    architectures = ["slowfast", "i3d", "i3d_nln"]

    exp_base_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
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

            for metric in metrics:
                print("Creating frame vs metric plots for ", arch, " ", channel, " and ", metric)

                metric_val = df[metric]
                frame_id = df["frame_id"]
                video_name = df["input_vid_idx"]

                s = df.pivot_table(index="frame_id", columns="input_vid_idx", values=metric, aggfunc='mean')
                ax = s.plot(color='gray', linestyle='None', label="_hidden")
                s.mean(1).plot(ax=ax, color='b', linestyle='--', label='Mean')
                ax.get_legend().remove() 
                ax.set_xlabel("frame id")
                ax.set_ylabel({metric})

                file_path = os.path.join(output_folder, f"frames_vs_{metric}_.png")
                plt.savefig(file_path)
                plt.close()    


def multi_model_frame_vs_metric_plot(
    exp,
    vis_technique,
    softmax,
    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
):

    architectures = ["slowfast", "i3d", "i3d_nln"]

    exp_base_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
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
                                    output_base_folder, f"experiment_{exp}"
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
            
                metric_val = df[metric]
                frame_id = df["frame_id"]
                video_name = df["input_vid_idx"]

                s = df.pivot_table(index="frame_id", columns="input_vid_idx", values=metric, aggfunc='mean')
                pivot_list.append(s)


        # labels hardcoded from order of loop above, ideally would switch to be labeled by variable
        # TODO is this a problem? ^^
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
      


