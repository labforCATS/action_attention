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


def multiple_box_plot(
    metric,
    architecture,
    gc_variant,
    softmax,
    channel,
    output_folder,
    motion_class = None,
    exp_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments/metric_results",
    experiments = [1,2, 3, 4,5 ,"5b"],
):
    """
    Creates one plot containing multiple box and whisker plots.
    Metric on the y axis and the experiment number on the x-axis.

    REMINDER: When changing what experiments are used above, make sure to 
    change the "file_name" variable below.
    """

    all_exp_metric_filtered = []
    # DIANE PLEASE REMEMBER TO CHANGE THE NAME BEFORE RUNNING PLS
    file_name = "all_experiments"

    # Find correct file with metric calculations
    for exp in experiments:
        csv_path = os.path.join(
            exp_dir, f"experiment_{exp}", architecture, gc_variant,
            f"exp_{exp}_{architecture}_{softmax}.csv"
        )
        df = pd.read_csv(csv_path)

        # filter for only correctly labeled videos
        metric_filtered = df.loc[df["channel"] == channel]
        metric_filtered = metric_filtered.loc[df["correct"] == True]

        if motion_class is not None:
            metric_filtered = df.loc[df["label"] == motion_class]
        
        # Look only at the correct metric and convert into list
        metric_filtered = metric_filtered[metric]
        metric_filtered = metric_filtered.tolist()

        # Get rid of all nan values before plotting
        nan_filtered = []
        for x in metric_filtered:
            if x == x:
                nan_filtered.append(x)

        all_exp_metric_filtered.append(nan_filtered)


    if motion_class is None:
        category = "all"
    else:
        category = motion_class

    print("Configuration: ", f"{metric} with {architecture} {channel} and {gc_variant}")  

    # output pathway for plots
    file_path = os.path.join(
        output_folder, file_name, f"{architecture}", f"{channel}", f"{gc_variant}",
        f"{softmax}"   
    )
    # create output directory if it doesn't exist already
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # plot!!
    fig,ax = plt.subplots()

    for i,lst in enumerate(all_exp_metric_filtered): 
        ax.boxplot(lst,positions=[i]) 

    plt.xlabel(f"Experiment")
    plt.ylabel(metric)
    # relabel x-axis with experiment numbers
    plt.xticks(range(len(experiments)), experiments)
    # plt.title(f"Distribution of {metric} values over all Experiments")

    plt.savefig(os.path.join(file_path,f"boxplot_{metric}.png"))
    plt.close()
    # pdb.set_trace()




def diane_main():
    # conditions to generate plots
    experiments = [1, 2, 3, 4, 5, "5b"]
    architectures = ["i3d", "slowfast"] #["slowfast", "i3d", "i3d_nln"]
    gc_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
    softmax_status = ["pre_softmax", "post_softmax"] #["pre_softmax", "post_softmax"]
    metrics = ["kl_div", "iou", "pearson", "mse", "covariance"]

    #################################################
    #     booleans to control what is generated     #
    #################################################
    use_energy = False  # create energy plots
    use_single_plots = False  # create box + whisker plots and histograms
    use_multiple_plots = True
    use_motion_classes = False  # whether all values should be done per motion class
    calc_framewise_activations = False  # create new framewise activation CSV's

    # base directories
    exp_base_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
    output_base_folder = (
        "/research/cwloka/projects/diane_sandbox/synthetic_output/boxplots/"
    )
    base_dir = os.path.join(exp_base_dir, "metric_results")

    for arch in architectures:
        for cam in gc_variants:
            for softmax in softmax_status:
                for metric in metrics:
                    
                    if (arch == "slowfast"):
                        for channel in ["slow", "fast"]:
                            multiple_box_plot(
                                metric,
                                arch,
                                cam,
                                softmax,
                                channel,
                                output_base_folder,
                            )
                    else:
                            multiple_box_plot(
                                metric,
                                arch,
                                cam,
                                softmax,
                                "rgb",
                                output_base_folder,
                            ) 
                            #pdb.set_trace()















# if __name__ == "__main__":
#     main()
