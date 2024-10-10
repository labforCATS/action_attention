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
    experiments = [ 4, "5b"],
):
    """
    Creates one plot containing multiple box and whisker plots.
    Metric on the y axis and the experiment number on the x-axis.

    REMINDER: When changing what experiments are used above, make sure to 
    change the "file_name" variable below.
    """

    all_exp_metric_filtered = []
    # DIANE PLEASE REMEMBER TO CHANGE THE NAME BEFORE RUNNING PLS
    folder_name = "experiment_4_5b"

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
        output_folder, folder_name, f"{architecture}", f"{channel}", f"{gc_variant}",
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


def mult_config_boxplot(
    experiment,
    architecture,
    channel,
    metric,
    output_folder,
    input_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments/metric_results",
):
    """
    Creates 6 side-by-side box-and-whisker plots. Each of the 6 boxplots corresponds to a combination 
    of CAM technique (eigen-cam, grad-cam, or grad-cam ++) and softmax status (pre or post softmax).
    Each function call analyzes one architecture (and channel) and experiment.

        experiment:     one int or string corresponding to the specific experiment being analyzed
        architecture:   string corresponding to one network architecture
        channel:        string indicating "rgb" channel for i3d or "slow"/"fast" for slowfast
        metric:         string for what metric to use
        output_folder:  string file pathway for where boxplot should be saved
        exp_dir:        string file pathway for where data is drawn from

    """

    # Create list of lists
    all_exp_metric_filtered = []
    # Create list of x-tick labels
    x_labels = []

    # loop through cams and softmaxes
    for gc_variant in ["eigen_cam", "grad_cam", "grad_cam_plusplus"]:
        for softmax in ["pre_softmax", "post_softmax"]:

            # open appropriate csv file
            csv_path = os.path.join(
                input_dir, f"experiment_{experiment}", architecture, gc_variant,
                f"exp_{experiment}_{architecture}_{softmax}.csv"
            )
            df = pd.read_csv(csv_path)

            # filter for corect channel, correct predictions, and metric
            metric_filtered = df.loc[df["channel"] == channel]
            metric_filtered = metric_filtered.loc[df["correct"] == True]
            metric_filtered = metric_filtered[metric]
            metric_filtered = metric_filtered.tolist()

            # filter out nan values
            nan_filtered = []
            for x in metric_filtered:
                if x == x:
                    nan_filtered.append(x)

            # add to the list of lists
            all_exp_metric_filtered.append(nan_filtered)

            # Create shorted naming for x-tick labels
            if (gc_variant == "eigen_cam"):
                vis = "EC"
            elif (gc_variant == "grad_cam"):
                vis = "GC"
            else:
                vis = "GC++"

            if (softmax == "pre_softmax"):
                sm = "pre"
            else:
                sm = "post"
            x_labels.append(f"{vis}_{sm}")
            
    # Create correct output file pathing
    file_path = os.path.join(
        output_folder, "boxplots", f"experiment_{experiment}", f"{architecture}", f"{channel}")
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # PLOT!
    fig,ax = plt.subplots()
    for i,lst in enumerate(all_exp_metric_filtered): 
        ax.boxplot(lst,positions=[i], widths = 0.5)  

    # Label axis
    plt.xlabel(f"Configuration")
    plt.ylabel(metric)

    # relabel x-axis with correct configuration
    plt.xticks(range(len(x_labels)), x_labels)
    ax.tick_params(axis='both', which='major', labelsize=8)

    # save plots
    plt.savefig(os.path.join(file_path,f"all_config_boxplot_{metric}.png"))
    plt.close()


def activation_boxplot(
    experiments,
    architecture,
    gc_variant,
    softmax,
    channel,
    output_folder,
    motion_class = None,
    input_dir =  "/research/cwloka/data/action_attn/synthetic_motion_experiments" 
):
    all_activations = []
    file_name = "experiment"

    for exp in experiments:

        file_name += f"_{exp}"

        framewise_root_dir = os.path.join(
            input_dir,
            f"experiment_{exp}",
            f"{architecture}_output",
        )

        heatmap_folder = ""
        for entry in os.listdir(framewise_root_dir):
            if "heatmaps_epoch_" in entry:
                heatmap_folder = entry
        framewise_csv_path = os.path.join(
            framewise_root_dir, heatmap_folder, gc_variant, softmax, f"{channel}_framewise_activations.csv"
        )
        df = pd.read_csv(framewise_csv_path)
        activations_only = df.loc[df["channel"] == channel]
        activations_only = activations_only["mean_activations"].tolist()

        all_activations.append(activations_only)
    
    file_path = os.path.join(
        output_folder, file_name, f"{architecture}", f"{channel}", f"{gc_variant}", f"{softmax}")
    if not os.path.exists(file_path):
        os.makedirs(file_path)

     # PLOT!
    fig,ax = plt.subplots()
    for i,lst in enumerate(all_activations): 
        ax.boxplot(lst,positions=[i], widths = 0.5)  

    # Label axis
    plt.xlabel(f"Configuration")
    plt.ylabel("activation per frame")

    # relabel x-axis with correct configuration
    plt.xticks(range(len(experiments)), experiments)
    ax.tick_params(axis='both', which='major', labelsize=8)

    # save plots
    plt.savefig(os.path.join(file_path,f"boxplot_activations.png"))
    plt.close()


def precision_recall_boxplot(
    experiments,
    architecture,
    gc_variant,
    softmax,
    channel,
    output_folder,
    motion_class = None,
    input_dir = "/research/cwloka/data/action_attn/diane_synthetic/metric_results/" 
):  
    for metric in ["precision", "recall"]:
        all_metric_values = []
        file_name = "experiment"
        for exp in experiments:
            file_name += f"_{exp}"
            framewise_root_dir = os.path.join(
                input_dir,
                f"experiment_{exp}",
                f"{architecture}",
                f"{gc_variant}"
            )
            framewise_csv_path = os.path.join(
                framewise_root_dir, f"exp_{exp}_{architecture}_{softmax}_frames.csv"
            )
            df = pd.read_csv(framewise_csv_path)

            metric_filtered = df.loc[df["channel"] == channel]   
            metric_filtered = metric_filtered[metric].tolist()
            nan_filtered = []
            for x in metric_filtered:
                if x == x:
                    nan_filtered.append(x)

            all_metric_values.append(nan_filtered)

        file_path = os.path.join(
            output_folder, file_name, f"{architecture}", f"{channel}", f"{gc_variant}", f"{softmax}")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        fig,ax = plt.subplots()
        for i,lst in enumerate(all_metric_values): 
            ax.boxplot(lst,positions=[i], widths = 0.5)  

        # Label axis
        plt.xlabel(f"Experiment")
        plt.ylabel(f"{metric} per frame")

        # relabel x-axis with correct configuration
        plt.xticks(range(len(experiments)), experiments)
        ax.tick_params(axis='both', which='major', labelsize=8)

        # save plots
        plt.savefig(os.path.join(file_path,f"boxplot_{metric}.png"))
        plt.close()

def diane_main():
    """
    This is a main function of sorts but I don't want it to run everytime I run my code.
    By changing the booleans corresponding to different types of boxplots and the conditions,
    you can generate all boxplots associated to the different conditions. 

    This code is to generate ALL plots of a kind. To create just one or a couple, I reccomend
    just doing function calls in ipython terminal. 
    """
    #####################################
    #     WHAT BOXPLOTS DO YOU WANT?    #
    #####################################
    compare_experiments = False
    compare_configs = False
    activation_plots = True
    precision_recall = False
    
    # conditions to generate plots
    experiments = [1, 2, 3, 4, 5, "5b"]                 #[1, 2, 3, 4, 5, "5b"]
    architectures = ["slowfast", "i3d", "i3d_nln"]
    gc_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
    softmax_status = ["pre_softmax", "post_softmax"]
    metrics = ["kl_div", "iou", "pearson", "mse", "covariance"]

    # base directories
    exp_base_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
    output_base_folder = (
        "/research/cwloka/projects/diane_sandbox/synthetic_output/boxplots/"
    )
    base_dir = os.path.join(exp_base_dir, "metric_results")

    if compare_experiments:
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

    if compare_configs:
        for exp in experiments:
            for arch in architectures:
                for metric in metrics:
                    if (arch == "slowfast"):
                        for channel in ["slow", "fast"]:
                            mult_config_boxplot(
                                exp,
                                arch, 
                                channel, 
                                metric, 
                                output_base_folder)
                    else:
                        mult_config_boxplot(
                            exp, 
                            arch, 
                            "rgb", 
                            metric, 
                            output_base_folder)

    if activation_plots:
        for arch in architectures:
            for cam in gc_variants:
                for softmax in softmax_status:

                    if (arch == "slowfast"):
                            for channel in ["slow", "fast"]:
                                activation_boxplot(
                                    experiments,
                                    arch,
                                    cam,
                                    softmax,
                                    channel,
                                    output_base_folder,
                                )
                    else:
                        activation_boxplot(
                            experiments,
                            arch,
                            cam,
                            softmax,
                            "rgb",
                            output_base_folder,
                        ) 

    if precision_recall:
        for arch in architectures:
            for cam in gc_variants:
                for softmax in softmax_status:
                    if (arch == "slowfast"):
                            for channel in ["slow", "fast"]:
                                precision_recall_boxplot(
                                    experiments,
                                    arch,
                                    cam,
                                    softmax,
                                    channel,
                                    output_base_folder,
                                )
                    else:
                        precision_recall_boxplot(
                            experiments,
                            arch,
                            cam,
                            softmax,
                            "rgb",
                            output_base_folder,
                        ) 















# if __name__ == "__main__":
#     main()