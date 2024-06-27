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


def set_motion_classes(
    exp,
    exp_base_dir="/research/cwloka/data/action_attn/synthetic_motion_experiments",
    label_json_name="synthetic_motion_labels.json",
):
    """
    Retrieves the motion classes for the specific experiment
    Args:
        exp: experiment number
        exp_base_dir: base directory for the experimental results, defaults to
            the Synthetic Motion directory
        label_json_name: name for the json file containing the labels, defaults to
            the labels for Synthetic Motion
    Returns:
        list consisting of the motion classes for the specific experiment
    """
    json_path = os.path.join(exp_base_dir, f"experiment_{exp}", label_json_name)
    with open(json_path, "r") as f:
        json_contents = json.load(f)
    return list(json_contents.keys())


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


def framewise_activation(
    heatmap_dir: str,
):
    """
    Takes in a path to a folder corresponding to the heatmaps for a single
    video, and calculates the framewise mean activation
    Args:
        heatmap_dir (str): path to heatmap folder
    Returns:
        numpy array of the mean activations for each frame
    """
    if not os.path.exists(heatmap_dir):
        raise TypeError("Invalid heatmap path")

    heatmap_vol = load_heatmaps(heatmap_dir)
    raw_activations = np.mean(heatmap_vol, axis=(1, 2))
    return raw_activations


def isolate_metric_values(
    metric: str,
    motion_class: str,
    input_vid_idx: int,
    frames_df,
):
    """
    Isolates entries of the dataframe according to the parameters, returning
    the result as a numpy array
    Args:
        metric (str): metric column to isolate
        motion_class (str): motion class for the entry to isolate
        input_vid_idx (str): the video index to isolate
        frames_df (pandas DataFrame): dataframe containing the entry
    Returns:
        numpy array containing the results of the filtered dataframe
    """
    filtered_df = frames_df.loc[
        (frames_df["label"] == motion_class)
        & (frames_df["input_vid_idx"] == input_vid_idx)
    ]
    filtered_df = filtered_df[[metric]]
    return filtered_df.to_numpy().flatten()


def retrieve_json(
    experiment_num,
    root_exp_dir="/research/cwloka/data/action_attn/synthetic_motion_experiments",
):
    """
    Takes in an experiment number and directory to experiment folder and
    returns the contents of the json corresponding to testing

    Args:
        experiment_num: experimental number to retrieve json from
        root_exp_dir: directory for experimental results, defaults to synthetic
            motion experiments
    Returns:
        List of dictionaries consisting of the JSON entries
    """
    json_directory = os.path.join(
        root_exp_dir, f"experiment_{experiment_num}", f"synthetic_motion_test.json"
    )
    if not os.path.exists(json_directory):
        raise TypeError("Invalid path; check experiment_num or root_exp_dir")
    with open(json_directory, "r") as f:
        full_json = json.load(f)
    return full_json


def vid_id_to_idx(video_id: str) -> (str, int):
    """
    Converts a video_id from [class]_[index:06d] into a (str, int) tuple
    consisting of (motion class, video index)
    """
    split_id = video_id.split("_")
    motion_class = split_id[0]
    vid_idx = int(split_id[1])
    return (motion_class, vid_idx)


def single_energy_plot(
    metric_values,
    activation_values,
    metric: str,
    motion_class=None,
    output_dir="/research/cwloka/projects/nikki_sandbox/action_attention/metric_exploration/plots",
):
    """
    Makes an energy plot where framewise metric values can be compared
    alongside the framewise activation values from a GradCAM technique

    Args:
        metric_values: list of the average metric values per frame
        activation_values: list of the average framewise activation per frame
        metric: metric whose results metric_results correspond to
        motion_class: current motion class, defaults to None (meaning metric_results
            and activation_values were computed according to the entire experimental
            configuration)
        output_dir: directory to output the plot to, TODO add in default location
    """
    # verify that metric_values and activation_values have the same length
    assert len(metric_values) == len(activation_values)

    # create the plot and set the title
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    if motion_class is None:
        fig.suptitle(f"{metric} and Activation vs Frame Number")
    else:
        fig.suptitle(f"{metric} and Activation vs Frame Number ({motion_class})")

    # plot the metric values
    ax1.plot(metric_values, "o-")
    ax1.set_ylabel(metric)
    ax1.set_xlabel("Frame Index")
    ax1.set_title(f"{metric} vs Frame Index")

    # plot the activation values
    ax2.plot(activation_values, "o-r")
    ax2.set_ylabel("Activation value")
    ax2.set_xlabel("Frame index")
    ax2.set_title("Raw Activation vs Frame Index")

    # set the plot name to 'all' if a motion class is not specified
    if motion_class is None:
        plot_name = "all"
    else:
        plot_name = motion_class
    file_path = os.path.join(output_dir, f"energy_plot_{metric}_{plot_name}.png")
    plt.savefig(file_path)
    plt.close()


def count_frames(dataframe, channel, max_val_to_search=100) -> int:
    """
    Finds the maximum frame index for a specific channel within the dataframe
    Args:
        dataframe: pandas dataframe to search for max frame id in
        channel: channel to search
        max_val_to_search: maximum number of rows to inspect
    Returns:
        int, number of frames in all of the videos
    """
    mini_df = dataframe.iloc[:max_val_to_search]
    mini_df = mini_df[mini_df["channel"] == channel]
    mini_df = mini_df[["frame_id"]]
    num_frames = mini_df.max().iloc[0] + 1
    return int(num_frames)


def store_framewise_activations(
    experiment,
    architecture,
    gc_variant,
    softmax,
    channel,
    exp_base_dir="/research/cwloka/data/action_attn/synthetic_motion_experiments",
):
    """
    Calculates and stores the framewise average activations for a specific
    experimental configuration. Does not have logic to handle a specific epoch
    to pull results from.
    Args:
        experiment: experiment number
        architecture: model architecture
        gc_variant: GradCAM variant
        softmax: where gradients are taken; either pre_softmax or post_softmax
        channel: input channel for the architecture
        exp_base_dir: base directory for experimental results
    """
    model = ""
    non_local = False
    if "i3d" in architecture:
        model = "i3d"
        if "nln" in architecture:
            non_local = True
    elif "slowfast" in architecture:
        model = "slowfast"
    else:
        raise NotImplementedError("Add in logic for architecture")

    post_softmax_status = True if "post" in softmax else False

    exp_config_root_dir = os.path.join(
        exp_base_dir,
        f"experiment_{experiment}",
        f"{architecture}_output",
    )

    results_folder = ""
    for entry in os.listdir(exp_config_root_dir):
        if "heatmaps_epoch_" in entry:
            results_folder = entry
    exp_config_folder = os.path.join(
        exp_config_root_dir, results_folder, gc_variant, softmax
    )
    heatmap_root_dir = os.path.join(exp_config_folder, "frames")
    json_contents = retrieve_json(experiment)

    # number of frames * number of videos
    data_dict = {
        "experiment": [],
        "model": [],
        "nonlocal": [],
        "gradcam_variant": [],
        "post_softmax": [],
        "input_vid_idx": [],
        "frame_id": [],
        "mean_activations": [],
        "label": [],
        "channel": [],
    }

    print("Starting framewise activation calculations")
    count = 0
    for entry_dict in json_contents:
        print(count)
        count += 1
        (video_class, video_idx) = vid_id_to_idx(entry_dict["video_id"])
        heatmap_path = os.path.join(
            heatmap_root_dir, video_class, entry_dict["video_id"], channel
        )
        framewise_activations = framewise_activation(heatmap_path)
        num_frames = len(framewise_activations)
        if len(data_dict["experiment"]) == 0:
            data_dict["mean_activations"] = framewise_activations.tolist()
            data_dict["experiment"] = [experiment] * num_frames
            data_dict["model"] = [model] * num_frames
            data_dict["nonlocal"] = [non_local] * num_frames
            data_dict["post_softmax"] = [post_softmax_status] * num_frames
            data_dict["gradcam_variant"] = [gc_variant] * num_frames
            data_dict["input_vid_idx"] = [video_idx] * num_frames
            data_dict["label"] = [video_class] * num_frames
            data_dict["frame_id"] = [x + 1 for x in range(num_frames)]
            data_dict["channel"] = [channel] * num_frames
        else:
            data_dict["mean_activations"] += framewise_activations.tolist()
            data_dict["experiment"] += [experiment] * num_frames
            data_dict["model"] += [model] * num_frames
            data_dict["nonlocal"] += [non_local] * num_frames
            data_dict["post_softmax"] += [post_softmax_status] * num_frames
            data_dict["gradcam_variant"] += [gc_variant] * num_frames
            data_dict["input_vid_idx"] += [video_idx] * num_frames
            data_dict["label"] += [video_class] * num_frames
            data_dict["frame_id"] += [x + 1 for x in range(num_frames)]
            data_dict["channel"] += [channel] * num_frames

    results_dataframe = pd.DataFrame.from_dict(data_dict)

    print("Results are now saving")
    output_path = os.path.join(
        exp_config_folder, f"{channel}_framewise_activations.csv"
    )
    results_dataframe.to_csv(output_path, index=False)


def energy_plots(
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
    Creates energy plots for each metric within a specific experimental
    configuration and stores them in the specified output folder

    Args:
        dataframe: pandas dataframe containing metric results
        experiment: experiment number
        architecture: model architecture
        gc_variant: GradCAM variant
        softmax: whether gradients were taken pre or post softmax
        output_folder: folder to store plots in
        motion_class: motion class to retrieve results from, default None
            (meaning results should be averaged across all motion classes)
        metrics: metrics to use results from, defaults to
            ["kl_div", "iou", "pearson", "mse", "covariance"]
    """
    # number of videos
    num_items = 0

    # number of frames in each video
    num_frames = count_frames(dataframe, channel)

    # initialize the lists to store the total sums of activations
    activation_totals = [0] * num_frames
    metric_totals = {}
    for metric in metrics:
        metric_totals[metric] = [0] * num_frames

    # find the folder where the heatmap activations are stored
    exp_root_dir = os.path.join(
        "/research/cwloka/data/action_attn/synthetic_motion_experiments",
        f"experiment_{experiment}",
        f"{architecture}_output",
    )
    results_folder = ""
    for entry in os.listdir(exp_root_dir):
        if "heatmaps_epoch_" in entry:
            results_folder = entry
    heatmap_root_dir = os.path.join(
        exp_root_dir, results_folder, gc_variant, softmax, "frames"
    )

    # retrieve the json containing infomration about testing
    json_contents = retrieve_json(experiment)

    print("Beginning iteration through testing JSON")
    for entry_dict in json_contents:
        num_items += 1
        print("\tCurrent index: ", num_items)
        (video_class, video_idx) = vid_id_to_idx(entry_dict["video_id"])
        heatmap_path = os.path.join(
            heatmap_root_dir, video_class, entry_dict["video_id"], channel
        )

        # calculate framewise activations
        framewise_activations = framewise_activation(heatmap_path)
        for i in range(num_frames):
            activation_totals[i] += framewise_activations[i]

        # calculate metric values
        for metric in metrics:
            metric_values = isolate_metric_values(
                metric, video_class, video_idx, dataframe
            )
            for i in range(num_frames):
                metric_totals[metric][i] += metric_values[i]
    print("Finished iterating through JSON")

    # calculate the framewise averages, then plot the results
    for metric in metrics:
        framewise_averages = list(map(lambda x: x / num_items, activation_totals))
        metric_averages = list(map(lambda x: x / num_items, metric_totals[metric]))
        single_energy_plot(
            metric_averages,
            framewise_averages,
            metric,
            motion_class=motion_class,
            output_dir=output_folder,
        )


def main():
    # conditions to generate plots
    experiments = [1, 2, 3, 4, 5, "5b"]
    architectures = ["slowfast", "i3d", "i3d_nln"]
    architectures = ["i3d", "i3d_nln"]
    gc_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
    softmax_status = ["pre_softmax", "post_softmax"]
    metrics = ["kl_div", "iou", "pearson", "mse", "covariance"]

    # booleans for whether or not to create framewise energy plots,
    # non-framewise plots, and whether to create separate plots for each
    # motion class
    use_energy = False
    use_single_plots = False
    use_motion_classes = False
    calc_framewise_activations = True

    # base directories
    exp_base_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
    output_base_folder = "/research/cwloka/projects/nikki_sandbox/action_attention/metric_exploration/plots"
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
                        """
                            experiment,
                            architecture,
                            gc_variant,
                            softmax,
                            channel,
                            exp_base_dir="/research/cwloka/data/action_attn/synthetic_motion_experiments",
                        """
                        if calc_framewise_activations:
                            store_framewise_activations(
                                exp,
                                arch,
                                vis_technique,
                                softmax,
                                channel,
                                exp_base_dir = exp_base_dir,
                            )

                        # create the plot output folder
                        output_folder = os.path.join(
                            output_base_folder,
                            f"experiment_{exp}",
                            arch,
                            vis_technique,
                            softmax,
                            channel,
                        )
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)

                        # retrieve and load csv for results
                        data_folder_path = os.path.join(
                            base_dir, f"experiment_{exp}", arch, vis_technique
                        )
                        if use_energy:
                            frames_csv_path = os.path.join(
                                data_folder_path,
                                f"exp_{exp}_{arch}_{softmax}_frames.csv",
                            )
                            df = pd.read_csv(frames_csv_path)
                            df.drop_duplicates(inplace=True)

                            if use_motion_classes:
                                for motion_class in motion_classes:
                                    energy_plots(
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
                                energy_plots(
                                    df,
                                    exp,
                                    arch,
                                    vis_technique,
                                    softmax,
                                    channel,
                                    output_folder,
                                    motion_class=None,
                                )
                        if use_single_plots:
                            # retrieve and load csv with results
                            csv_path = os.path.join(
                                data_folder_path, f"exp_{exp}_{arch}_{softmax}.csv"
                            )
                            df = pd.read_csv(csv_path)
                            df.drop_duplicates(inplace=True)
                            if use_motion_classes:
                                for motion_class in motion_classes:
                                    single_plot(
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
                                single_plot(
                                    df,
                                    exp,
                                    arch,
                                    vis_technique,
                                    softmax,
                                    channel,
                                    output_folder,
                                    motion_class=None,
                                )
                        # pdb.set_trace()


if __name__ == "__main__":
    main()
