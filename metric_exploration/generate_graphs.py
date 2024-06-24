import os
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from slowfast.visualization.connected_components_utils import load_heatmaps


def single_plot(
    dataframe,
    experiment,
    architecture,
    gc_variant,
    softmax,
    output_folder,
    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
):
    """
    single_plot generates a histogram and boxplot for each calculated metric
    value for a given dataframe

    """
    for metric in metrics:
        metric_filtered = dataframe[metric]

        print("Creating Histograms")
        fig = plt.hist(metric_filtered, bins=20, log=True)
        plt.xlabel(f"{metric} between GT and actual activation (log scale)")
        plt.ylabel("frequency")
        plt.title(f"Experiment {experiment}: Distribution of {metric} values")
        file_path = os.path.join(
            output_folder,
            f"exp_{experiment}_{architecture}_{gc_variant}_{softmax}_{metric}_histogram.png",
        )
        plt.savefig(file_path)
        plt.close()

        print("Creating Box and Whiskers")
        fig = plt.boxplot(metric_filtered, vert=False)
        plt.xlabel(f"{metric} between GT and actual activation (log scale)")
        plt.title(f"Experiment {experiment}: Distribution of {metric} values")
        file_path = os.path.join(
            output_folder,
            f"exp_{experiment}_{architecture}_{gc_variant}_{softmax}_{metric}_boxplot.png",
        )
        plt.savefig(file_path)
        plt.close()


def calculate_framewise_activation(
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
        # TODO: decide if it should be a fatal error or just ignore
        raise TypeError("Invalid path")
    heatmap_vol = load_heatmaps(heatmap_dir)
    raw_activations = np.mean(heatmap_vol, axis=(1, 2))
    return raw_activations


def create_energy_plot(
    metric_values,
    activation_values,
    metric: str,
    output_dir="/research/cwloka/projects/nikki_sandbox/action_attention/metric_exploration/plots",
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"{metric} and Raw Activation vs Index")

    ax1.plot(metric_values, "o-")
    ax1.set_ylabel(metric)
    ax1.set_xlabel("Frame Index")
    ax1.set_title(f"{metric} vs Frame Index")

    ax2.plot(activation_values, "o-r")
    ax2.set_ylabel("Activation value")
    ax2.set_xlabel("Frame index")
    ax2.set_title("Raw Activation vs Frame Index")

    # TODO: come up with a naming convention for the plots (may need logic for plot type)
    file_path = os.path.join(output_dir, "energy_plot.png")
    plt.savefig(file_path)
    plt.close()


def plot_single_vid_energy(
    dataframe,
    figure,
):
    """
    isolate each of the videos within the motion class
    have a list of the motion classes?

    """
    return


def main():
    ## iterate through the experimental conditions, call framewise
    ## and full metrics
    experiments = [1, 2, 3, 4, 5, "5b"]
    architectures = ["slowfast", "i3d", "i3d_nln"]
    gc_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
    softmax_status = ["pre_softmax", "post_softmax"]
    metrics = ["kl_div", "iou", "pearson", "mse", "covariance"]
    use_frames = False
    use_single_plots = True

    output_folder = "/research/cwloka/projects/nikki_sandbox/action_attention/metric_exploration/plots"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    base_dir = (
        "/research/cwloka/data/action_attn/synthetic_motion_experiments/metric_results"
    )
    for exp in experiments:
        for arch in architectures:
            for vis_technique in gc_variants:
                for softmax in softmax_status:
                    data_folder_path = os.path.join(
                        base_dir, f"experiment_{exp}", arch, vis_technique
                    )
                    if use_frames:
                        frames_csv_path = os.path.join(
                            data_folder_path, f"exp_{exp}_{arch}_{softmax}_frames.csv"
                        )
                        df = pd.read_csv(frames_csv_path)
                        # frames_plot()
                        print("make sure to call framewise plots once fixed")
                    if use_single_plots:
                        csv_path = os.path.join(
                            data_folder_path, f"exp_{exp}_{arch}_{softmax}.csv"
                        )
                        df = pd.read_csv(csv_path)
                        single_plot(
                            df, exp, arch, vis_technique, softmax, output_folder
                        )


if __name__ == "__main__":
    main()
