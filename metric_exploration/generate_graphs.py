import os
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import pdb


def single_plot(
    dataframe,
    experiment,
    architecture,
    gc_variant,
    softmax,
    output_folder,
    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
):
    
    # add in functionality to pull from the appropriate folder/csv
    # change the output


    # isolate all frames for one video, then do num entries as num frames
    # get total length, then divide by number of frames for number of videos


    for metric in metrics:
        metric_filtered = dataframe[metric]


        print("Creating Histograms")
        fig = plt.hist(metric_filtered, bins=20, log=True)
        plt.xlabel(f"{metric} between GT and actual activation (log scale)")
        plt.ylabel("frequency")
        plt.title(f"Experiment {experiment}: Distribution of {metric} values")
        file_path = os.path.join(output_folder, f"exp_{experiment}_{architecture}_{gc_variant}_{softmax}_{metric}_histogram.png")
        plt.savefig(file_path)
        plt.close()

        print("Creating Box and Whiskers")
        fig = plt.boxplot(metric_filtered, vert=False)
        plt.xlabel(f"{metric} between GT and actual activation (log scale)")
        plt.title(f"Experiment {experiment}: Distribution of {metric} values")
        file_path = os.path.join(output_folder, f"exp_{experiment}_{architecture}_{gc_variant}_{softmax}_{metric}_boxplot.png")
        plt.savefig(file_path)
        plt.close()


    # for exp in experiments:
    #     print(f"experiment: {exp}")
    #     exp_filtered = dataframe[dataframe["experiment"] == exp]

    #     print("Creating Frame Graph")

    #     for metric in metrics:
    #         print(f"metric: {metric}")

    #         ## try plt.figure
    #         metric_filtered = exp_filtered[metric]

    #         print("Creating Histograms")
    #         fig = plt.hist(metric_filtered, bins=20, log=True)
    #         plt.xlabel(f"{metric} between GT and actual activation (log scale)")
    #         plt.ylabel("frequency")
    #         plt.title(f"Experiment {exp}: Distribution of {metric} values")
    #         file_path = os.path.join(output_folder, f"exp_{exp}_{metric}_histogram.png")
    #         plt.savefig(file_path)
    #         plt.close()

    #         print("Creating Box and Whiskers")
    #         fig = plt.boxplot(metric_filtered, vert=False)
    #         plt.xlabel(f"{metric} between GT and actual activation (log scale)")
    #         plt.title(f"Experiment {exp}: Distribution of {metric} values")
    #         file_path = os.path.join(output_folder, f"exp_{exp}_{metric}_boxplot.png")
    #         plt.savefig(file_path)
    #         plt.close()


def plot_single_vid_energy(
    dataframe,
    figure,
):
    




def frames_plot(
    dataframe,
    experiments=[1, 2, 3, 4, 5, "5b"],
    architectures=["slowfast", "i3d", "i3d_nln"],
    metrics=["kl_div", "iou", "pearson", "mse", "covariance"],
    gc_variants=["grad_cam", "grad_cam_plusplus", "eigen_cam"],
):


    # make a single plot for each experiment

    for metric in metrics:
        print(f"metric: {metric}")

        metric_filtered = dataframe[metric]

        print("Creating Histograms")
        fig = plt.hist(metric_filtered, bins=20, log=True)
        plt.xlabel(f"{metric} between GT and actual activation (log scale)")
        plt.ylabel("frequency")
        plt.title(f"Experiment 1: Distribution of {metric} values")
        file_path = os.path.join(output_folder, f"frames_{metric}_histogram.png")
        plt.savefig(file_path)
        plt.close()


def main():
    # csv_path = "/research/cwloka/data/action_attn/synthetic_motion_experiments/metric_results/del_later_full_results.csv"
    # df = pd.read_csv(csv_path)
    # print(df.columns)
    # # filter_by_exp(df)

    ## iterate through the experimental conditions, call framewise
    ## and full metrics
    experiments=[1, 2, 3, 4, 5, "5b"]
    # architectures=["slowfast", "i3d", "i3d_nln"]
    architectures=["i3d", "i3d_nln"]
    gc_variants=["grad_cam", "grad_cam_plusplus", "eigen_cam"]
    softmax_status = ["pre_softmax", "post_softmax"]
    metrics=["kl_div", "iou", "pearson", "mse", "covariance"]
    use_frames = True
    use_single_plots = True

    
    output_folder = "/research/cwloka/projects/nikki_sandbox/action_attention/metric_exploration/plots"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    base_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments/metric_results"
    for exp in experiments:
        for arch in architectures:
            for vis_technique in gc_variants:
                for softmax in softmax_status:
                    data_folder_path = os.path.join(
                        base_dir,
                        f"experiment_{exp}",
                        arch,
                        vis_technique
                    )
                    if use_frames:
                        frames_csv_path = os.path.join(data_folder_path, f"exp_{exp}_{arch}_{softmax}_frames.csv")
                        df = pd.read_csv(frames_csv_path)
                        # frames_plot()
                        print("make sure to call framewise plots once fixed")
                    if use_single_plots:
                        csv_path = os.path.join(data_folder_path, f"exp_{exp}_{arch}_{softmax}.csv")
                        df = pd.read_csv(csv_path)
                        single_plot(df, exp, arch, vis_technique, softmax, output_folder)

                        # dataframe,
                        # experiment,
                        # architecture,
                        # gc_variant,
                        # softmax,
                        # output_folder,

                        # (df, 1, "i3d", "eigen_cam", "post_softmax", output_folder)
                    # call framewise
                    # call exp-wide


    # frames_path = "/research/cwloka/data/action_attn/synthetic_motion_experiments/metric_results/experiment_1/i3d/eigen_cam/exp_1_i3d_post_softmax_frames.csv"
    # df = pd.read_csv(frames_path)
    # print(df.columns)
    # # frames_plot(df)

    # single_plot(df, 1, "i3d", "eigen_cam", "post_softmax", output_folder)


if __name__ == "__main__":
    main()
