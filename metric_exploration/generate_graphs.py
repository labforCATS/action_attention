import os
import pdb
import pandas as pd
import matplotlib.pyplot as plt



def filter_by_exp(dataframe, experiments = [1, 2, 3, 4, 5, "5b"],
                  architectures = ["slowfast", "i3d", "i3d_nln"],
                  metrics = ["kl_div", "iou", "pearson", "mse", "covariance"],
                  gc_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]):

    # make a single plot for each experiment
    output_folder = "/research/cwloka/projects/nikki_sandbox/action_attention/metric_exploration/plots"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for exp in experiments:
        print(f"experiment: {exp}")
        exp_filtered = dataframe[dataframe['experiment'] == exp]
        for metric in metrics:
            print(f"metric: {metric}")

            ## try plt.figure
            metric_filtered = exp_filtered[metric]
            
            fig = plt.hist(exp_filtered[metric], bins=20)

            plt.xlabel(f"{metric} between GT and actual activation")
            plt.ylabel("frequency")
            plt.title(f"Experiment {exp}: Distribution of {metric} values")

            
            file_path = os.path.join(output_folder, f"exp_{exp}_{metric}_histogram.png")
            plt.savefig(file_path)
            plt.close()


def main():
    csv_path = "/research/cwloka/data/action_attn/synthetic_motion_experiments/metric_results/del_later_full_results.csv"
    df = pd.read_csv(csv_path)
    filter_by_exp(df)


if __name__ == '__main__':
    main()
