import pandas as pd

# Eigencam post softmax I3d, full motion sequences baseline
CSV_PATH = "/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_1/i3d_output/metric_results.csv"
METRIC_OPTIONS = ["kl_div", "mse", "covariance", "pearson", "iou"]

def evaluate_experiments(df: pd.DataFrame, experiment_nums: list[int]):
    """Runs metrics for a given list of experiments. Can be run for one or multiple
        experiments.

    Args:
        df (pd.DataFrame): DataFrame containing experimental results
        experiment_num (int): Integer indicating the number of the experiment. Note: 5b
                is experiment 6.

    Returns:
        pd.DataFrame: dataframe containing the results from analysis of experiment
                metrics. Calculates mean. 
    """
    # access relevant experiment rows
    experiments_df = df.loc[df['experiment'].isin(experiment_nums)]

    # get results on those rows
    means = experiments_df[METRIC_OPTIONS].mean()
    stds = experiments_df[METRIC_OPTIONS].std()

    # create dataframe from results
    results_dict = {'Mean': means, 'Standard Deviation': stds}
    results_df = pd.DataFrame(results_dict)

    return results_df


#       - find doc on what the different experiments are varying
#       - comparison between correct and wrong predictions
#       - comparison for distinct classes (7 in experiment 1, shapes -- circle, line, quadrilateral, sinusoid, spiral, triangle, zigzag)
#   - subdivide by visualization (eg eigencam), do analysis there
#   - group static experiments / dynamic experiments to do analysis in those groups

if __name__ == "__main__":
    # load csv contents into pandas dataframe
    input_df = pd.read_csv(CSV_PATH)
    results_df = evaluate_experiments(input_df, [1])
    print(results_df)
