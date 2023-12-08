import pandas as pd

# Eigencam post softmax I3d, full motion sequences baseline
CSV_PATH = "/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_1/i3d_output/metric_results.csv"

def evaluate_experiments(df: pd.DataFrame, experiment_nums: list[int] = None) -> pd.DataFrame:
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
    if experiment_nums is None:
        experiments_df = df
    else:
        experiments_df = df.loc[df['experiment'].isin(experiment_nums)]

    # get results on those rows
    results_df = evaluate(experiments_df) # uses default of all metrics

    return results_df

def evaluate_labels(df: pd.DataFrame, labels: optional[list[str]] = None) -> pd.DataFrame:
    """TODO

    Args:
        df (pd.DataFrame): TODO
        labels (optional[list[str]], optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: TODO
    """
    # access relevant experiment rows
    if labels is None:
        experiments_df = df
    else:
        experiments_df = df.loc[df['label'].isin(labels)]

    # TODO

def evaluate(df: pd.DataFrame, metrics: list[str] = ["kl_div", "mse", "covariance", "pearson", "iou"]) -> pd.DataFrame:
    """Finds mean and standard deviation for selected metrics of input dataframe

    Args:
        df (pd.DataFrame): Dataframe containing experimental results to evaluate
        metrics (list[str], optional): Metrics to run statistics on. Defaults to ["kl_div", "mse", "covariance", "pearson", "iou"], all metrics available.

    Returns:
        pd.DataFrame: DataFrame containing the statistical results for the selected metrics of the input dataframe
    """
    means = df[metrics].mean()
    stds = df[metrics].std()

    # create dataframe from results
    results_dict = {'mean': means, 'std': stds}
    results_df = pd.DataFrame(results_dict)
    
    return results_df


#       - find doc on what the different experiments are varying
#       - comparison for distinct classes (7 in experiment 1, shapes -- circle, line, quadrilateral, sinusoid, spiral, triangle, zigzag)
#   - subdivide by visualization (eg eigencam), do analysis there
#   - group static experiments / dynamic experiments to do analysis in those groups

if __name__ == "__main__":
    # load csv contents into pandas dataframe
    input_df = pd.read_csv(CSV_PATH)
    print(input_df.columns)
    results_df = evaluate_experiments(input_df, [1])
    # print(results_df)
