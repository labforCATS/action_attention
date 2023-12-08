# Eigencam pre softmax I3d, full motion sequences baseline
CSV_PATH = "/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_1/i3d_output/metric_results.csv"

# load csv contents into pandas dataframe
def load_csv(csv_path=CSV_PATH):
    """ """


# inspect dataframe output to validate correctness
# figure out what kinds of analysis we want to do
#   - subdivide by experiments, do analysis on experiment level
#       - find doc on what the different experiments are varying
#       - analyze numbers themselves
#           - pearson correlation
#           - MSE
#           - KL divergence
#           - covariance
#           - IoU
#       - comparison between correct and wrong predictions
#       - comparison for distinct classes (7 in experiment 1, shapes -- circle, line, quadrilateral, sinusoid, spiral, triangle, zigzag)
#   - subdivide by visualization (eg eigencam), do analysis there
#   - group static experiments / dynamic experiments to do analysis in those groups
