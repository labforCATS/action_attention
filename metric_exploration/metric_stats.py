"""
Calculates statistics (ex. mean, quartiles, stdev) for metrics on different configurations
stores all values in csv files separated by experiment number.
"""

import os
import csv
import pdb
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import statistics
from slowfast.visualization.connected_components_utils import load_heatmaps

ORIGIN_DIRECTORY = "/research/cwloka/data/action_attn/synthetic_motion_experiments/metric_results/"
OUTPUT_DIRECTORY = "/research/cwloka/projects/diane_sandbox/synthetic_output/csv_files/"

def create_csv(
    experiment,
    architectures,
    gc_variants,
    softmax_status,
    metrics,
    output_dir = OUTPUT_DIRECTORY,
    exp_dir = ORIGIN_DIRECTORY,
):
    """
    calculates stats across configurations for all metrics
    and writes them into a csv file for the experiment.
    Only handles one experiment (dataset) per function call
    """
    
    # Creates the list of dictionaries
    csv_rows = []

    # Headers for the first row of csv.
    # Note that if dictionary keys don't match, there will be issues
    csv_headers = ["exp", "model", "nonlocal", "gradcam_variant", "post_softmax", 
                    "channel", "correct", "metric", "total_videos", "min", "max", 
                    "range", "mean", "median", "Q1", "Q3", "stdev"]

    
    for architecture in architectures:
        if (architecture == "i3d_nln"):
            non_local = True
        else:
            non_local = False
    
        for gc_variant in gc_variants:
            for softmax in softmax_status:
                
                # Get Path to csv files with metric values
                csv_path = os.path.join(
                    exp_dir, f"experiment_{experiment}", architecture, gc_variant,
                    f"exp_{experiment}_{architecture}_{softmax}.csv"
                )
                df = pd.read_csv(csv_path)

                # We will calculate stats for only correct values and all values
                for correct in [True, False]:
                    config_filtered = df.loc[df["correct"] == correct]

                    # If we are looking at slowfast, loop through both channels
                    if (architecture == "slowfast"):
                        for channel in ["slow", "fast"]:
                            channel_filtered = config_filtered.loc[config_filtered["channel"] == channel]
                            for metric in metrics:
                                metric_filtered = channel_filtered[metric]
                                metric_filtered = metric_filtered.tolist()

                                # filter nan values
                                nan_filtered = [x for x in metric_filtered if x == x]

                                # use config values for first couple columns and calculate stats
                                d = {"exp": experiment, "model": architecture, "nonlocal": non_local, "gradcam_variant": gc_variant,
                                    "post_softmax": softmax, "channel": channel, "correct": correct, "metric": metric}
                                d["total_videos"] = len(nan_filtered)
                                d["min"] = min(nan_filtered)
                                d["max"] = max(nan_filtered)
                                d["range"] = max(nan_filtered) - min(nan_filtered)
                                d["mean"] = statistics.fmean(nan_filtered)
                                d["median"] = statistics.median(nan_filtered)
                                d["Q1"] = np.quantile(nan_filtered, .25)
                                d["Q3"] = np.quantile(nan_filtered, .75)
                                d["stdev"] = statistics.pstdev(nan_filtered)

                                csv_rows += [d]
                                #pdb.set_trace()
                    else:
                        channel = "rgb"

                    for metric in metrics:
                        metric_filtered = config_filtered[metric]
                        metric_filtered = metric_filtered.tolist()

                        nan_filtered = [x for x in metric_filtered if x == x]

                        

                        d = {"exp": experiment, "model": architecture, "nonlocal": non_local, "gradcam_variant": gc_variant,
                             "post_softmax": softmax, "channel": channel, "correct": correct, "metric": metric}
                        d["total_videos"] = len(nan_filtered)
                        d["min"] = min(nan_filtered)
                        d["max"] = max(nan_filtered)
                        d["range"] = max(nan_filtered) - min(nan_filtered)
                        d["mean"] = statistics.fmean(nan_filtered)
                        d["median"] = statistics.median(nan_filtered)
                        d["Q1"] = np.quantile(nan_filtered, .25)
                        d["Q3"] = np.quantile(nan_filtered, .75)
                        d["stdev"] = statistics.pstdev(nan_filtered)

                        csv_rows += [d]
                        #pdb.set_trace()

    # write to csv
    keys = csv_rows[0].keys()
    with open(output_dir + f"experiment_{experiment}.csv", "w") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(csv_rows)


                            # for loop through all metrics
                            # filter for nans
                            # perform all calculations and add dictionary to csv rows
                            

                    
                # channel, correct, metric columns 
    # filter dataframe through for loops
        # calculate stats
        # add a row to dictionary

    # write to csv




def main():
    # Specifies what configurations we want to look at
    # BESIDES experiments, these likely should NOT be changed
    experiments = [1, 2, 3, 4, 5, "5b"]
    architectures = ["slowfast", "i3d", "i3d_nln"]
    gc_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
    softmax_status = ["pre_softmax", "post_softmax"]
    metrics = ["kl_div", "iou", "pearson", "mse", "covariance"]

    for experiment in experiments:
        create_csv(experiment, architectures, gc_variants, softmax_status, metrics)



if __name__ == "__main__":
    main()