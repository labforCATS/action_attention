"""
Creates all frame-vs-metric and frame-vs-activation plots from metric results CSVs.
Plots are created with only data from videos/frames that were correctly classified.
Plots should generate within a couple minutes. 
"""

import os
import pdb
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings
from PIL import Image
from slowfast.visualization.connected_components_utils import load_heatmaps

### global variables ###
experiments = [1, 2, 3, 4, 5, "5b"]
architectures = ["slowfast", "i3d", "i3d_nln"]
gc_variants = ["grad_cam", "grad_cam_plusplus", "eigen_cam"]
softmax_status = ["pre_softmax", "post_softmax"]
metrics = ["kl_div", "iou", "pearson", "mse", "covariance", "precision", "recall"]
# exp_comparisons = [[1, 4], [1, 3, 4], [1, 2], [4, 5], [4, "5b"]]
exp_comparisons = [[1, 4]]

#### FOR REGULAR METRICS #####
# base_data_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
# output_base_folder = "/research/cwloka/data/action_attn/alex_synthetic"
# results_dir = "/research/cwloka/data/action_attn/diane_synthetic/metric_results"
##############################

#### FOR BOUNDING BOXES #####
# base_data_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
# output_base_folder = "/research/cwloka/data/action_attn/bbox_plots"
# results_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments_bbox/metric_results"
##############################

#### FOR ROTATED BOUNDING BOXES #####
base_data_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments"
output_base_folder = "/research/cwloka/data/action_attn/rotated_bbox_plots"
results_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments_rot_bbox/metric_results"
##############################


experiment_stimulus_name_dict = {
    1 : "Motion",
    2 : "Discr. Motion",
    3 : "Bijection",
    4 : "Appearance", 
    5 : "Static Targets",
    "5b" : "Solo Targets"
}
model_cam_name_dict = {
    "eigen_cam" : "EigenCAM",
    "grad_cam" : "GradCAM",
    "grad_cam_plusplus" : "GradCAM++",
    "cam_plusplus" : "GradCAM++", 
    "i3d_rgb": "I3D",
    "i3dFalsergb": "I3D",
    "i3d_nln" : "NLN",
    "i3dTruergb": "NLN",
    "slowfast_fast" : "SlowFast Fast",
    "slowfastFalsefast": "SlowFast Fast",
    "slowfast_slow" : "SlowFast Slow",
    "slowfastFalseslow": "SlowFast Slow",
}

multi_model_color_dict = {
    "i3dFalsergb" : "#AC5EF0", # I3d is purple
    "i3dTruergb" : "#BD075F", # NLN is pink
    "slowfastFalsefast" : "#FBB705", # Fast is yellow
    "slowfastFalseslow" : "#02A1A5",  # Slow is teal
}

vivid_experiment_color_dict = {
    1 : "#0072B2", # dark blue
    2 : "#12C34F", #green
    3 : "#F3BB0B", # yellow
    4 : "#E62B62",  # fuschia
    5 : "#94027A", # dark purple
    "5b" : "#56C8E9" # light blue
}
pastel_experiment_color_dict = {
    # RBGA hex values use alpha = 0.3
    1 : ("#62B6E44d"), # dark blue
    2 : ("#85E8984d"), # green
    3 : ("#F7F1A74d"), # yellow
    4 : ("#FFBBCC4d"), # pink
    5 : ("#C5C1F34d"), # light purple
    "5b" : ("#94D8EC4d")# light blue
}

label_color_dict = {
    "circle": ("#82e3e8"),
    "line": ("#74bcdb"),
    "quadrilateral": ("#d1f0e9"),
    "sinusoid": ("#d1eef0"),
    "spiral": ("#d1d8f0"),
    "triangle": ("#6dc2ae"),
    "zigzag": ("#8faeeb"),
    "Cat": ("#f7c1c1"),
    "Cattle": ("#fcc0b1"),
    "Fish": ("#fccfac"),
    "Flower": ("#fcacc7"),
    "Motorcycle": ("#f0afd7"),
    "Train": ("#f59fe8"),
    "Truck": ("#dbabc6")
}

label_marker_dict = {
    "circle": "x",
    "line": "o",
    "quadrilateral": "v",
    "sinusoid": "^",
    "spiral": "s",
    "triangle": "D",
    "zigzag": "h",
    "Cat": "x",
    "Cattle": "o",
    "Fish": "v",
    "Flower": "^",
    "Motorcycle": "s",
    "Train": "D",
    "Truck": "h"
}

# dictionary of absolute maximum of metric across all CAM and models
cleaned_metric_dict = {'kl_div_1_pre_softmax': 7.035656107244919,
 'iou_1_pre_softmax': 0.2293559305863784,
 'pearson_1_pre_softmax': 5.984556658297634e-06,
 'mse_1_pre_softmax': 0.18324503864892436,
 'covariance_1_pre_softmax': 0.0033476120351479564,
 'precision_1_pre_softmax': 0.3690542145404286,
 'recall_1_pre_softmax': 1.0,
 'kl_div_1_post_softmax': 3.19724894127214,
 'iou_1_post_softmax': 0.21140675665905245,
 'pearson_1_post_softmax': 0.0003918476884032921,
 'mse_1_post_softmax': 0.2637779866463797,
 'covariance_1_post_softmax': 0.003234556411187054,
 'precision_1_post_softmax': 0.27689039364011797,
 'recall_1_post_softmax': 1.0,
 'kl_div_2_pre_softmax': 9.439739646501016,
 'iou_2_pre_softmax': 0.17204435921228484,
 'pearson_2_pre_softmax': 5.385998705043918e-06,
 'mse_2_pre_softmax': 0.18701039318086715,
 'covariance_2_pre_softmax': 0.002201245960990435,
 'precision_2_pre_softmax': 0.227263564922881,
 'recall_2_pre_softmax': 0.9407197111591371,
 'kl_div_2_post_softmax': 3.8478945116191814,
 'iou_2_post_softmax': 0.1654554337928647,
 'pearson_2_post_softmax': 0.00010977447604246125,
 'mse_2_post_softmax': 0.15545827663987466,
 'covariance_2_post_softmax': 0.002152407257149793,
 'precision_2_post_softmax': 0.21914447269144485,
 'recall_2_post_softmax': 0.9549331022659512,
 'kl_div_3_pre_softmax': 4.263908222865258,
 'iou_3_pre_softmax': 0.22656338926880468,
 'pearson_3_pre_softmax': 6.661749446216575e-06,
 'mse_3_pre_softmax': 0.060959850648502706,
 'covariance_3_pre_softmax': 0.003050043592065885,
 'precision_3_pre_softmax': 0.3002356669974934,
 'recall_3_pre_softmax': 0.9725274258661166,
 'kl_div_3_post_softmax': 6.529284261858769,
 'iou_3_post_softmax': 0.19104072643506026,
 'pearson_3_post_softmax': 0.00018938765417563197,
 'mse_3_post_softmax': 0.04553083188033143,
 'covariance_3_post_softmax': 0.0024204905577918227,
 'precision_3_post_softmax': 0.25407301969938384,
 'recall_3_post_softmax': 1.0,
 'kl_div_4_pre_softmax': 5.41798514850815,
 'iou_4_pre_softmax': 0.2625434064451517,
 'pearson_4_pre_softmax': 7.470192326193211e-06,
 'mse_4_pre_softmax': 0.05950425646423308,
 'covariance_4_pre_softmax': 0.0020817616541472252,
 'precision_4_pre_softmax': 0.38717367385190476,
 'recall_4_pre_softmax': 0.6642431163984627,
 'kl_div_4_post_softmax': 3.5388993806392066,
 'iou_4_post_softmax': 0.20520017933969648,
 'pearson_4_post_softmax': 0.00014381403372225137,
 'mse_4_post_softmax': 0.0322179353300184,
 'covariance_4_post_softmax': 0.0016767388149155043,
 'precision_4_post_softmax': 0.3514779124595773,
 'recall_4_post_softmax': 0.9072433192686357,
 'kl_div_5_pre_softmax': 10.196139007729755,
 'iou_5_pre_softmax': 0.220054695636876,
 'pearson_5_pre_softmax': 7.762328266667548e-06,
 'mse_5_pre_softmax': 0.042068160138253416,
 'covariance_5_pre_softmax': 0.0030577964754668666,
 'precision_5_pre_softmax': 0.2786835693423155,
 'recall_5_pre_softmax': 0.9728091915155823,
 'kl_div_5_post_softmax': 9.559964788545631,
 'iou_5_post_softmax': 0.22396769234419414,
 'pearson_5_post_softmax': 4.737965902534215e-05,
 'mse_5_post_softmax': 0.03990665877796796,
 'covariance_5_post_softmax': 0.0030663945363231175,
 'precision_5_post_softmax': 0.275090288273044,
 'recall_5_post_softmax': 0.9606579162614013,
 'kl_div_5b_pre_softmax': 7.185282550931246,
 'iou_5b_pre_softmax': 0.25201544295468986,
 'pearson_5b_pre_softmax': 9.646821945878151e-06,
 'mse_5b_pre_softmax': 0.11928374824358842,
 'covariance_5b_pre_softmax': 0.0036542927404916116,
 'precision_5b_pre_softmax': 0.43766576997495726,
 'recall_5b_pre_softmax': 0.9959794914049029,
 'kl_div_5b_post_softmax': 7.185282550931246,
 'iou_5b_post_softmax': 0.24813590082732048,
 'pearson_5b_post_softmax': 0.00012971551751824088,
 'mse_5b_post_softmax': 0.11928374824358842,
 'covariance_5b_post_softmax': 0.003607487639106988,
 'precision_5b_post_softmax': 0.4354284754785913,
 'recall_5b_post_softmax': 0.9946167559129488,
 'activation_1_pre_softmax': 89.94815963745117,
 'activation_1_post_softmax': 16.170370483398436,
 'activation_2_pre_softmax': 81.79341462787829,
 'activation_2_post_softmax': 20.72054732473273,
 'activation_3_pre_softmax': 52.77462348159479,
 'activation_3_post_softmax': 21.61646340110085,
 'activation_4_pre_softmax': 28.10983488831339,
 'activation_4_post_softmax': 3.3719471364781475,
 'activation_5_pre_softmax': 28.28434909119898,
 'activation_5_post_softmax': 26.347219038982782,
 'activation_5b_pre_softmax': 44.96758355034722,
 'activation_5b_post_softmax': 44.96758355034722}

class_id_marker_dict = {
    0 : "x",
    1 : "o",
    2 : "v",
    3 : "^",
    4 : "<", 
    5 : ">",
    6 : "s",
    7 : "D",
    8 : "h"
}

def merge_metric_and_activation_csv(arch, vis_technique, softmax, channel, exp):
    # get framewise metric dataframe
    # for target-mask (no-bounding-boxes)
    # frame_metric_csv = os.path.join(
    #     results_dir, f"experiment_{exp}", arch, vis_technique, f"exp_{exp}_{arch}_{softmax}_frames.csv"
    # )

    # for regular bounding boxes
    # frame_metric_csv = os.path.join(
    #     results_dir, f"experiment_{exp}", arch, vis_technique, f"exp_{exp}_bbox_{arch}_{softmax}_frames.csv"
    # )

    # for rotated bounding boxes
    frame_metric_csv = os.path.join(
        results_dir, f"experiment_{exp}", arch, vis_technique, f"rot_bbox", f"exp_{exp}_bbox_{arch}_{softmax}_frames.csv"
    )

    df = pd.read_csv(frame_metric_csv)
    df = df.loc[df["channel"] == channel]
    df.drop_duplicates(inplace = True)

    # get framewise activation dataframe
    framewise_root_dir = os.path.join(
        base_data_dir, f"experiment_{exp}", f"{arch}_output",
    )

    heatmap_folder = ""
    for entry in os.listdir(framewise_root_dir):
        if "heatmaps_epoch_" in entry:
            heatmap_folder = entry

    framewise_csv_path = os.path.join(
        framewise_root_dir, heatmap_folder, vis_technique, softmax, f"{channel}_framewise_activations.csv"
    )
    
    framewisedf = pd.read_csv(framewise_csv_path)

    # merge the two
    try:
        df["mean_activations"] = framewisedf["mean_activations"].values
        pd.testing.assert_series_equal(df["input_vid_idx"], framewisedf["input_vid_idx"], check_index=False)
        pd.testing.assert_series_equal((df["frame_id"] + 1), framewisedf["frame_id"], check_index=False) 
        # frames are 1-indexed in metrics CSV, 0-indexed in framewise activations CSV
    except:
        print('error occurred with metric df', str(frame_metric_csv), "and framewise df", str(framewise_csv_path ))
        pdb.set_trace()

    if channel == "slow":
        df["frame_id"] *= 4 # slow channel has 1/4 the frame rate of all other channels
    
    df = df.loc[df["correct"] == True] # only use data from correctly-classified videos

    return df

def find_metric_maximums():
    # only had to run once to get dictionary above
    metric_dict = {}
    for exp in experiments:
        for arch in architectures:
            for vis_technique in gc_variants:
                for softmax in softmax_status:
                    if arch == "slowfast":
                            channels = ["slow", "fast"]
                    elif arch in ["i3d", "i3d_nln"]:
                            channels = ["rgb"]
                    else:
                            raise NotImplementedError("Add in logic for handling channels")
                    for channel in channels:
                        df = merge_metric_and_activation_csv(arch, vis_technique, softmax, channel, exp)

                        for metric in metrics:

                            metric_df = df[["input_vid_idx", "label","label_numeric", "frame_id", metric]].copy()
                            maxlist = []

                            for class_id in metric_df["label_numeric"].unique():
                                plot_df = metric_df[metric_df["label_numeric"] == class_id]
                                mean_df = plot_df.groupby(["frame_id"]).mean(numeric_only = True)

                                maxlist.append(mean_df[metric].max())

                            key = f"{metric}_{exp}_{softmax}"
                            metric_dict.setdefault(key, [])
                            metric_dict[f"{metric}_{exp}_{softmax}"].append(max(maxlist))

                        activation_df = df[["input_vid_idx", "label","label_numeric", "frame_id", "mean_activations"]].copy()
                        maxlist = []

                        for class_id in activation_df["label_numeric"].unique():
                            plot_df = activation_df[metric_df["label_numeric"] == class_id]
                            mean_df = plot_df.groupby(["frame_id"]).mean(numeric_only = True)
                            maxlist.append(mean_df["mean_activations"].max())

                        key = f"activation_{exp}_{softmax}"
                        metric_dict.setdefault(key, [])
                        metric_dict[f"activation_{exp}_{softmax}"].append(max(maxlist))

    for exp in experiments:
        for key, value in metric_dict.items():
                cleaned_metric_dict[key] = max(value)

######################################################################################################
# FUNCTIONS #
######################################################################################################


def rescaled_multi_experiment_frames_vs_metric_plots(
    experiment_subset_list,
    vis_technique,
    softmax,
):
    ########################## shared lines for all multi experiment plots ############################
    warnings.filterwarnings("ignore") # avoid spam of warnings that these lines are not on legend

    for arch in architectures:
        if arch == "slowfast":
                channels = ["slow", "fast"]
        elif arch in ["i3d", "i3d_nln"]:
                channels = ["rgb"]
        else:
                raise NotImplementedError("Add in logic for handling channels")

        for channel in channels:
            output_folder = os.path.join(
                                output_base_folder,
                                f"multi_experiment_{experiment_subset_list}", arch, vis_technique, softmax, channel
                            )

            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

            arch_model_grid_folder = os.path.join(
                                output_base_folder,
                                f"multi_experiment_{experiment_subset_list}", "arch_model_grid", softmax)

            if not os.path.exists(arch_model_grid_folder):
                                os.makedirs(arch_model_grid_folder)

            dataframe_list = []

            for i in range(len(experiment_subset_list)):
                exp = experiment_subset_list[i]
                df = merge_metric_and_activation_csv(arch, vis_technique, softmax, channel, exp)
                dataframe_list.append(df)
            
    ########################## shared lines for all multi experiment plots ############################

            for metric in metrics:
                fig, ax = plt.subplots()

                pivot_list = []

                for i in range(len(dataframe_list)):


                    metric_df = dataframe_list[i][["input_vid_idx", "label","label_numeric", "frame_id", metric]].copy()
                    # metric_df["unique_id"] = metric_df["input_vid_idx"].astype(str) + metric_df["label_numeric"].astype(str)

                    for class_id in metric_df["label_numeric"].unique():
                        plot_df = metric_df[metric_df["label_numeric"] == class_id]
                        mean_df = plot_df.groupby(["frame_id"]).mean()

                        classlabel = plot_df["label"].iloc[0]

                        mean_df.plot(ax=ax, y=metric, color=pastel_experiment_color_dict[experiment_subset_list[i]], label="_nolegend_")

                        # import pdb; pdb.set_trace()

                        # unique color for each class
                        # mean_df.plot(ax=ax, y=metric, color=label_color_dict[classlabel], label=classlabel)

                        # unique marker for each class
                        mean_df['frameindex'] = mean_df.index
                        # marker at start of line
                        ax.scatter(x=mean_df['frameindex'][::8], y=mean_df[metric][::8], color=vivid_experiment_color_dict[experiment_subset_list[i]], marker = label_marker_dict[classlabel], label=classlabel)
                    
                    mean_df = metric_df.groupby(["frame_id"]).mean()
                    stimuluslabel = experiment_stimulus_name_dict[experiment_subset_list[i]]
                    mean_df.plot(ax=ax, y=metric, color=vivid_experiment_color_dict[experiment_subset_list[i]], label=f"{stimuluslabel} mean")
                    
                ax.legend(bbox_to_anchor=(1.1, 1.05))

                ax.set_xlabel("frame id")
                ax.set_ylabel({metric})

                file_path = os.path.join(output_folder, f"rescaled_multi_exp_frames_vs_{metric}_.png")
                plt.savefig(file_path, bbox_inches='tight')

                y_limits = []
                for i in experiment_subset_list:
                    config = metric + "_" + str(i) + "_" + softmax
                    y_limits.append(cleaned_metric_dict[config])
                ax.set_ylim([0, max(y_limits)])

                ax.get_legend().remove() # no need for legend on grid plots

                arch_cam_grid_path = os.path.join(arch_model_grid_folder, f"rescaled_frames_vs_{metric}_{arch}_{channel}_{vis_technique}.png")
                plt.savefig(arch_cam_grid_path)

                plt.cla()
                plt.clf()
            
            print("plotted for", arch, channel)

    plt.close()

def rescaled_multi_experiment_frames_vs_activation_plots(
    experiment_subset_list,
    vis_technique,
    softmax,
):
    ########################## shared lines for all multi experiment plots ############################
    warnings.filterwarnings("ignore") # avoid spam of warnings that these lines are not on legend

    for arch in architectures:
        if arch == "slowfast":
                channels = ["slow", "fast"]
        elif arch in ["i3d", "i3d_nln"]:
                channels = ["rgb"]
        else:
                raise NotImplementedError("Add in logic for handling channels")

        for channel in channels:
            output_folder = os.path.join(
                                output_base_folder,
                                f"multi_experiment_{experiment_subset_list}", arch, vis_technique, softmax, channel
                            )

            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

            arch_model_grid_folder = os.path.join(
                                output_base_folder,
                                f"multi_experiment_{experiment_subset_list}", "arch_model_grid", softmax)

            if not os.path.exists(arch_model_grid_folder):
                                os.makedirs(arch_model_grid_folder)

            dataframe_list = []

            for i in range(len(experiment_subset_list)):
                exp = experiment_subset_list[i]
                df = merge_metric_and_activation_csv(arch, vis_technique, softmax, channel, exp)
                dataframe_list.append(df)

            ########################## shared lines for all multi experiment plots ############################

            fig, ax = plt.subplots()

            for i in range(len(dataframe_list)):
                metric_df = dataframe_list[i][["input_vid_idx", "label","label_numeric", "frame_id", "mean_activations"]].copy()


                for class_id in metric_df["label_numeric"].unique():

                    plot_df = metric_df[metric_df["label_numeric"] == class_id]
                    mean_df = plot_df.groupby(["frame_id"]).mean()
                    classlabel = plot_df["label"].iloc[0]
                    mean_df.plot(ax=ax, y="mean_activations", color=pastel_experiment_color_dict[experiment_subset_list[i]], label=classlabel)

                mean_df = metric_df.groupby(["frame_id"]).mean()
                stimuluslabel = experiment_stimulus_name_dict[experiment_subset_list[i]]
                mean_df.plot(ax=ax, y="mean_activations", color=vivid_experiment_color_dict[experiment_subset_list[i]], label =f"{stimuluslabel} mean")
                
            ax.legend(bbox_to_anchor=(1.1, 1.05))

            ax.set_xlabel("frame id")
            ax.set_ylabel("mean_activations")

            file_path = os.path.join(output_folder, f"rescaled_multi_exp_frames_vs_activation_.png")
            plt.savefig(file_path, bbox_inches='tight')

            y_limits = []
            for i in experiment_subset_list:
                config = "activation" + "_" + str(i) + "_" + softmax
                y_limits.append(cleaned_metric_dict[config])
            ax.set_ylim([0, max(y_limits)])

            ax.get_legend().remove() # no need for legend on grid plots

            arch_cam_grid_path = os.path.join(arch_model_grid_folder, f"rescaled_frames_vs_activation_{arch}_{channel}_{vis_technique}.png")
            plt.savefig(arch_cam_grid_path)
            plt.cla()
            plt.clf()
        
            print("plotted for", arch, channel)
    plt.close()

def rescaled_arch_model_grid_metric(
    experiment_subset_list = [1, 4], metric = "iou", softmax = "pre_softmax"
    ):

    flattened_image_dir = os.path.join(output_base_folder, f"multi_experiment_{experiment_subset_list}", "arch_model_grid", softmax)

    image_name_list = []
    for __, __, files in os.walk(flattened_image_dir):
        for f in files:
            if f[19:(19+len(metric))] == metric:
                image_name_list.append(f)
    
    assert len(image_name_list) == 12 # 4 arch/channel by 3 cams 
    image_name_list.sort() 
    # places in order of nln eigen, nln gc, nln ++, i3d eigen, i3d gc, i3d ++,
    #                    fast eigen, fast gc, fast ++, slow eigen, slow gc, slow ++

    assert image_name_list[0][-21:] == "nln_rgb_eigen_cam.png" # check sorting is correct
    assert image_name_list[4][-20:] == "i3d_rgb_grad_cam.png" # check sorting is correct
    assert image_name_list[-1][-26:] == "slow_grad_cam_plusplus.png" # check sorting is correct

    fig,ax = plt.subplots(3,4)

    plt.subplots_adjust(left=0.1,
                    bottom=0.2, 
                    right=0.9, 
                    top=0.8, 
                    wspace=0.05, 
                    hspace=0.0001)

    for j in range(4): # columns of models
        for i in range(3): # rows of CAMs
            name = image_name_list[(3*j)+i]
            with open(flattened_image_dir + "/" + name, "rb") as f:
                image = Image.open(f)
                ax[i][j].imshow(image)

                if metric == "kl_div": # kl_div has a _ in the name, which disrupts later string handling
                    name = "_".join(name.split("_")[:2]) + "_kldiv_" + "_".join(name.split("_")[4:])

                name = name[10:] # remove rescaled_ from beginning
                model_name = name.split("_")[3:5] # i3d_nln, i3d_rgb, or slowfast_channel
                model_name = "_".join(model_name)
                model_name = model_cam_name_dict[model_name]
                cam_name = name.split("_")[-2:] # eigen_cam, grad_cam, or cam_plusplus
                cam_name = "_".join(cam_name)
                cam_name = cam_name[:-4] # remove png
                cam_name = model_cam_name_dict[cam_name]
                
                ax[i][j].set_xlabel(model_name)
                ax[i][j].set_ylabel(cam_name)

    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
        a.label_outer()

    stimulus_set_names = [experiment_stimulus_name_dict[x] for x in experiment_subset_list]

    # titles not wanted for paper, but useful for internal files
    # fig.suptitle(f"{metric} for {stimulus_set_names} ({softmax})", x=0.5, y=0.85, fontsize=10)

    file_path = os.path.join(flattened_image_dir, f"rescaled_GRID_frames_vs_{metric}_{softmax}.png")
    plt.savefig(file_path, dpi=500)
    plt.close()

def rescaled_arch_model_grid_activation(
    experiment_subset_list = [1, 4], softmax = "pre_softmax"
    ):

    flattened_image_dir = os.path.join(output_base_folder, f"multi_experiment_{experiment_subset_list}", "arch_model_grid", softmax)

    image_name_list = []
    for __, __, files in os.walk(flattened_image_dir):
        for f in files:
            if f[19:(19+len("activation"))] == "activation": #frames_vs_{yvar}
                image_name_list.append(f)
    
    assert len(image_name_list) == 12 # 4 arch/channel by 3 cams 
    image_name_list.sort() 
    # places in order of nln eigen, nln gc, nln ++, i3d eigen, i3d gc, i3d ++,
    #                    fast eigen, fast gc, fast ++, slow eigen, slow gc, slow ++

    assert image_name_list[0][-21:] == "nln_rgb_eigen_cam.png" # check sorting is correct
    assert image_name_list[4][-20:] == "i3d_rgb_grad_cam.png" # check sorting is correct
    assert image_name_list[-1][-26:] == "slow_grad_cam_plusplus.png" # check sorting is correct

    fig,ax = plt.subplots(3,4)

    plt.subplots_adjust(left=0.1,
                    bottom=0.2, 
                    right=0.9, 
                    top=0.8, 
                    wspace=0.05, 
                    hspace=0.0001)

    for j in range(4): # columns of models
        for i in range(3): # rows of CAMs
            name = image_name_list[(3*j)+i]
            with open(flattened_image_dir + "/" + name, "rb") as f:
                image = Image.open(f)
                ax[i][j].imshow(image)

                name = name[10:]
                model_name = name.split("_")[3:5] # i3d_nln, i3d_rgb, or slowfast_channel
                model_name = "_".join(model_name)
                model_name = model_cam_name_dict[model_name]
                cam_name = name.split("_")[-2:] # eigen_cam, grad_cam, or cam_plusplus
                cam_name = "_".join(cam_name)
                cam_name = cam_name[:-4] # remove png
                cam_name = model_cam_name_dict[cam_name]
                
                ax[i][j].set_xlabel(model_name)
                ax[i][j].set_ylabel(cam_name)

    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
        a.label_outer()

    stimulus_set_names = [experiment_stimulus_name_dict[x] for x in experiment_subset_list]

    # titles not wanted for paper, but useful for internal files
    # fig.suptitle(f"activation for stimulus sets {stimulus_set_names} ({softmax})", x=0.5, y=0.85, fontsize=12)

    file_path = os.path.join(flattened_image_dir, f"rescaled_GRID_frames_vs_activation_{softmax}.png")
    plt.savefig(file_path, dpi=500)
    plt.close()

######################################################################################################
# "generate all" functions to generate one kind of plot for all applicable configurations
######################################################################################################

def gen_all_rescaled_multi_experiment_plots():
    for subset in exp_comparisons:
        for vis_technique in gc_variants:
            for softmax in softmax_status:
                print("starting for", subset, vis_technique, softmax)
                rescaled_multi_experiment_frames_vs_metric_plots(
                    subset,
                    vis_technique,
                    softmax,
                )
                
                rescaled_multi_experiment_frames_vs_activation_plots(
                    subset,
                    vis_technique,
                    softmax,
                )
                print("finished plots for ", subset, vis_technique, softmax)


def gen_all_rescaled_grid_frames_vs_metric_plots():
    for subset in exp_comparisons:
        for metric in metrics:
            for softmax in softmax_status:
                
                rescaled_arch_model_grid_metric(experiment_subset_list = subset, metric=metric, softmax=softmax)

                print("grid metric plots for ", subset, metric, softmax)

def gen_all_rescaled_grid_frames_vs_activation_plots():
    for subset in exp_comparisons:
        for softmax in softmax_status:
        
            rescaled_arch_model_grid_activation(experiment_subset_list = subset, softmax=softmax)

            print("grid activation plots for ", subset, softmax)
