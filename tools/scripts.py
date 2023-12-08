from slowfast.utils.parser import load_config

import os
import json
import shutil
import pdb
import traceback


def output_idx_to_input(input_json_path, output_vid_idx):
    """Given the index for an output data, return the target class and index
    of the input video.

    Args:
        input_json_path (str): path to the json file containing input data
            information. for example, "/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_1/synthetic_motion_test.json"
        output_vid_idx (int): index for the output video (assumes 0-based index)
    """
    with open(input_json_path, "r") as f:
        vid_logs = json.load(f)  # list of dictionaries

    input_vid_log = vid_logs[output_vid_idx]
    target_class = input_vid_log["labels"]
    input_vid_idx = int(input_vid_log["video_id"].split("_")[1])
    return target_class, input_vid_idx


def reformat_output_dirs(outputs_dir):
    """Reformats the output files so that they are sorted by class and
    reindexed to match their input video index.

    Args:
        outputs_dir (str): output directory containing files with one of the
            following structures:
                * each input video corresponds to a folder of frames from multiple channels, e.g. "outputs_dir/{output_vid_idx}/{channel}/{output_vid_idx}_{frame_idx}.jpg"
                * each input video corresponds to a set 3d heatmap files for multiple channels, e.g. "outputs_dir/{vid_idx:06d}/{channel}_{vid_idx:06d}.html"
            for example, valid paths include
            "/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_1/slowfast_output/heatmaps/grad_cam/pre_softmax/frames"
            and
            "/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_1/slowfast_output/heatmaps/grad_cam/pre_softmax/3d_volumes"
            "/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_1/slowfast_output/heatmaps/grad_cam/pre_softmax/heatmap_overlay/frames"
            "/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_1/slowfast_output/heatmaps/grad_cam/pre_softmax/heatmap_overlay/videos"
    """
    try:
        temp_split_paths = outputs_dir.split("experiment_")
        experiment_root_dir = (
            temp_split_paths[0] + "experiment_" + temp_split_paths[1][0]
        )
        print(experiment_root_dir)
        input_json_path = os.path.join(
            experiment_root_dir, "synthetic_motion_test.json"
        )

        # get all target classes
        with open(input_json_path, "r") as f:
            vid_logs = json.load(f)  # list of dictionaries
        class_names = set([vid_log["labels"] for vid_log in vid_logs])
        print(class_names)

        last_folder = os.path.basename(outputs_dir)
        second_last_folder = os.path.basename(os.path.dirname(outputs_dir))

        # iterate over all items in the outputs_dir, move into corresponding target class dir, and rename all downstream files so they use the input vid idx
        original_fnames = sorted(
            os.listdir(outputs_dir)
        )  # fix the original file names, because we will be creating new folders in the same directory
        for fname in original_fnames:
            # the folder name will be a 6 digit number
            # make sure the file is valid and not some junk like .DS_Store
            if len(fname) == 6 and fname.isnumeric():
                # get the target class and original index of the input video
                output_vid_idx = int(fname)
                target_class, input_vid_idx = output_idx_to_input(
                    input_json_path, output_vid_idx
                )

                curr_dir = os.path.join(outputs_dir, fname)

                if (
                    second_last_folder != "heatmap_overlay"
                    and last_folder == "3d_volumes"
                ):
                    new_dir = os.path.join(
                        outputs_dir,
                        target_class,
                        f"{target_class}_{input_vid_idx:06d}",
                    )
                    os.makedirs(new_dir, exist_ok=True)

                    for file in sorted(os.listdir(curr_dir)):
                        # make sure these are valid html files. the file name should have the format {channel}_{vid_idx}.html.
                        if file.endswith(".html"):
                            channel = file.split("_")[0]

                            curr_file_path = os.path.join(curr_dir, file)
                            new_file_path = os.path.join(
                                new_dir,
                                f"{target_class}_{input_vid_idx:06d}_{channel}.html",
                            )

                            # copy the file to the new location
                            shutil.copy(curr_file_path, new_file_path)
                            print("new_file_path", new_file_path)

                elif (
                    second_last_folder != "heatmap_overlay"
                    and last_folder == "frames"
                ):
                    for channel in sorted(os.listdir(curr_dir)):
                        new_dir = os.path.join(
                            outputs_dir,
                            target_class,
                            f"{target_class}_{input_vid_idx:06d}",
                            channel,
                        )
                        os.makedirs(new_dir, exist_ok=True)

                        for frame_name in sorted(
                            os.listdir(os.path.join(curr_dir, channel))
                        ):
                            # make sure the file is valid and not some junk like .DS_Store
                            if frame_name.endswith(".jpg"):
                                # frame has the file name {output_vid_idx:06d}_{frame_idx:06d}.jpg

                                frame_idx = int(frame_name[:-4].split("_")[1])

                                curr_frame_path = os.path.join(
                                    curr_dir, channel, frame_name
                                )
                                new_frame_path = os.path.join(
                                    new_dir,
                                    f"{target_class}_{input_vid_idx:06d}_{channel}_{frame_idx:06d}.jpg",
                                )

                                # copy the file to the new location
                                shutil.copy(curr_frame_path, new_frame_path)
                                print("new_frame_path", new_frame_path)

                elif (
                    second_last_folder == "heatmap_overlay"
                    and last_folder == "frames"
                ):
                    raise NotImplementedError
                elif (
                    second_last_folder == "heatmap_overlay"
                    and last_folder == "videos"
                ):
                    raise NotImplementedError
    except Exception as e:
        print(traceback.format_exc())
        pdb.set_trace()
        raise e


def run_reformat_output_dirs(cfg_path):
    # set up cfg
    cfg = load_config(args=None, path_to_config=cfg_path)

    # check all the places where outputs are saved (e.g. heatmap volumes,
    # frames, overlays), and reformat their outputs
    heatmaps_root_dir = os.path.join(
        cfg.OUTPUT_DIR,
        "heatmaps",
        cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.METHOD,
        "post_softmax"
        if cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.POST_SOFTMAX
        else "pre_softmax",
    )

    # heatmap frames are always automatically saved
    print("reformatting heatmap frames")
    heatmaps_frames_root_dir = os.path.join(
        heatmaps_root_dir,
        "frames",
    )
    try:
        reformat_output_dirs(heatmaps_frames_root_dir)
    except Exception as e:
        pdb.set_trace()
        raise e

    # heatmap 3d volumes are always automatically saved
    print("reformatting heatmap 3d volumes")
    heatmap_volume_root_dir = os.path.join(heatmaps_root_dir, "3d_volumes")
    try:
        reformat_output_dirs(heatmap_volume_root_dir)
    except Exception as e:
        pdb.set_trace()
        raise e

    # check if overlay videos and frames are saved
    if cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.SAVE_OVERLAY_VIDEO:
        print("reformatting heatmap overlay videos and frames")
        heatmap_overlay_frames_root_dir = os.path.join(
            heatmaps_root_dir,
            "heatmap_overlay",
            "frames",
        )
        reformat_output_dirs(heatmap_overlay_frames_root_dir)
        heatmap_overlay_video_root_dir = os.path.join(
            heatmaps_root_dir,
            "heatmap_overlay",
            "videos",
        )
        reformat_output_dirs(heatmap_overlay_video_root_dir)


def run_reformat_all_outputs():
    root_dir = "/research/cwloka/data/action_attn/synthetic_motion_experiments/"

    # iterate over all experiments
    for exp in ["1", "2", "3", "4", "5", "5b"]:
        # iterate over all configs
        configs_dir = os.path.join(root_dir, f"experiment_{exp}/configs")
        print(sorted(os.listdir(configs_dir)))
        # pdb.set_trace()
        for cfg_name in sorted(os.listdir(configs_dir)):
            # only use the visualization configs
            if cfg_name.startswith("vis_"):
                print(f"reformatting for: {cfg_name}")
                cfg_path = os.path.join(configs_dir, cfg_name)
                # cfg = load_config(args=None, path_to_config=cfg_path)
                try:
                    run_reformat_output_dirs(cfg_path)
                except Exception as e:
                    print()
                    pdb.set_trace()
                    raise e