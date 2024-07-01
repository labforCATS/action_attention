# This file contains helper functions to reformat the extracted frames of
# Kinetics data to be usable with UCF dataloading. The functions should only
# need to be run once, but are saved here for the sake of future reference in
# case it needs to be run again

import os
import json
import shutil
from tqdm import tqdm


def make_csv(fdir="/research/cwloka/data/action_attn/kinetics_extracted"):
    """Create a new <test/val/train>.csv file that follows the format of the
    UCF csvs.
    """
    fname = "test_NEW.csv"

    kinetics_test_fname = "Kinetics-test.json"

    with open(os.path.join(fdir, kinetics_test_fname), "r") as f:
        kinetics_test_dicts = json.load(f)

    all_files = os.listdir(fdir)
    print(len(all_files))

    video_id_list = [
        kinetics_test_dicts[i]["id"] for i in range(len(kinetics_test_dicts))
    ]
    print(len(video_id_list))

    path_to_csv = os.path.join(fdir, fname)
    with open(path_to_csv, "w") as f:
        header = "original_vido_id video_id frame_id path labels\n"
        f.write(header)


        for video_id in tqdm(video_id_list, total=len(video_id_list)):
            video_dir = os.path.join(fdir, video_id)

            for frame_id in range(1, 1 + len(os.listdir(video_dir))):
                f.write(
                    "{} {} {} {} {}\n".format(
                        video_id,
                        video_id,
                        frame_id,
                        os.path.join(fdir, video_id, f"{video_id}_{frame_id:06d}.jpg"),
                        '""',
                    )
                )


def make_mini():
    """Create a new data directory containing a subset of 10 videos from the
    extracted Kinetics dataset
    """
    fdir = "/research/cwloka/data/action_attn/kinetics_extracted"
    fdir_mini = fdir + "_mini"

    # make Kinetics-test.json
    kinetics_test_fname = "Kinetics-test.json"

    with open(os.path.join(fdir, kinetics_test_fname), "r") as f:
        kinetics_test_dicts_orig = json.load(f)

    kinetics_test_dicts_mini = kinetics_test_dicts_orig[:10]

    all_files = os.listdir(fdir)

    for d in kinetics_test_dicts_mini:
        assert d["id"] in all_files

        # copy over the subfolders of data
        if not os.path.exists(os.path.join(fdir_mini, d["id"])):
            shutil.copytree(
                os.path.join(fdir, d["id"]), os.path.join(fdir_mini, d["id"])
            )

    # create the json file
    with open(os.path.join(fdir_mini, kinetics_test_fname), "w") as f:
        kinetics_test_json_mini = json.dumps(kinetics_test_dicts_mini, indent=2)
        f.write(kinetics_test_json_mini)

    # make test_NEW.csv
    make_csv(fdir_mini)
