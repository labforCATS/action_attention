import os
import pandas as pd
import json
import cv2
import csv
import urllib

###
# This script should be run after the Kinetics400 dataset is freshly downloaded and extracted (as mp4 files).
# For our purposes, we only need to run these functions on the test split because our model is pretrained on K400.
# As with other processing files, this should only need to be run once (6/26/2024) but a copy is kept here
# for reference.
###

UNCROPPED_DATA_DIRECTORY = "/research/cwloka/data/action_attn/alex_kinetics/kinetics-dataset/k400/test/"
# could be changed to train/ or val/ at the end if videos in these splits are ever needed

CROPPED_DATA_DIRECTORY = "/research/cwloka/data/action_attn/alex_kinetics/kinetics_small/" 

EXTRACTED_FRAMES_DIRECTORY = "/research/cwloka/data/action_attn/alex_kinetics/kinetics_extracted/"

ORIGINAL_ANNOTATIONS_TEST_CSV = "/research/cwloka/data/action_attn/alex_kinetics/kinetics-dataset/k400/annotations/test.csv"


def resize_videos(
    origin = UNCROPPED_DATA_DIRECTORY, destination = CROPPED_DATA_DIRECTORY
):
    """
    For each video in Kinetics, resize so that the shorter dimension is now 256 pixels.
    Aspect ratio should stay the same. 
    Resized videos are placed in the cropped data directory with the same file name.
    """
    for __, __, files in os.walk(origin):
        count = 0
        for f in files:
            print(f)
            if count % 1000 == 0:
                print("Number of Videos Resized: ", count)
                
            video_name = f[:-18] + ".mp4"

            vid = cv2.VideoCapture(origin + f)  # THIS IS ASSUMING KINETICS ISN'T SPLIT INTO MULTIPLE FOLDERS
            # Find height and width of video in pixels
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            print(f, "has height: ", height, " and width: ", width)

            if height > width: # in portrait mode
            # Crop the shorter side (width) to 256 pixels
                os.system(
                    "ffmpeg -i {0}  -vf scale='256:-2' {1}{2}".format(
                        origin + f, destination, video_name
                    )
                )
            else: # in landscape mode
            # Crop the shorter side (height) to 256 pixels
                os.system(
                    "ffmpeg -i {0}  -vf scale='-2:256' {1}{2}".format(
                        origin + f, destination, video_name
                    )
                )
            count += 1



def extract_frames(
    origin = CROPPED_DATA_DIRECTORY, destination = EXTRACTED_FRAMES_DIRECTORY, fps=30
):
    """
    Walk through the folder of kinetics videos, extract the name of the video,
    pull out the frames at a rate of fps
    Frames for each video are placed in a separate folder named after the original video.
    """
    for __, __, files in os.walk(origin):
        for f in files:
            if (f[-4:] == ".mp4"):
                # Exclude the .mp4 from the name
                video_name = f[:-4]

                # Create the final folder
                video_folder = destination + video_name + "/"

                if not os.path.exists(video_folder):
                    os.mkdir(video_folder)

                # Extract the frames
                os.system(
                    "ffmpeg -i {0}  -vf fps=fps={3} {1}{2}'_%06d.jpg'".format(
                        origin + f, video_folder, video_name, fps
                    )
                )
            else: 
                continue

def create_kinetics_test_label_json(destination = EXTRACTED_FRAMES_DIRECTORY):
    """
    Make necessary helper file for make_frames_csv().
    Ucf-test.json contains a list of dictionaries with the keys "id" (unique video name), "label" (non-unique action category), "template", and "placeholders". 
    Example: 
        {"id": "BreastStroke_g06_c03", "label": "BreastStroke", "template": "BreastStroke", "placeholders": []}
    """
    annotate = pd.read_csv(ORIGINAL_ANNOTATIONS_TEST_CSV)
    list_of_dictionaries = []

    # each row of the dataframe annotate contains info for one unique video in the test split
    for i in range(annotate.shape[0]): 
        if i % 1000 == 0:
            print(i)     
        label = annotate["label"].values[i]
        name = annotate["youtube_id"].values[i]  
        d = {"id": name}
        d["label"] = label
        d["template"] = label #TODO: why do we have duplicate label and template
        d["placeholders"] = [] #TODO: why is this here
        list_of_dictionaries += [d]

    # Save file
    with open(destination + "Kinetics-test.json", "w") as outfile:
        json.dump(list_of_dictionaries, outfile)


def make_frames_csv(origin = EXTRACTED_FRAMES_DIRECTORY):
    """
    Create a new test.csv file that is necessary for the dataloader, Kinetics_by_frame.py.
    Each row contains the corresponding "original_vido_id", "video_id", "frame_id", "path" (to individual frame),
    and "labels" (which contains " marks). Note that 'original_vido_id' is a typo from the original Facebook repo.
    
    Example (each row in this docstring example is a separate column in the CSV, all for one row):

    _-kAj5nEqM0 
    _-kAj5nEqM0 
    000074 
    /research/cwloka/data/action_attn/alex_kinetics_mini/kinetics_extracted/_-kAj5nEqM0/_-kAj5nEqM0_000074.jpg """"""

    """ 
    filename = "test.csv"
    
    with open(os.path.join(origin, "Kinetics-test.json"), "r") as f:
        test_dictionaries = json.load(f)

    test_count = 1
    # first element of test_csv_rows is the csv column headers
    test_csv_rows = [["original_vido_id", "video_id", "frame_id", "path", "''"]]
    for video in test_dictionaries:
        csv_id = video["id"]

        # endline = "''"
        # parse the video id from json file to get the name of the directory containing the frames
        folder = origin + video["id"] + "/"
        # walk through each of the frames
        for __,__,file in os.walk(folder):
            for f in file:
                if test_count % 100000 == 0:
                    print(test_count)
                # pull frame_num from file name and exclude ".jpg"
                frame_num = str(f.split("_")[-1])[:-4]
            # formatting for row to be added to the train.csv
                test_csv_rows += [[csv_id, csv_id, '{num:0{width}}'.format(num=int(frame_num), width=6), folder + f, '""']]
                test_count += 1

    # write to test.csv
    with open(origin + "test.csv", "w") as f:
        writer = csv.writer(f, delimiter = " ")
        for row in test_csv_rows:
            writer.writerow(row)