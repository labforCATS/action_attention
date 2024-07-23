import os
import pandas as pd
import json
import cv2
import csv
import random

"""
Functions should be run in order from top-to-bottom:
  resize_videos, extract_frames, make_test_csv, and create_json (in that order).
  random_crop and horizontal_flip were added for data augmentation (6/17/2024)


The UNCROPPED_DATA_DIRECTORY version of ucf50 was a fresh download from 
https://www.crcv.ucf.edu/data/UCF50.php. This file should not need to be run again
as data processing is hopefully a one-time process (6/13/2024). 

"""

UNCROPPED_DATA_DIRECTORY = "/research/cwloka/data/action_attn/ucf50/"
CROPPED_DATA_DIRECTORY = "/research/cwloka/data/action_attn/ucf50_small/"
EXTRACTED_FRAMES_DIRECTORY = "/research/cwloka/data/action_attn/ucf50_extracted/"
AUGMENTED_DATA_DIRECTORY = "/research/cwloka/data/action_attn/ucf50_augmented/"

def resize_videos(origin=UNCROPPED_DATA_DIRECTORY):
    """
    Crop any videos in the ucf50 dataset if needed (resize so that the shorter dimension is now 256 pixels). Aspect ratio stays the same.
    All videos, including those reformatted, are located in ucf50_small.
    """
    for __, directory,files in os.walk(origin):
        # Go through each subdirectory indicating the classification
        for d in directory:
            # Go through videos in each subdirectory
            for __,__,files in os.walk(origin + "/" + d + "/"):
                for f in files:
                    print(f) 

                    vid = cv2.VideoCapture(f)
                    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(height)
                    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                    print(width)

                    if height > width: # if is taller than wide
                        os.system(
                            "ffmpeg -i {0}  -vf scale='256:-1' {1}{2}".format(
                                origin + d + "/" + f, CROPPED_DATA_DIRECTORY, f
                            )
                        )
                    else: # if video is wider than tall
                        os.system(
                            "ffmpeg -i {0}  -vf scale='-1:256' {1}{2}".format(
                                origin + d + "/" + f, CROPPED_DATA_DIRECTORY, f
                            )
                        )
                    break

def extract_frames(origin=CROPPED_DATA_DIRECTORY, destination=EXTRACTED_FRAMES_DIRECTORY, fps=24):
    """
    Walk through the folder of cropped videos, then pull out the frames and write them into the new
    destination directory. 
    """
    # Walk through the folder
    for __, __, files in os.walk(origin):
        for f in files:

            # Exclude the file extension (.avi) from the name
            video_name = f[:-4]

            # Create the final folder within the ucf50_extracted directory
            video_folder = destination + video_name + "/"

            if not os.path.exists(video_folder):
                os.mkdir(video_folder)

            # Extract the frames
            os.system(
                "ffmpeg -i {0}  -vf fps=fps={3} {1}{2}'_%06d.jpg'".format(
                    origin + f, video_folder, video_name, fps
                )
            )

def make_label_csv(origin = CROPPED_DATA_DIRECTORY):
    """ Creates the label.csv file that will be used to create 
    the remaining essential JSON files in create_json() below. 

    label.csv has two columns, video_id and label. 
    video_id is the unique video name (e.g. BreastStroke_g06_c03). label is the action category (e.g BreastStroke).
    It has as many rows as there are input videos in origin. 
    """

    label_csv = []

    for root, dirs, files in os.walk(origin, topdown=False):
        for name in files:
             if(name[-4:] == ".avi"):
                label = name.split("_")[1]
                video_id = name[2:-4] # remove v_ and .api
                label_csv += [[video_id, label]]

    with open(CROPPED_DATA_DIRECTORY + "label.csv", 'w') as file:
        writer = csv.writer(file)
        fields = ["video_id", "label"]
        writer.writerow(fields)
        for row in label_csv:
            writer.writerow(row)

def create_json(origin = CROPPED_DATA_DIRECTORY):
    """
    Creates all JSON and CSV files needed for data annotations.
      Ucf-labels.json matches category labels (e.g. BreastStroke) with a unique category number (e.g. 0). 
      Ucf-test.json, Ucf-train.json, and Ucf-validation.json each contain a list of dictionaries with 
        the keys "id" (unique video name), "label" (non-unique action category), "template", and "placeholders". 
      train.csv and val.csv contain numbering and pathways for individual frames needed for training model.

      {"id": "BreastStroke_g06_c03", "label": "BreastStroke", "template": "BreastStroke", "placeholders": []}

    Uses a hard-coded train/test/validation split (80/10/10) to divide data.
    """
    annotate = pd.read_csv(CROPPED_DATA_DIRECTORY + "label.csv")

    ### Ucf-labels.json ###
    label_dictionary = {}
    label_count = 0

    for a in annotate["label"]:
        if a not in label_dictionary.keys():
            label_dictionary[a] = label_count
            label_count += 1

    with open(CROPPED_DATA_DIRECTORY + "Ucf-labels.json", "w") as outfile:
        json.dump(label_dictionary, outfile)

    ### Ucf-test.json, Ucf-train.json, and Ucf-validation.json ###

    list_of_dictionaries = []
    list_of_names = []

    for __, __, files in os.walk(origin):
        for filename in files:
            if(filename[-4:] == ".avi"):
                video_id = filename[2:-4] # remove "v_" and ".avi" from video_id
            else:
                continue # ignore existing csv and json files
            list_of_names += [video_id]

    # Walk through the names with annotation
    for i in range(len(list_of_names)):
        name = list_of_names[i]
        if i % 100 == 0:
            print(i)
        d = {"id": name} # d is a dictionary with all info for a single video
        label = annotate.loc[annotate["video_id"] == name]["label"].values[0]
        d["label"] = label
        d["template"] = label
        d["placeholders"] = []

        list_of_dictionaries += [d]

    # 80/10/10 train/validation/test split
    train_dictionaries = list_of_dictionaries[669:6014]
    val_dictionaries = list_of_dictionaries[6015:]
    test_dictionaries = list_of_dictionaries[0:668]

    with open(CROPPED_DATA_DIRECTORY + "Ucf-test.json", "w") as outfile:
        json.dump(test_dictionaries, outfile)
    with open(CROPPED_DATA_DIRECTORY + "Ucf-train.json", "w") as outfile:
        json.dump(train_dictionaries, outfile)
    with open(CROPPED_DATA_DIRECTORY + "Ucf-val.json", "w") as outfile:
        json.dump(val_dictionaries, outfile)

    ### CSVs ###

    # create counter for number of frames processed
    val_count = 1
        # first element of val_csv_rows is the csv column headers
    val_csv_rows = [["original_vido_id", "video_id", "frame_id", "path", "''"]]
    # loop through each of the videos in the json file 
    for video in val_dictionaries:
        csv_id = video["id"]
        # endline = "''"
        # parse the video id from json file to get the name of the directory containing the frames
        folder = EXTRACTED_FRAMES_DIRECTORY + "v_" + video["id"] + "/"
        # walk through each of the frames
        for __,__,file in os.walk(folder):
            for f in file:
                # formatting for row to be added to the val.csv
                frame_num = str(f.split("_")[-1])[:-4]
                val_csv_rows += [[csv_id, csv_id,'{num:0{width}}'.format(num=int(frame_num), width=6), folder + f, '""']]
                val_count += 1
                if val_count % 10000 == 0:
                    print(val_count)

    # write to val.csv
    with open(CROPPED_DATA_DIRECTORY + "val.csv", "w") as f:
        writer = csv.writer(f, delimiter = " ")
        for row in val_csv_rows:
            writer.writerow(row)

    # repeat process to create train.csv
    train_count = 1
    # first element of train_csv_rows is the csv column headers
    train_csv_rows = [["original_vido_id", "video_id", "frame_id", "path", "''"]]
    for video in train_dictionaries:
        csv_id = video["id"]
        # endline = "''"
        # parse the video id from json file to get the name of the directory containing the frames
        folder = EXTRACTED_FRAMES_DIRECTORY + "v_" + video["id"] + "/"
        # walk through each of the frames
        for __,__,file in os.walk(folder):
            for f in file:
                # pull frame_num from file name and exclude ".jpg"
                frame_num = str(f.split("_")[-1])[:-4]
                # formatting for row to be added to the train.csv
                train_csv_rows += [[csv_id, csv_id, '{num:0{width}}'.format(num=int(frame_num), width=6), folder + f, '""']]
                train_count += 1
                if train_count % 100000 == 0:
                    print(train_count)

    # write to train.csv
    with open(CROPPED_DATA_DIRECTORY + "train.csv", "w") as f:
        writer = csv.writer(f, delimiter = " ")
        for row in train_csv_rows:
            writer.writerow(row)

    # repeat process to create test.csv
    test_count = 1
    # first element of test_csv_rows is the csv column headers
    test_csv_rows = [["original_vido_id", "video_id", "frame_id", "path", "''"]]
    for video in test_dictionaries:
        csv_id = video["id"]
        # endline = "''"
        # parse the video id from json file to get the name of the directory containing the frames
        folder = EXTRACTED_FRAMES_DIRECTORY + "v_" + video["id"] + "/"
        # walk through each of the frames
        for __,__,file in os.walk(folder):
            for f in file:
                # pull frame_num from file name and exclude ".jpg"
                frame_num = str(f.split("_")[-1])[:-4]
                # formatting for row to be added to the train.csv
                test_csv_rows += [[csv_id, csv_id, '{num:0{width}}'.format(num=int(frame_num), width=6), folder + f, '""']]
                test_count += 1
                if test_count % 10000 == 0:
                    print(test_count)

    # write to test.csv
    with open(CROPPED_DATA_DIRECTORY + "test.csv", "w") as f:
        writer = csv.writer(f, delimiter = " ")
        for row in test_csv_rows:
            writer.writerow(row)


def random_crop(origin = CROPPED_DATA_DIRECTORY, destination= AUGMENTED_DATA_DIRECTORY):
    """
    Takes in original ucf50 videos and crops the videos centered about a random point.
    This is contrary to the function resize_videos which crops about the center. 

    The outputed videos are created to augment our dataset
     and are placed in the destination directory
    """
    # Go through videos in each subdirectory
    for __,__,files in os.walk(origin):
        for f in files:
            if (f[-4:] == ".avi"):
                print(f) 

                video_folder = destination + "random_crop/"

                vid = cv2.VideoCapture(f)
                height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(height)
                width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                print(width)

                if (width > height):
                    y_ref = int(22 * random.random())
                    x_ref = int(0.08 * width * random.random())
                    os.system(
                        "ffmpeg -i {0} -vf scale=-'1:278', crop=height:width:y_ref:x_ref {1}{2}".format(
                            origin + f, video_folder, f 
                        )
                    )
                else: 
                    y_ref = int(0.08 * height * random.random())
                    x_ref = int(22 * random.random())
                    os.system(
                        "ffmpeg -i {0} -vf scale='278:-1', crop =height:width:y_ref:x_ref {1}{2}".format(
                            origin + f, video_folder, f 
                        )
                    )

                return

   

def horizontal_flip(origin = UNCROPPED_DATA_DIRECTORY, destination = AUGMENTED_DATA_DIRECTORY):
    """
    Takes in original ucf50 videos and crops the videos centered about a random point.
    This is contrary to the function resize_videos which crops about the center. 

    The outputed videos are created to increase our dataset
     and are placed in the destination directory
    """

    for __, directory,files in os.walk(origin):
        # Go through each subdirectory indicating the classification
        for d in directory:
            # Go through videos in each subdirectory
            for __,__,files in os.walk(origin + "/" + d + "/"):
                for f in files:

                    # new directory for flipped videos
                    video_folder = destination + "horizonatal_flip/"

                    if not os.path.exists(video_folder):
                        os.mkdir(video_folder)

                    os.system(
                        # horizontal flip videos
                        "ffmpeg -i {0} -vf hflip -c:a copy {1}{2}".format(
                            origin + d + "/" + f, video_folder, f 
                        )
                    )
