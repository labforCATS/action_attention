import os
import pandas as pd

# import ffmpeg
import json
import cv2
import csv

KINETICS_FOLDER = (
    "/research/cwloka/data/action_attn/kinetics/kinetics-dataset/k400/test/"
)
KINETICS_LABELS_PATH = (
    "/research/cwloka/data/action_attn/kinetics_extracted/Kinetics-labels.json"
)
TEST_CSV_PATH = "/research/cwloka/data/action_attn/kinetics/kinetics-dataset/k400/annotations/test.csv"
TEST_ANSWERS_PATH = "/research/cwloka/data/action_attn/kinetics_small/test.csv"
DESTINATION_FOLDER = "/research/cwloka/data/action_attn/kinetics_small/"


def resize_videos(
    origin="/research/cwloka/data/action_attn/kinetics/kinetics-dataset/k400/test/",
):
    for __, __, files in os.walk(origin):
        total = len(files)
        count = 0
        for f in files:
            if count % 1000 == 0:
                print(count)
            video_name = f[:-18] + ".mp4"
            vid = cv2.VideoCapture(f)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            if height > width:
                os.system(
                    "ffmpeg -i {0}  -vf scale='256:-1' {1}{2}".format(
                        origin + f, DESTINATION_FOLDER, video_name
                    )
                )
            else:
                os.system(
                    "ffmpeg -i {0}  -vf scale='-1:256' {1}{2}".format(
                        origin + f, DESTINATION_FOLDER, video_name
                    )
                )
            count += 1


# read every video and turn it into frames
def extract_frames(origin=KINETICS_FOLDER, destination=DESTINATION_FOLDER, fps=24):
    """
    Walk through the folder of kinetics videos, extract the name of the video,
    pull out the frames at a rate of fps
    """
    # Walk through the folder
    for __, __, files in os.walk(origin):
        for f in files:

            # Exclude the timestamps from the name (checked against csv file)
            video_name = f[:-18]

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


# Create the label file
def action_labels():
    """
    Pull the labels from the annotation file, number them, and put in json file
    """
    # Read csv
    annotate = pd.read_csv(TEST_CSV_PATH)

    # Initiate variables
    dictionary = {}
    count = 0

    # Iterate through the dataframe
    for a in annotate["label"]:
        if a not in dictionary.keys():
            dictionary[a] = count
            count += 1
    # print(dictionary)
    # Save file
    with open(DESTINATION_FOLDER + "Kinetics-labels.json", "w") as outfile:
        json.dump(dictionary, outfile)


def ucf_test_file_creation(origin=KINETICS_FOLDER):
    """
    Make the file with a list of dictionaries for the labels of each video
    """
    annotate = pd.read_csv(TEST_CSV_PATH)

    # Initiate variables
    list_of_dictionaries = []
    list_of_names = []

    # Walk through the folder
    for __, __, files in os.walk(origin):
        for f in files:

            # Exclude the timestamps from the name (checked against csv file)
            video_name = f[:-18]

            list_of_names += [video_name]

    print(len(list_of_names))

    # Walk through the names with annotation
    for i in range(len(list_of_names)):
        name = list_of_names[i]
        if i % 100 == 0:
            print(i)
        d = {"id": name}
        label = annotate.loc[annotate["youtube_id"] == name]["label"].values[0]
        d["label"] = label
        d["template"] = label
        d["placeholders"] = []
        list_of_dictionaries += [d]

    # Save file
    with open(DESTINATION_FOLDER + "Kinetics-test.json", "w") as outfile:
        json.dump(list_of_dictionaries, outfile)


def make_test_answers():
    # load in Kinetics-labels.json to get string/number label correspondence
    labels_json = open(KINETICS_LABELS_PATH)
    labels = json.load(labels_json)

    # load in test csv
    test = pd.read_csv(TEST_CSV_PATH)

    # start list for test_answers.csv
    test_answers = []

    # iterate through videos
    for row in test.T.items():
        id_n = DESTINATION_FOLDER + row[1]["youtube_id"] + ".mp4"
        label = row[1]["label"]
        test_answers += [[id_n, labels[label]]]

    # write to test_answers.csv
    with open(TEST_ANSWERS_PATH, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        for row in test_answers:
            writer.writerow(row)
