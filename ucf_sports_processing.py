import json
import shutil
import os
import csv
import cv2
import numpy as np

### Data loader for UCF sports data set. Should not need to be run again but keeping a copy for records.
### Train/val splits and annotations are hardcoded in; this should *not* be used as a template for 
### any future data processing. Validation and testing data are the same. 
### Uses 150 videos from UCF sports original and 103 augmented versions. 

# Text labels corresponding to the numbers
labels = [
    "Diving",
    "Golf-Swing",
    "Kicking",
    "Lifting",
    "Riding-Horse",
    "Run",
    "Skateboarding",
    "SwingBench",
    "SwingBar",
    "Walk",
]

# Video names for testing
test = [
    "001",
    "002",
    "003",
    "004",
    "015",
    "016",
    "017",
    "018",
    "019",
    "020",
    "033",
    "034",
    "035",
    "036",
    "037",
    "038",
    "053",
    "054",
    "059",
    "060",
    "061",
    "062",
    "071",
    "072",
    "073",
    "074",
    "084",
    "085",
    "086",
    "087",
    "096",
    "097",
    "098",
    "099",
    "100",
    "101",
    "116",
    "117",
    "118",
    "119",
    "129",
    "130",
    "131",
    "132",
    "133",
    "134",
    "135",
]

# Video names for training (numbers above 150 are horizontally flipped videos)
train = [
    "005",
    "006",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "014",
    "021",
    "022",
    "023",
    "024",
    "025",
    "026",
    "027",
    "028",
    "029",
    "030",
    "031",
    "032",
    "039",
    "040",
    "041",
    "042",
    "043",
    "044",
    "045",
    "046",
    "047",
    "048",
    "049",
    "050",
    "051",
    "052",
    "055",
    "056",
    "057",
    "058",
    "063",
    "064",
    "065",
    "066",
    "067",
    "068",
    "069",
    "070",
    "075",
    "076",
    "077",
    "078",
    "079",
    "080",
    "081",
    "082",
    "083",
    "088",
    "089",
    "090",
    "091",
    "092",
    "093",
    "094",
    "095",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "110",
    "111",
    "112",
    "113",
    "114",
    "115",
    "120",
    "121",
    "122",
    "123",
    "124",
    "125",
    "126",
    "127",
    "128",
    "136",
    "137",
    "138",
    "139",
    "140",
    "141",
    "142",
    "143",
    "144",
    "145",
    "146",
    "147",
    "148",
    "149",
    "150",
    "151",
    "152",
    "153",
    "154",
    "155",
    "156",
    "157",
    "158",
    "159",
    "160",
    "161",
    "162",
    "163",
    "164",
    "165",
    "166",
    "167",
    "168",
    "169",
    "170",
    "171",
    "172",
    "173",
    "174",
    "175",
    "176",
    "177",
    "178",
    "179",
    "180",
    "181",
    "182",
    "183",
    "184",
    "185",
    "186",
    "187",
    "188",
    "189",
    "190",
    "191",
    "192",
    "193",
    "194",
    "195",
    "196",
    "197",
    "198",
    "199",
    "200",
    "201",
    "202",
    "203",
    "204",
    "205",
    "206",
    "207",
    "208",
    "209",
    "210",
    "211",
    "212",
    "213",
    "214",
    "215",
    "216",
    "217",
    "218",
    "219",
    "220",
    "221",
    "222",
    "223",
    "224",
    "225",
    "226",
    "227",
    "228",
    "229",
    "230",
    "231",
    "232",
    "233",
    "234",
    "235",
    "236",
    "237",
    "238",
    "239",
    "240",
    "241",
    "242",
    "243",
    "244",
    "245",
    "246",
    "247",
    "248",
    "249",
    "250",
    "251",
    "252",
    "253",
]

# Labels for all ucf videos, including copied training videos for the flips
answers = [
    -1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
]

# Number of each class for caluculating weights
count = [14, 18, 20, 6, 12, 13, 12, 20, 13, 22]


# Start relevant functions
def switch_to_json(filename, new_filename):
    """
    Moves a txt file with two columns into json format
    (switching order of columns)

    Useful for turning the originally-downloaded 
    UCF sports dataset labels to the format for SlowFast.
    """
    # Read in the file
    with open(filename, "r") as f:
        lines = f.readlines()

    # Split the lines
    for i in range(len(lines)):
        lines[i] = lines[i].split()

    # Move it into a dictionary
    new_lines = {}
    for line in lines:
        new_lines[line[1]] = line[0]

    # Create the new file
    with open(new_filename, "w") as f:
        json.dump(new_lines, f)


def create_json(path):
    """
    Hard-coded method to create a json
    file with the ucf sports labels
    and place it in path directory
    """
    # Create the dictionary
    dictionary = {
        "Diving": 0,
        "Golf-Swing": 1,
        "Kicking": 2,
        "Lifting": 3,
        "Riding-Horse": 4,
        "Run": 5,
        "Skateboarding": 6,
        "Swing": 7,
        "Walk": 8,
    }

    # Write it to the directory
    with open(path + "Ucf-labels.json", "w") as f:
        json.dump(dictionary, f)


def create_frame_folders(path):
    """
    Take the frames with original labels
    and rename them, order folders, and
    place the frames in correct order

    Useful immediately after downloding ucf sports

    Some folders do not have frames -- use ffmpeg
    to pull them out
    """
    # Create the folders for each video
    os.mkdir(os.path.join(path, "data"))
    for i in range(1, 151):
        name = "00" + str(i)
        new_path = os.path.join(path, "data/" + name[-3:])
        os.mkdir(new_path)

    # Walk the path and pull all of the video frames out, then
    #   copy them to the proper place
    allfiles = list(os.walk(path))
    for item in allfiles:
        foldername, LoDirs, LoFiles = item
        for filename in LoFiles:
            if foldername[-4:] != "jpeg" and foldername[-5:] != "jpeg2":
                if filename[-3:] == "jpg":
                    filename_parts = filename.split("_")
                    foldername_parts = foldername.split("\\")
                    new_filename = (
                        ".\\data\\"
                        + foldername_parts[-1]
                        + "\\"
                        + foldername_parts[-1]
                        + "_"
                        + filename_parts[-1]
                    )
                    old_filename = foldername + "\\" + filename
                    shutil.copyfile(old_filename, new_filename)


def single_val():
    """
    Create a val json for a single video
    Parameters are hard coded
    """
    test_json = [{"id": "1", "label": "Walk", "template": "Walk", "placeholders": []}]
    with open("./Richard2/Ucf-validation.json", "w") as f:
        json.dump(test_json, f)


def create_train_and_val_json(path):
    """
    From a list of ids create a json for training,
    test, and validation.

    Argument is directory in which to place it
    """

    # Create a list of dictionaries from test list
    test_json = []
    for video in test:
        id = answers[int(video)]
        diction = {
            "id": video,
            "label": labels[id],
            "template": labels[id],
            "placeholders": [],
        }
        test_json += [diction]

    # Write to the testing json
    with open(path + "Ucf-test.json", "w") as f:
        json.dump(test_json, f)

    # Write to the validation json (same data for ucf sports)
    with open(path + "Ucf-val.json", "w") as f:
        json.dump(test_json, f)

    # Create a list of dictionaries from train list
    train_json = []
    for video in train:
        id = answers[int(video)]
        diction = {
            "id": video,
            "label": labels[id],
            "template": labels[id],
            "placeholders": [],
        }
        train_json += [diction]

    # Write to the training json
    with open(path + "Ucf-train.json", "w") as f:
        json.dump(train_json, f)


def create_test_csv(path):
    """
    Create the test answers csv

    Argument is directory to write to
    """
    # Find the label for each test video
    csv1 = []
    for vid in test:
        id = answers[int(vid)]
        csv1 += [[vid, labels[id]]]

    # Write labels to the file
    with open(path + "test_answers.csv", "w") as f:
        writer = csv.writer(f)
        for row in csv1:
            writer.writerow(row)


def create_train_csv(write_path):
    """
    Create the train csv

    Argument is directory it write to
    """
    # Create the header for the csv file
    new_csv = [["original_vido_id", "video_id", "frame_id", "path", "''"]]

    # Loop through the training videos
    for vid in train:
        path = write_path + vid
        allfiles = list(os.walk(path))

        # Pull the frames out per video
        for item in allfiles:
            foldername, LoDirs, LoFiles = item
            for filename in LoFiles:
                if filename[-3:] == "jpg":
                    filename_parts = filename.split("_")
                    new_path = (
                        "/media/cats/32b7c353-4595-42d8-81aa-d029f1556567/ucf/"
                        + vid
                        + "/"
                        + filename
                    )

                    # Create the row for the frame
                    row = [int(vid), int(vid), filename_parts[-1][:-4], new_path, '""']
                    new_csv += [row]

    # Write the data to a csv
    with open(write_path + "train.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(new_csv)


boring_test = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
]


def create_val_csv(new_path):
    """
    Create the val csv

    Argument is the directory that the data is written to
    """
    # Create the header
    new_csv = [["original_vido_id", "video_id", "frame_id", "path", "''"]]

    # Loop through the testing videos
    for vid in boring_test:
        path = new_path + vid
        allfiles = list(os.walk(path))
        for item in allfiles:
            foldername, LoDirs, LoFiles = item
            for filename in LoFiles:
                if filename[-3:] == "jpg":
                    filename_parts = filename.split("_")
                    # new_path = "/media/cats/32b7c353-4595-42d8-81aa-d029f1556567/ucf/" + vid + "/" + filename
                    new_p = (
                        "/research/cwloka/data/action_attn/ucf_singleframe/"
                        + vid
                        + "/"
                        + filename
                    )

                    # Create the row for the frame
                    row = [int(vid), int(vid), filename_parts[-1][:-4], new_p, '""']
                    new_csv += [row]

    # Write the data to the file
    with open(new_path + "val.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(new_csv)


def rename_frames(path):
    """
    Take original ucf sports frames and rename them in order

    Argument is current path
    """
    # Loop through videos
    for video in test + train:
        frames = []
        files = list(os.walk(path + video + "/"))
        for item in files:
            foldername, LoDirs, LoFiles = item
            for filename in LoFiles:
                frames += [filename]
        count = 1

        # Loop through frames
        for file in frames:
            number = "00000" + str(count)
            new_name = path + video + "/" + video + "_" + number[-6:] + ".jpg"
            os.rename(path + video + "/" + file, new_name)
            count += 1


def resize_frames(original_path, new_path):
    path = original_path
    heights = []
    vids = test + train

    # Loop through all the videos
    for video in vids:
        frames = []
        files = list(os.walk(path + video + "/"))

        # Loop through all the frames
        for item in files:
            foldername, LoDirs, LoFiles = item
            for filename in LoFiles:
                if filename[-3:] == "jpg":
                    image = path + video + "/" + filename
                    img = cv2.imread(image)
                    h, w, _ = img.shape
                    if [h, w] not in heights:
                        heights += [[h, w]]

                    # If it is horizontal frame
                    if h < w:
                        new_h = 256
                        new_w = int((256 / h) * w)

                        # Resize the image
                        resized_img = cv2.resize(img, (new_w, new_h))
                        buffer = int((new_w - 256) / 2)
                        final_img = np.zeros([256, 256, 3])

                        # Center crop
                        for x in range(new_h):
                            for y in range(buffer, buffer + 256):
                                final_img[x][y - buffer] = resized_img[x][y]
                        cv2.imwrite(new_path + video + "/" + filename, final_img)

                    # If the video is vertical
                    else:
                        new_w = 256
                        new_h = int((256 / w) * h)

                        # Resize the image
                        resized_img = cv2.resize(img, (new_w, new_h))
                        buffer = int((new_h - 256) / 2)
                        final_img = np.zeros([256, 256, 3])

                        # Center crop
                        for x in range(buffer, buffer + 256):
                            for y in range(new_w):
                                final_img[x - buffer][y] = resized_img[x][y]
                        cv2.imwrite(new_path + video + "/" + filename, final_img)


def flip_horizontally(new_path):
    """
    Create new, horizontally flipped videos
    from the training data

    new_path is where to place them
    """
    count = 151

    # Loop through the training videos
    for vid in train:
        path = new_path + vid + "/"
        os.mkdir(new_path + str(count))
        files = list(os.walk(path))

        # Loop through the frames
        for item in files:
            foldername, LoDirs, LoFiles = item
            for filename in LoFiles:
                if filename[-3:] == "jpg":
                    image = path + filename
                    img = cv2.imread(image)
                    final_img = np.zeros([256, 256, 3])

                    # Switch the pixels
                    for x in range(256):
                        for y in range(256):
                            final_img[x][255 - y] = img[x][y]

                    # Write the final images
                    cv2.imwrite(new_path + str(count) + "/" + filename, final_img)
        count += 1


