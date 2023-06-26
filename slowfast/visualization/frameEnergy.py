import cv2
import os
import matplotlib.pyplot as plt
import numpy
import shutil
import random

# number of videos in test (47 for UCF, ??? for Kinetics)
NUM_VIDEOS = 47
# I don't think we use this anywhere... why did we put it in?
TEST = [
    "",
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


def one_frame_energy(name):
    """
    Computes the % activated of the total image as the frame energy
    input:
        name - a full file path to a heatmap image to read
    output:
        a float describing the frame's % activation
    """
    frame = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    count = 0
    height, width = frame.shape
    for i in range(height):
        for j in range(width):
            count += frame[i][j] / 255
    return count / (height * width)  # percent activated


def loop_frames(heatmap_name, folder, new_folder="", save=False):
    # start with frame0, will increment later on
    file_path = folder + heatmap_name + "frame0.jpg"
    count = 0
    energy = []

    while os.path.exists(file_path):
        energy += [one_frame_energy(file_path)]
        count += 1
        file_path = folder + heatmap_name + "frame" + str(count) + ".jpg"

    if save:
        fig = plt.figure()
        plt.plot(range(len(energy)), energy)
        fig.savefig(new_folder + heatmap_name + "plot.png")
        plt.close(fig)
        plt.close("all")
    return energy


def loop_videos(folder, new_folder, pathway_num=0):
    for i in range(1, NUM_VIDEOS + 1):
        heatmap_name = "heatmap" + str(i) + "pathway" + str(pathway_num)
        loop_frames(heatmap_name, folder, new_folder, save=True)
        print(i)


def max_energy_video(heatmap_num, heatmap_folder, model_name):
    heatmap_name = "heatmap" + str(heatmap_num) + "pathway1"
    energy = loop_frames(heatmap_name, heatmap_folder)
    max_energy_index = numpy.argmax(energy)

    # video_folder = "/research/cwloka/data/action_attn/output_i3d/inputs/"
    video_folder = "/research/cwloka/data/action_attn/output_slowfast/inputs/"
    max_frame_name = (
        "input" + str(heatmap_num) + "pathway1frame" + str(max_energy_index) + ".jpg"
    )
    print(max_frame_name)
    input_path = video_folder + max_frame_name

    output_folder = (
        "/research/cwloka/data/action_attn/ucf_high_singleframe_"
        + model_name
        + "/"
        + str(heatmap_num)
        + "/"
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(len(energy)):
        output_name = str(heatmap_num) + "_" + str(i) + ".jpg"
        shutil.copyfile(input_path, output_folder + output_name)


def min_energy_video(heatmap_num, heatmap_folder, model_name):
    heatmap_name = "heatmap" + str(heatmap_num) + "pathway1"
    energy = loop_frames(heatmap_name, heatmap_folder)
    min_energy_index = numpy.argmin(energy)

    if model_name == "i3d":
        video_folder = "/research/cwloka/data/action_attn/output_i3d/inputs/"
    else:
        video_folder = "/research/cwloka/data/action_attn/output_slowfast/inputs/"

    min_frame_name = (
        "input" + str(heatmap_num) + "pathway1frame" + str(min_energy_index) + ".jpg"
    )
    print(min_frame_name)
    input_path = video_folder + min_frame_name

    output_folder = (
        "/research/cwloka/data/action_attn/ucf_low_singleframe_"
        + model_name
        + "/"
        + str(heatmap_num)
        + "/"
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(len(energy)):
        output_name = str(heatmap_num) + "_" + str(i) + ".jpg"
        shutil.copyfile(input_path, output_folder + output_name)


def rand_energy_video(heatmap_num, heatmap_folder, model_name, j):
    if model_name == "fast":
        frames = 16
    else:
        frames = 64

    random_energy_index = random.randint(0, frames - 1)

    if model_name == "i3d":
        video_folder = "/research/cwloka/data/action_attn/output_i3d/inputs/"
    else:
        video_folder = "/research/cwloka/data/action_attn/output_slowfast/inputs/"

    random_frame_name = (
        "input" + str(heatmap_num) + "pathway1frame" + str(random_energy_index) + ".jpg"
    )
    input_path = video_folder + random_frame_name
    print(input_path)

    output_folder = (
        "/research/cwloka/data/action_attn/ucf_rand_"
        + str(j)
        + "_singleframe_"
        + model_name
        + "/"
        + str(heatmap_num)
        + "/"
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(0, frames):
        output_name = str(heatmap_num) + "_" + str(i) + ".jpg"
        shutil.copyfile(input_path, output_folder + output_name)
        print(output_folder + output_name)


def get_boring_videos(loop, heatmap_folder=None, frame_type=None, model_name=None):
    if loop:
        for model in ["i3d", "slow", "fast"]:
            if model == "i3d":
                heatmap_folder = (
                    "/research/cwloka/data/action_attn/output_i3d/heatmaps/"
                )
            elif model == "slow" or model == "fast":
                heatmap_folder = (
                    "/research/cwloka/data/action_attn/output_slowfast/heatmaps/"
                )
            else:
                print("model name incorrect, please input 'i3d', 'slow', or 'fast'")
                raise ValueError

            for type in ["max", "min"]:
                for i in range(1, 48):
                    loop_video(i, heatmap_folder, model, type)
            for rand_iter in range(5):
                for i in range(1, 48):
                    loop_video(i, heatmap_folder, model, "rand", rand_iter)
    else:
        if frame_type == "max":
            for i in range(1, 48):
                max_energy_video(i, heatmap_folder, model_name)
        elif frame_type == "min":
            for i in range(1, 48):
                min_energy_video(i, heatmap_folder, model_name)
        elif frame_type == "rand":
            for j in range(5):
                for i in range(1, 48):
                    rand_energy_video(i, heatmap_folder, model_name, j)


def loop_video(heatmap_num, heatmap_folder, model_name, frame_type, rand_iter=None):
    heatmap_name = "heatmap" + str(heatmap_num) + "pathway0"
    energy = loop_frames(heatmap_name, heatmap_folder)
    if frame_type == "max":
        energy_index = numpy.argmax(energy)
    elif frame_type == "min":
        energy_index = numpy.argmin(energy)
    elif frame_type == "rand":
        if model_name == "fast":
            num_frames = 16
        else:
            num_frames = 64
        energy_index = random.randint(0, num_frames - 1)
    else:
        print("wrong frame_type - please input 'min', 'max', or 'rand'")

    if energy_index < 3:
        frames = list(range(8)) * 8
    elif energy_index > 5:
        frames = list(range(56, 64)) * 8
    else:
        frames = list(range(energy_index - 3, energy_index + 5)) * 8

    if model_name == "i3d":
        video_folder = "/research/cwloka/data/action_attn/output_i3d/inputs/"
    else:
        video_folder = "/research/cwloka/data/action_attn/output_slowfast/inputs/"

    if rand_iter is None:
        output_folder = (
            "/research/cwloka/data/action_attn/ucf_"
            + frame_type
            + "_loop_"
            + model_name
            + "/"
            + str(heatmap_num)
            + "/"
        )
    else:
        output_folder = (
            "/research/cwloka/data/action_attn/ucf_"
            + frame_type
            + "_"
            + str(rand_iter)
            + "_loop_"
            + model_name
            + "/"
            + str(heatmap_num)
            + "/"
        )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(len(frames)):
        if model_name == "fast":
            frame_name = (
                "input" + str(heatmap_num) + "pathway1frame" + str(frames[i]) + ".jpg"
            )
        else:
            frame_name = (
                "input" + str(heatmap_num) + "pathway0frame" + str(frames[i]) + ".jpg"
            )
        input_path = video_folder + frame_name
        output_name = str(heatmap_num) + "_" + str(i) + ".jpg"
        shutil.copyfile(input_path, output_folder + output_name)
