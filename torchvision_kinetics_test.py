# Code copied from "facebookresearch_pytorchvideo_slowfast.ipynb" colab notebook
# Accessed from: https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/


import torch
import torchvision
import cv2

# Choose the `slowfast_r50` model 
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 
# from kinetics_by_frame import KineticsByFrame
# ^ cant do that because of detectron 2 and kinetics_by_frame.py is dependent on slowfast module. 

device = "cpu"
model = model.eval()
model = model.to(device)


json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
try: urllib.URLopener().retrieve(json_url, json_filename)
except: urllib.request.urlretrieve(json_url, json_filename)


with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")


side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
num_clips = 10
num_crops = 3

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)
# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second




KINETICS_VIDEO_PATH = "/research/cwloka/data/action_attn/kinetics_small/"

# Read json file for test-set to get video names (id) and correct label
with open("/research/cwloka/data/action_attn/kinetics_extracted/Kinetics-test.json", "r") as read_content: 
    kinetics_dicts = (json.load(read_content))

# Only testing first 500 videos
kinetics_dicts = kinetics_dicts[:2000]

accuracy_count = 0
iteration = 0

for kinetics_vid in kinetics_dicts:
    video_path = KINETICS_VIDEO_PATH + kinetics_vid["id"] + ".mp4"

    # Initialize an EncodedVideo helper class and load the video
    video = EncodedVideo.from_path(video_path)

    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0

    temp_video = cv2.VideoCapture(video_path)
    fps = temp_video.get(cv2.CAP_PROP_FPS)
    totalNoFrames = temp_video.get(cv2.CAP_PROP_FRAME_COUNT)
    clip_duration = totalNoFrames / fps

    end_sec = clip_duration

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)


    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = [i.to(device)[None, ...] for i in inputs]


    # Pass the input clip through the model
    preds = model(inputs)

    # Get the predicted classes
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=1).indices[0]

    # Map the predicted classes to the label names
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    # print("Top predicted label: %s" % ", ".join(pred_class_names))
    # print("Real label: ", kinetics_vid["label"])

    if (pred_class_names[0] == kinetics_vid["label"]):
        accuracy_count += 1
    
    iteration += 1
    if (iteration%100 == 0):
        print(iteration)
        

print("total correct videos:", accuracy_count)
print("accuracy is: ", accuracy_count/2000)