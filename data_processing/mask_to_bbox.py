# This file generates bounding boxes for the our synthetic dataset given the binary masks for the target objects.
# The minbbox function was pulled from CS153 HW3 starter code (Thanks prof. Wloka!).

# We convert the target masks for the test sets of all of the synthetic datasets into bounding boxes
# We will save the bounding box images in under the synthetic data folder in data
import os
import json
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb

# Change the experiment number to generate all bounding boxes
# exp1 = 
# exp2 = 
# exp3 = 
# exp4
# exp5

exp_masks = "/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_4/test/target_masks/"
output_folder = "/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_4/test/target_bboxes/"


def load_img(impath):
    """
    Loads an image from a specified location and returns it in RGB format.
    Input:
    - impath: a string specifying the target image location.
    Returns an RGB image.
    """
    img = cv2.imread(impath)
    return img


def minbbox(mask):
    """
    Takes in a binary mask and returns a minimal bounding box around
    the non-zero elements.
    Input:
    - mask: a mask image (should contain only 0 and non-zero elements)
    Returns:
    - bbox: a list of four points in the form: [min_x, min_y, max_x, max_y]
    """

    min_h = np.where(mask > 0)[0].min()
    max_h = np.where(mask > 0)[0].max()
    min_w = np.where(mask > 0)[1].min()
    max_w = np.where(mask > 0)[1].max()

    return [min_w, min_h, max_w, max_h]


def all_boxes(origin = exp_masks, destination = output_folder):

    # I hate os.walk. So much os.walk goes through each class, each video, then each frame
    for __, directory,_ in os.walk(exp_masks):
        for d in directory:
            for __,subdir,_ in os.walk(exp_masks + "/" + d + "/"):
                for sub in subdir:
                    for _,_,files in os.walk(exp_masks + "/" + d + "/" + sub + "/" ):
                        for f in files:
                            # mask is just a numpy array
                            mask = load_img(exp_masks + d + "/" + sub + "/" + f)


                            # pdb.set_trace()
                            dim1 = len(mask)
                            dim2 = len(mask[1])

                            # find the boundaries for the bounding boxes
                            left, top, right, bottom = minbbox(mask)

                            # create the bounding box
                            bbox = np.zeros([dim1, dim2])
                            for y in range(dim1):
                                for x in range(dim2):
                                    if y>=top and y<=bottom and x>= left and x<=right:
                                        bbox[y,x] = 1


                            video_folder = destination + d + "/"
                            # create the path to save
                            if not os.path.exists(video_folder):
                                os.mkdir(video_folder)
                            if not os.path.exists(video_folder + sub):
                                os.mkdir(video_folder + sub)
                            
                            # save the bbox as an image in the destination folder (and in the correct subdirectories of course)
                            plt.imsave(os.path.join(video_folder ,sub, f), bbox, cmap=cm.gray)
                            #pdb.set_trace()
                        

                    