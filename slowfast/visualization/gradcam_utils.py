#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from slowfast.visualization.weight_calcs import get_model_weights
import slowfast.utils.logging as logging
import slowfast.datasets.utils as data_utils
from slowfast.visualization.utils import get_layer, replace_layer
from slowfast.visualization.connected_components_utils import (
    plot_heatmap,
    load_heatmaps,
)
from scripts import output_idx_to_input

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy
import sys
import cv2
import os
import pdb
import copy
import gc

logger = logging.get_logger(__name__)


class GradCAM:
    """
    GradCAM class helps create localization maps using the Grad-CAM method for input videos
    and overlap the maps over the input videos as heatmaps.
    https://arxiv.org/pdf/1610.02391.pdf
    """

    def __init__(
        self,
        model,
        target_layers,
        data_mean,
        data_std,
        method,
        cfg,
        post_softmax=False,
        colormap="viridis",
    ):
        """
        Args:
            model (model): the model to be used.
            target_layers (list of str(s)): name of convolutional layer to be used to get
                gradients and feature maps from for creating localization maps.
            data_mean (tensor or list): mean value to add to input videos.
            data_std (tensor or list): std to multiply for input videos.
            post_softmax (bool): whether to compute gradients with respect to
                post-softmax score. If True, gradient is wrt post-softmax; if
                False, gradient is wrt pre-softmax.
            colormap (Optional[str]): matplotlib colormap used to create heatmap.
                See https://matplotlib.org/3.3.0/tutorials/colors/colormaps.html
        """
        self.model = model
        self.post_softmax = post_softmax
        # Run in eval mode.
        self.model.eval()
        print("GC UTILS 63: init is running")
       
        if not post_softmax:
            #create copy of model without softmax for gradient computations
            model_copy = copy.deepcopy(self.model)

            self.model_without_softmax = replace_layer(
                model=model_copy,
                layer_name=cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.SOFTMAX_LAYER,
                replacement_layer=torch.nn.Identity(),
            )

            self.model_without_softmax.eval()

        

            ### TODO: troubleshoot 6/25/24

        self.target_layers = target_layers

        self.gradients = {}
        self.activations = {}
        self.method = method
        self.colormap = plt.get_cmap(colormap)
        self.data_mean = data_mean
        self.data_std = data_std

        if post_softmax:
            self._register_hooks(self.model)
        else:
            self._register_hooks(self.model_without_softmax)

    def _register_single_hook(self, model, layer_name):
        """Register forward and backward hook to a layer in a model to obtain
        gradients and activations.

        Args:
            model: model to which the hook is registered.
            layer_name (str): name of the layer.
        """

        def get_gradients(module, grad_input, grad_output):
            """Create a hook that will be called every time the gradients with
            respect to a module are computed.
            """
            self.gradients[layer_name] = grad_output[0].detach()

        def get_activations(module, input, output):
            """Create a hook that will be called every time after forward() has
            computed an output.

            The input contains only the positional arguments given to the module
            """
            self.activations[layer_name] = output.clone().detach()

        target_layer = get_layer(model, layer_name=layer_name)
        target_layer.register_forward_hook(get_activations)
        target_layer.register_full_backward_hook(get_gradients)

    def _register_hooks(self, model):
        """Register hooks to layers in `self.target_layers`.

        Args:
            model: model to which the hooks are registered.
        """
        for layer_name in self.target_layers:
            self._register_single_hook(model=model, layer_name=layer_name)

    def _calculate_localization_map(self, inputs, labels=None):
        """
        Calculate localization map for all inputs with Grad-CAM.
        Args:
            inputs (list of tensor(s)): the input clips.
            labels (Optional[tensor]): labels of the current input clips.
        Returns:
            localization_maps (list of ndarray(s)): the localization map for
                each corresponding input.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """

        assert len(inputs) == len(
            self.target_layers
        ), "Must register the same number of target layers as the number of input pathways."

        # input_clone = [inp.clone() for inp in inputs]
        # with torch.no_grad():
        #     input_clone = [inp.clone() for inp in inputs]
        #     print("no grad")
        #     preds = self.model(input_clone)


        # get scores for all classes
        if self.post_softmax:
            # compute score after softmax
            input_clone = [inp.clone() for inp in inputs]
            preds = self.model(input_clone)

            # print("GC UTILS 145: running post softmax before scores assigned")
            # # print(torch.cuda.memory_summary())
            scores_for_all_classes = preds
            # print("GC UTILS 148: running post softmax, preds set")
            # print(torch.cuda.memory_summary())
        else:
            with torch.no_grad():
                input_clone = [inp.clone() for inp in inputs]
                print("no grad")
                preds = self.model(input_clone)
            
            # compute score before softmax
            # print("GC UTILS 152: running PRE softmax before scores assigned")
            # print(torch.cuda.memory_summary())
            scores_for_all_classes = self.model_without_softmax(input_clone)

            # print("GC UTILS 155: running PRE softmax, preds set")
            # print(torch.cuda.memory_summary())
          

        # get score for single class
        if labels is None:  # get score for top predicted class
            score = torch.max(scores_for_all_classes, dim=-1)[0]
        else:  # get score for true class
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            score = torch.gather(scores_for_all_classes, dim=1, index=labels)

        # print("GC UTILS 176")
        # print(torch.cuda.memory_summary())

        # compute the gradient of the score wrt the target layers
        score = torch.sum(score)
        self.model.zero_grad()
        score.backward()  # Computes the gradient of current tensor w.r.t. graph leaves

        # print("GC UTILS 184")
        # print(torch.cuda.memory_summary())

        localization_maps = []
        for i, inp in enumerate(inputs):
            _, _, T, H, W = inp.size()



            gradients = self.gradients[self.target_layers[i]]
            activations = self.activations[self.target_layers[i]]
            B, C, Tg, _, _ = gradients.size()

            weights = get_model_weights(
                inputs=inputs,
                grads=gradients.view(B, C, Tg, -1),
                activations=activations.view(B, C, Tg, -1),
                method=self.method,
            )
            weights = weights.view(B, C, Tg, 1, 1)
            localization_map = torch.sum(
                weights * activations, dim=1, keepdim=True
            )
            localization_map = F.relu(localization_map)
            localization_map = F.interpolate(
                localization_map,
                size=(T, H, W),
                mode="trilinear",
                align_corners=False,
            )
            localization_map_min, localization_map_max = (
                torch.min(localization_map.view(B, -1), dim=-1, keepdim=True)[
                    0
                ],
                torch.max(localization_map.view(B, -1), dim=-1, keepdim=True)[
                    0
                ],
            )
            localization_map_min = torch.reshape(
                localization_map_min, shape=(B, 1, 1, 1, 1)
            )
            localization_map_max = torch.reshape(
                localization_map_max, shape=(B, 1, 1, 1, 1)
            )
            # Normalize the localization map.
            localization_map = (localization_map - localization_map_min) / (
                localization_map_max - localization_map_min + 1e-6
            )
            localization_map = localization_map.data

            localization_maps.append(localization_map)


        print("GC UTILS 221: running outside at the end of for loop")
        print(torch.cuda.memory_summary())


        return localization_maps, preds

    def __call__(
        self, output_dir, inputs, video_indices, cfg, labels=None, alpha=0.5
    ):
        """
        Visualize localization maps on their corresponding inputs as heatmaps
        using Grad-CAM, Eigen-CAM, or Grad-CAM++ and saves heatmap-related
        information

        Args:
            inputs (list of tensor(s)): the input clips.
            video_idx (tensor): index for the current input clip
            labels (Optional[tensor]): labels of the current input clips. If
                provided, gradcam will operate using the true labels; if None,
                gradcam will operate using predicted labels
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            alpha (float): transparency level of the heatmap, in the range [0, 1].
        Returns:
            result_ls (list of tensor(s)): the visualized inputs.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """
        alpha = 0.5
        result_ls = []

        # find latest checkpoint from output folder
        last_checkpoint = ""
        for __,__,file in os.walk(cfg.OUTPUT_DIR + "checkpoints/"):
            last_checkpoint = max(file)
        if (last_checkpoint != ""): 
            remove_file_ext = last_checkpoint.split(".")[0]
            epoch_selected = (remove_file_ext.split("_"))[2] # separating epoch number from file path
        else:
            epoch_selected = 0 # for pretrained models

        # retrieve heatmaps
        localization_maps, preds = self._calculate_localization_map(
            inputs,
            labels=labels,
        )

        # isolate the json file corresponding to the labels for the motion classes
        
        experiment_root_dir = "/".join(output_dir.split("/")[:-1])
        
        input_json_path = os.path.join(
            cfg.DATA.PATH_TO_DATA_DIR, "{}-test.json".format((cfg.TRAIN.DATASET).capitalize())
        )
        #TODO: I don't like having to capitalize here -- why do we have our JSONs named as Ucf.json instead of ucf.json? 

        # information for saving videos
        save_vid_overlay = cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.SAVE_OVERLAY_VIDEO
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        crop_size = (cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE)

        #######################################################################
        ##  SETTING UP ROOT DIRECTORIES FOR HEATMAPS                         ##
        #######################################################################

        # root directory for all heatmap-related functionality
        heatmaps_root_dir = os.path.join(
            output_dir,
            f"heatmaps_epoch_{epoch_selected}",
            cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.METHOD,
            "post_softmax"
            if cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.POST_SOFTMAX
            else "pre_softmax",
        )

        # root directory for heatmap frames
        heatmaps_frames_root_dir = os.path.join(
            heatmaps_root_dir,
            "frames",
        )

        # root directory for heatmap volumes
        heatmap_volume_root_dir = os.path.join(heatmaps_root_dir, "3d_volumes")

        # root directory for original input overlaid with heatmap frames
        heatmap_overlay_frames_root_dir = os.path.join(
            heatmaps_root_dir,
            "heatmap_overlay",
            "frames",
        )

        # root directory for original input video overlaid with heatmap activations
        heatmap_overlay_video_root_dir = os.path.join(
            heatmaps_root_dir,
            "heatmap_overlay",
            "videos",
        )

        # iterate over channels (e.g. rgb, slow, fast)
        for channel_idx, localization_map in enumerate(localization_maps):
            if cfg.MODEL.ARCH == "slowfast":
                channel = "slow" if channel_idx == 0 else "fast"
            elif cfg.MODEL.ARCH == "i3d":
                channel = "rgb"
            else:
                raise NotImplementedError(
                    "implement channel logic for this architecture"
                )

            # Convert localization map from (B, 1, T, H, W) to (B, T, H, W)
            localization_map = localization_map.squeeze(dim=1)
            if localization_map.device != torch.device("cpu"):
                localization_map = localization_map.cpu()

            heatmap = self.colormap(localization_map.numpy())
            heatmap = heatmap[:, :, :, :, :3]

            # Permute input from (B, C, T, H, W) to (B, T, H, W, C)
            curr_inp = inputs[channel_idx].permute(0, 2, 3, 4, 1)
            if curr_inp.device != torch.device("cpu"):
                curr_inp = curr_inp.cpu()
            curr_inp = data_utils.revert_tensor_normalize(
                curr_inp, self.data_mean, self.data_std
            )

            heatmap = torch.from_numpy(heatmap)
            curr_inp = alpha * heatmap + (1 - alpha) * curr_inp

            # Permute inp to (B, T, C, H, W)
            curr_inp = curr_inp.permute(0, 1, 4, 2, 3)

            result_ls.append(curr_inp)

            # iterate over videos in batch
            for v, vid_idx in enumerate(video_indices.numpy()):
                # heatmap to save
                map_to_save = localization_map.numpy()[v]

                # heatmap overlay to save
                overlay_to_save = curr_inp[v, :, :, :, :]

                # retrieve the target class and index for the current input
                target_class, input_vid_idx = output_idx_to_input(input_json_path, vid_idx)

                # paths for heatmap frames and overlaid frames, respectively
                visualization_path = os.path.join(
                    heatmaps_frames_root_dir, 
                    target_class,
                    # f"{target_class}_{input_vid_idx:06d}",
                    # TODO: we changed this on 6/13/24
                    f"{target_class}_{input_vid_idx}",
                    channel,
                )
                overlay_path = os.path.join(
                    heatmap_overlay_frames_root_dir, 
                    f"{input_vid_idx}" # TODO: Added this 6/14/24
                )

                # initialize video writer for saving overlaid heatmap video
                video_path = os.path.join(
                    heatmap_overlay_video_root_dir, 
                    f"{input_vid_idx}"
                )

                if not os.path.exists(video_path):
                    os.makedirs(video_path)

                if save_vid_overlay:
                    video_name = os.path.join(
                        video_path, f"{channel}_{vid_idx:06d}.mp4"
                    )
                    overlay_video = cv2.VideoWriter(
                        video_name, fourcc, 25, crop_size
                    )

                # iterate over frames in video
                for frame_idx in range(len(map_to_save)):
                    frame_map = map_to_save[frame_idx] * 255
                    overlay_map = (
                        overlay_to_save[frame_idx, :, :, :]
                        .permute(1, 2, 0)
                        .numpy()
                        * 255
                    )
                    if save_vid_overlay:
                        overlay_video.write(numpy.uint8(overlay_map))

                    one_based_frame_idx = frame_idx + 1
                    frame_name = (
                        # f"{target_class}_{input_vid_idx:06d}_{channel}_{one_based_frame_idx:06d}.jpg"
                        f"{target_class}_{input_vid_idx}_{channel}_{one_based_frame_idx:06d}.jpg"
                        # TODO: edited this 6/13/24
                    )

                    if not os.path.exists(visualization_path):
                        os.makedirs(visualization_path)
                    
                    frame_path = os.path.join(visualization_path, frame_name)
                    
                    overlay_frame_tag = (
                        f"{vid_idx:06d}_{one_based_frame_idx:06d}.jpg"
                    )
                    overlay_frame_name = os.path.join(
                        overlay_path, overlay_frame_tag
                    )

                    # save overlay frames
                    if cfg.MODEL.ARCH == "slowfast":
                        # heatmap channel folder
                        channel_folder = os.path.join(
                            visualization_path, channel
                        )

                        if save_vid_overlay:
                            overlay_channel_folder = os.path.join(
                                overlay_path, channel
                            )
                            overlay_frame_name = os.path.join(
                                overlay_channel_folder, overlay_frame_tag
                            )
                            if not os.path.exists(overlay_channel_folder):
                                os.makedirs(overlay_channel_folder)
                    elif cfg.MODEL.ARCH == "i3d":
                        # heatmap channel folder
                        channel_folder = os.path.join(
                            visualization_path, channel
                        )

                        if save_vid_overlay:
                            overlay_channel_folder = os.path.join(
                                overlay_path, channel
                            )
                            overlay_frame_name = os.path.join(
                                overlay_channel_folder, overlay_frame_tag
                            )
                            if not os.path.exists(overlay_channel_folder):
                                os.makedirs(overlay_channel_folder)

                    else:
                        # add in logic to save the overlays
                        raise NotImplementedError(
                            "make subfolders for each pathway and put frames in correct subfolder for the specific visualization method"
                        )

                    # save the heatmap frame
                    cv2.imwrite(frame_path, frame_map)
                    if save_vid_overlay:
                        cv2.imwrite(overlay_frame_name, overlay_map)
                if save_vid_overlay:
                    cv2.destroyAllWindows()
                    overlay_video.release()

                # generate 3d heatmap volumes for the video
                t_scale = 0.25
                s_scale = 1 / 8

                img_stack = load_heatmaps(
                    visualization_path, t_scale=t_scale, s_scale=s_scale
                )
                heatmap_fname = os.path.join(
                    heatmap_volume_root_dir,
                    target_class,
                    f"{target_class}_{input_vid_idx}",
                    f"{target_class}_{input_vid_idx}_{channel}.html",
                )

                plot_heatmap(
                    volume=img_stack,
                    fpath=heatmap_fname,
                    surface_count=8,
                    t_scale=t_scale,
                    s_scale=s_scale,
                    slider=True,
                )


        return result_ls, preds
