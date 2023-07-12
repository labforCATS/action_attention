#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from slowfast.visualization.weight_calcs import get_model_weights

import slowfast.datasets.utils as data_utils
from slowfast.visualization.utils import get_layer
from slowfast.utils.connected_components_utils import (
    plot_heatmap,
    load_heatmaps,
)

import numpy
import sys
import cv2
import os
import pdb


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
        colormap="viridis",
    ):
        """
        Args:
            model (model): the model to be used.
            target_layers (list of str(s)): name of convolutional layer to be used to get
                gradients and feature maps from for creating localization maps.
            data_mean (tensor or list): mean value to add to input videos.
            data_std (tensor or list): std to multiply for input videos.
            colormap (Optional[str]): matplotlib colormap used to create heatmap.
                See https://matplotlib.org/3.3.0/tutorials/colors/colormaps.html
        """
        self.model = model
        # Run in eval mode.
        self.model.eval()
        self.target_layers = target_layers

        self.gradients = {}
        self.activations = {}
        self.method = method
        self.colormap = plt.get_cmap(colormap)
        self.data_mean = data_mean
        self.data_std = data_std
        self._register_hooks()

    def _register_single_hook(self, layer_name):
        """
        Register forward and backward hook to a layer, given layer_name,
        to obtain gradients and activations.
        Args:
            layer_name (str): name of the layer.
        """

        def get_gradients(module, grad_input, grad_output):
            self.gradients[layer_name] = grad_output[0].detach()

        def get_activations(module, input, output):
            self.activations[layer_name] = output.clone().detach()

        target_layer = get_layer(self.model, layer_name=layer_name)
        target_layer.register_forward_hook(get_activations)
        target_layer.register_full_backward_hook(get_gradients)

    def _register_hooks(self):
        """
        Register hooks to layers in `self.target_layers`.
        """
        for layer_name in self.target_layers:
            self._register_single_hook(layer_name=layer_name)

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
        input_clone = [inp.clone() for inp in inputs]
        preds = self.model(input_clone)

        # compute 'score' (aka the loss after softmax)
        if labels is None:
            score = torch.max(preds, dim=-1)[0]
        else:
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            score = torch.gather(preds, dim=1, index=labels)

        self.model.zero_grad()

        # sum the loss for the entire batch
        score = torch.sum(score)

        score.backward()  # Computes the gradient of current tensor w.r.t. graph leaves
        # Note that gradients can be implicitly created only for scalar outputs

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

        return localization_maps, preds

    def __call__(
        self, output_dir, inputs, video_indices, cfg, labels=None, alpha=0.5
    ):
        """
        Visualize the localization maps on their corresponding inputs as heatmap,
        using Grad-CAM.
        # TODO: fix docstring

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

        # retrieve heatmaps
        localization_maps, preds = self._calculate_localization_map(
            inputs, labels=labels
        )

        # save heatmaps
        heatmaps_frames_root_dir = os.path.join(
            output_dir,
            "heatmaps",
            cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.METHOD,
            "frames",
        )
        heatmap_volume_root_dir = os.path.join(
            cfg.OUTPUT_DIR,
            f"heatmaps/{cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.METHOD}/3d_volumes",
        )

        # iterate over channels (e.g. slow and fast)
        for channel_idx, localization_map in enumerate(localization_maps):
            if cfg.MODEL.ARCH == "slowfast":
                channel = "slow" if channel_idx == 0 else "fast"
            else:
                # since other visualization architectures don't necessarily
                # only have two input pathways, you have to add logic for it
                raise NotImplementedError(
                    "make subfolders for each pathway and put frames in correct subfolder for the specific visualization method"
                )

            # Convert (B, 1, T, H, W) to (B, T, H, W)
            localization_map = localization_map.squeeze(dim=1)
            if localization_map.device != torch.device("cpu"):
                localization_map = localization_map.cpu()

            # iterate over videos in batch
            for v, vid_idx in enumerate(video_indices.numpy()):
                map_to_save = localization_map.numpy()[v]
                visualization_path = os.path.join(
                    heatmaps_frames_root_dir, f"{vid_idx:06d}"
                )

                # iterate over frames and stitch them into the video

                for frame_idx in range(len(map_to_save)):
                    frame_map = map_to_save[frame_idx] * 255

                    one_based_frame_idx = frame_idx + 1
                    frame_name = f"{vid_idx:06d}_{one_based_frame_idx:06d}.jpg"
                    name = os.path.join(visualization_path, frame_name)

                    if cfg.MODEL.ARCH == "slowfast":
                        channel_folder = os.path.join(
                            visualization_path, channel
                        )
                        name = os.path.join(channel_folder, frame_name)
                        if not os.path.exists(channel_folder):
                            os.makedirs(channel_folder)
                    else:
                        # since other visualization architectures don't necessarily
                        # only have two input pathways, you have to add logic for it
                        raise NotImplementedError(
                            "make subfolders for each pathway and put frames in correct subfolder for the specific visualization method"
                        )

                    cv2.imwrite(name, frame_map)

                # generate 3d heatmap volumes for the video
                t_scale = 0.25
                s_scale = 1 / 8
                img_stack = load_heatmaps(
                    channel_folder, t_scale=t_scale, s_scale=s_scale
                )
                heatmap_fname = os.path.join(
                    heatmap_volume_root_dir,
                    f"{vid_idx:06d}/{channel}_{vid_idx:06d}.html",
                )
                plot_heatmap(
                    volume=img_stack,
                    fpath=heatmap_fname,
                    surface_count=8,
                    t_scale=t_scale,
                    s_scale=s_scale,
                    slider=True,
                )

            # prepare the results to be returned
            loc_map = localization_map  # (B, T, H, W)
            heatmap = self.colormap(loc_map.numpy())
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
            # TODO: turn curr_inp into a numpy array, add frame to video writer
            curr_inp = curr_inp.permute(0, 1, 4, 2, 3)

            result_ls.append(curr_inp)

        return result_ls, preds
