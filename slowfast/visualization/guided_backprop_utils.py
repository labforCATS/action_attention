# modified from github.com/jacobgil/pytorch-grad

import os
import numpy as np
import torch
from torch.autograd import Function
import cv2

from pytorch_grad_cam.utils.find_layers import replace_all_layer_type_recursive
from pytorch_grad_cam.guided_backprop import (
    GuidedBackpropReLU,
    GuidedBackpropReLUasModule,
)

import slowfast.datasets.utils as data_utils
import slowfast.utils.logging as logging
import pdb

logger = logging.get_logger(__name__)


class GuidedBackpropReLUModel:
    def __init__(self, model, data_mean, data_std):
        self.model = model
        self.model.eval()

        self.data_mean = data_mean
        self.data_std = data_std

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, inputs, video_indices, cfg, labels=None):
        # TODO: question
        # deconvolution is relative to a target layer in the network. so why
        # dont we specify a target layer for this implementation of backprop?
        # it implicitly compute the gradient wrt the output score post softmax...
        # is that intended?

        replace_all_layer_type_recursive(
            self.model, torch.nn.ReLU, GuidedBackpropReLUasModule()
        )  # replaces any ReLU layers with GuidedBackpropReLUasModule

        # set tensors to require gradient computation
        # TODO: is this not alr set elsewhere? test commenting this out
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].requires_grad_(True)
        else:
            inputs = inputs.requires_grad_(True)

        scores_for_all_classes = self.forward(inputs)
        # we need to do a forward pass not bc we actually care about the
        # modified model outputs but bc its necessary for computing gradients
        logger.debug("scores_for_all_classes")
        logger.debug(scores_for_all_classes)

        if labels is None:
            # select the score corresponding to top predicted label
            score = torch.max(scores_for_all_classes, dim=-1)[0]
        else:
            # select the score corresponding to true label
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            score = torch.gather(scores_for_all_classes, dim=1, index=labels)

        logger.debug("score")
        logger.debug(score)

        # compute the gradient of the score wrt the input ??
        score = torch.sum(score)
        self.model.zero_grad()  # TODO: test do we need to set model gradients to zero? no difference so far if this is commented out
        # score.backward()  # Computes the gradient of current tensor w.r.t. graph leaves

        score.backward(
            retain_graph=True
        )  # TODO: test why do we need to retain_graph?
        # is it bc we need to access the grads attribute later?

        # retrieve gradients
        if isinstance(inputs, (list,)):
            grad_outputs = []
            for i in range(len(inputs)):
                grad_outputs.append(inputs[i].grad)
                # shape (B, C, T, H, W)
        else:
            grad_outputs = inputs.grad

        # presumably these gradients are of loss wrt inputs?
        # why didn't the gradcam code just do .grad? is it bc we have multiple
        # channels? or maybe bc they have multiple gradients and want to store
        # all of them? but isnt it alr stored in the .grad attr?

        # replace any GuidedBackpropReLUasModule layers with ReLU to revert
        # model back to its original form before guided backprop
        replace_all_layer_type_recursive(
            self.model, GuidedBackpropReLUasModule, torch.nn.ReLU()
        )

        # TODO save outputs as frames
        output_root_dir = os.path.join(
            cfg.OUTPUT_DIR, "heatmaps", "guided_backprop"
        )
        output_frames_root_dir = os.path.join(output_root_dir, "frames")
        output_volume_root_dir = os.path.join(output_root_dir, "3d_volumes")

        for channel_idx, output in enumerate(grad_outputs):
            if cfg.MODEL.ARCH == "slowfast":
                channel = "slow" if channel_idx == 0 else "fast"
            elif cfg.MODEL.ARCH == "i3d":
                channel = "rgb" if channel_idx == 0 else None
            else:
                # since other visualization architectures don't necessarily
                # only have two input pathways, you have to add logic for it
                raise NotImplementedError(
                    "make subfolders for each pathway and put frames in correct subfolder for the specific visualization method"
                )

            logger.debug("range of output gradients from min to max")
            logger.debug(output.min())
            logger.debug(output.max())
            pdb.set_trace()
            # revert normalization so that the outputs is in the range 0-255
            # first permute input from (B, C, T, H, W) to (B, T, H, W, C)
            output = output.permute(0, 2, 3, 4, 1)
            if output.device != torch.device("cpu"):
                output = output.cpu()
            output = data_utils.revert_tensor_normalize(
                output, self.data_mean, self.data_std
            )

            # iterate over videos in batch
            for v, vid_idx in enumerate(video_indices.numpy()):
                # heatmap to save
                vid_highlights = output.numpy()[v]  # shape (T, H, W, C)

                video_root_dir = os.path.join(
                    output_frames_root_dir, f"{vid_idx:06d}", channel
                )
                if not os.path.exists(video_root_dir):
                    os.makedirs(video_root_dir)

                # iterate over frames in video
                for frame_idx in range(len(vid_highlights)):
                    frame_highlights = (
                        vid_highlights[frame_idx] * 255
                    )  # shape (H, W, C)
                    logger.debug(f"min px vals: {frame_highlights.min()}")
                    logger.debug(f"max px vals: {frame_highlights.max()}")
                    if (
                        int(frame_highlights.min()) != 114
                        or int(frame_highlights.max()) != 114
                    ):
                        print("some actual values??")
                        pdb.set_trace()

                    one_based_frame_idx = frame_idx + 1
                    frame_name = (
                        f"{vid_idx.item():06d}_{one_based_frame_idx:06d}.jpg"
                    )
                    frame_path = os.path.join(video_root_dir, frame_name)
                    cv2.imwrite(frame_path, np.uint8(frame_highlights))
                    print(f"saved {frame_path}")

        # TODO save outputs as 3d scatter plots

        # pdb.set_trace()
        return grad_outputs
