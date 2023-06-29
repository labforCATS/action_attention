#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import cv2
from sklearn.metrics import confusion_matrix

import slowfast.utils.logging as logging
import slowfast.datasets.utils as data_utils
from slowfast.datasets.utils import pack_pathway_output, tensor_normalize

logger = logging.get_logger(__name__)


def get_confusion_matrix(preds, labels, num_classes, normalize="true"):
    """
    Calculate confusion matrix on the provided preds and labels.
    Args:
        preds (tensor or lists of tensors): predictions. Each tensor is in
            in the shape of (n_batch, num_classes). Tensor(s) must be on CPU.
        labels (tensor or lists of tensors): corresponding labels. Each tensor is
            in the shape of either (n_batch,) or (n_batch, num_classes).
        num_classes (int): number of classes. Tensor(s) must be on CPU.
        normalize (Optional[str]) : {‘true’, ‘pred’, ‘all’}, default="true"
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix
            will not be normalized.
    Returns:
        cmtx (ndarray): confusion matrix of size (num_classes x num_classes)
    """
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    # Get the predicted class indices for examples.
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)), normalize=normalize
    )
    return cmtx


def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=None):
    """
    A function to create a colored and labeled confusion matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        class_names (Optional[list of strs]): a list of class names.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure


def plot_topk_histogram(tag, array, k=10, class_names=None, figsize=None):
    """
    Plot histogram of top-k value from the given array.
    Args:
        tag (str): histogram title.
        array (tensor): a tensor to draw top k value from.
        k (int): number of top values to draw from array.
            Defaut to 10.
        class_names (list of strings, optional):
            a list of names for values in array.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].
    Returns:
        fig (matplotlib figure): a matplotlib figure of the histogram.
    """
    val, ind = torch.topk(array, k)

    fig = plt.Figure(figsize=figsize, facecolor="w", edgecolor="k")

    ax = fig.add_subplot(1, 1, 1)

    if class_names is None:
        class_names = [str(i) for i in ind]
    else:
        class_names = [class_names[i] for i in ind]

    tick_marks = np.arange(k)
    width = 0.75
    ax.bar(
        tick_marks,
        val,
        width,
        color="orange",
        tick_label=class_names,
        edgecolor="w",
        linewidth=1,
    )

    ax.set_xlabel("Candidates")
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=-45, ha="center")
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    y_tick = np.linspace(0, 1, num=10)
    ax.set_ylabel("Frequency")
    ax.set_yticks(y_tick)
    y_labels = [format(i, ".1f") for i in y_tick]
    ax.set_yticklabels(y_labels, ha="center")

    for i, v in enumerate(val.numpy()):
        ax.text(
            i - 0.1,
            v + 0.03,
            format(v, ".2f"),
            color="orange",
            fontweight="bold",
        )

    ax.set_title(tag)

    fig.set_tight_layout(True)

    return fig


class GetWeightAndActivation:
    """
    A class used to get weights and activations from specified layers from a Pytorch model.
    """

    def __init__(self, model, layers):
        """
        Args:
            model (nn.Module): the model containing layers to obtain weights and activations from.
            layers (list of strings): a list of layer names to obtain weights and activations from.
                Names are hierarchical, separated by /. For example, If a layer follow a path
                "s1" ---> "pathway0_stem" ---> "conv", the layer path is "s1/pathway0_stem/conv".
        """
        self.model = model
        self.hooks = {}
        self.layers_names = layers
        # eval mode
        self.model.eval()
        self._register_hooks()

    def _get_layer(self, layer_name):
        """
        Return a layer (nn.Module Object) given a hierarchical layer name, separated by /.
        Args:
            layer_name (str): the name of the layer.
        """
        layer_ls = layer_name.split("/")
        prev_module = self.model
        for layer in layer_ls:
            prev_module = prev_module._modules[layer]

        return prev_module

    def _register_single_hook(self, layer_name):
        """
        Register hook to a layer, given layer_name, to obtain activations.
        Args:
            layer_name (str): name of the layer.
        """

        def hook_fn(module, input, output):
            self.hooks[layer_name] = output.clone().detach()

        layer = get_layer(self.model, layer_name)
        layer.register_forward_hook(hook_fn)

    def _register_hooks(self):
        """
        Register hooks to layers in `self.layers_names`.
        """
        for layer_name in self.layers_names:
            self._register_single_hook(layer_name)

    def get_activations(self, input, bboxes=None):
        """
        Obtain all activations from layers that we register hooks for.
        Args:
            input (tensors, list of tensors): the model input.
            bboxes (Optional): Bouding boxes data that might be required
                by the model.
        Returns:
            activation_dict (Python dictionary): a dictionary of the pair
                {layer_name: list of activations}, where activations are outputs returned
                by the layer.
        """
        input_clone = [inp.clone() for inp in input]
        if bboxes is not None:
            preds = self.model(input_clone, bboxes)
        else:
            preds = self.model(input_clone)

        activation_dict = {}
        for layer_name, hook in self.hooks.items():
            # list of activations for each instance.
            activation_dict[layer_name] = hook

        return activation_dict, preds

    def get_weights(self):
        """
        Returns weights from registered layers.
        Returns:
            weights (Python dictionary): a dictionary of the pair
            {layer_name: weight}, where weight is the weight tensor.
        """
        weights = {}
        for layer in self.layers_names:
            cur_layer = get_layer(self.model, layer)
            if hasattr(cur_layer, "weight"):
                weights[layer] = cur_layer.weight.clone().detach()
            else:
                logger.error("Layer {} does not have weight attribute.".format(layer))
        return weights


def get_indexing(string):
    """
    Parse numpy-like fancy indexing from a string.
    Args:
        string (str): string represent the indices to take
            a subset of from array. Indices for each dimension
            are separated by `,`; indices for different dimensions
            are separated by `;`.
            e.g.: For a numpy array `arr` of shape (3,3,3), the string "1,2;1,2"
            means taking the sub-array `arr[[1,2], [1,2]]
    Returns:
        final_indexing (tuple): the parsed indexing.
    """
    index_ls = string.strip().split(";")
    final_indexing = []
    for index in index_ls:
        index_single_dim = index.split(",")
        index_single_dim = [int(i) for i in index_single_dim]
        final_indexing.append(index_single_dim)

    return tuple(final_indexing)


def process_layer_index_data(layer_ls, layer_name_prefix=""):
    """
    Extract layer names and numpy-like fancy indexing from a string.
    Args:
        layer_ls (list of strs): list of strings containing data about layer names
            and their indexing. For each string, layer name and indexing is separated by whitespaces.
            e.g.: [layer1 1,2;2, layer2, layer3 150;3,4]
        layer_name_prefix (Optional[str]): prefix to be added to each layer name.
    Returns:
        layer_name (list of strings): a list of layer names.
        indexing_dict (Python dict): a dictionary of the pair
            {one_layer_name: indexing_for_that_layer}
    """

    layer_name, indexing_dict = [], {}
    for layer in layer_ls:
        ls = layer.split()
        name = layer_name_prefix + ls[0]
        layer_name.append(name)
        if len(ls) == 2:
            indexing_dict[name] = get_indexing(ls[1])
        else:
            indexing_dict[name] = ()
    return layer_name, indexing_dict


def process_cv2_inputs(frames, cfg):
    """
    Normalize and prepare inputs as a list of tensors. Each tensor
    correspond to a unique pathway.
    Args:
        frames (list of array): list of input images (correspond to one clip) in range [0, 255].
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    inputs = torch.from_numpy(np.array(frames)).float() / 255
    inputs = tensor_normalize(inputs, cfg.DATA.MEAN, cfg.DATA.STD)
    # T H W C -> C T H W.
    inputs = inputs.permute(3, 0, 1, 2)
    # Sample frames for num_frames specified.
    index = torch.linspace(0, inputs.shape[1] - 1, cfg.DATA.NUM_FRAMES).long()
    inputs = torch.index_select(inputs, 1, index)
    inputs = pack_pathway_output(cfg, inputs)
    inputs = [inp.unsqueeze(0) for inp in inputs]
    return inputs


def get_layer(model, layer_name):
    """
    Return the targeted layer (nn.Module Object) given a hierarchical layer name,
    separated by /.
    Args:
        model (model): model to get layers from.
        layer_name (str): name of the layer.
    Returns:
        prev_module (nn.Module): the layer from the model with `layer_name` name.
    """
    layer_ls = layer_name.split("/")
    prev_module = model
    # print(model)
    for layer in layer_ls:
        prev_module = prev_module._modules[layer]

    return prev_module


class TaskInfo:
    def __init__(self):
        self.frames = None
        self.id = -1
        self.bboxes = None
        self.action_preds = None
        self.num_buffer_frames = 0
        self.img_height = -1
        self.img_width = -1
        self.crop_size = -1
        self.clip_vis_size = -1

    def add_frames(self, idx, frames):
        """
        Add the clip and corresponding id.
        Args:
            idx (int): the current index of the clip.
            frames (list[ndarray]): list of images in "BGR" format.
        """
        self.frames = frames
        self.id = idx

    def add_bboxes(self, bboxes):
        """
        Add correspondding bounding boxes.
        """
        self.bboxes = bboxes

    def add_action_preds(self, preds):
        """
        Add the corresponding action predictions.
        """
        self.action_preds = preds


def save_inputs(data_loader, cfg, mode, save_video = False):
    """
    Saves the frames of the inputs to the model as a .jpg

    Inputs:
        data_loader: data loader for the dataset
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        mode(string): a value of either "test", "train", or "val"
        save_video (boolean): whether to save model inputs as a video
    Outputs:
        saves frames (and video, if applicable) to the output folder specified
        in config file
    """
    if (mode != "train" and mode != "test" and mode != "val"):
        raise ValueError("mode must be 'train' or 'test' or 'val'")
    output_folder_path = os.path.join(cfg.OUTPUT_DIR, mode)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # go through each batch passed to the model
    if mode == "test":
        video_indices = np.empty(cfg.TEST.BATCH_SIZE)
    else:
        video_indices = np.empty(cfg.TRAIN.BATCH_SIZE)
    for batch, (inputs, labels, index, time, meta) in enumerate(data_loader):
        print("we are at batch number:", batch)
        video_indices = index.numpy()
        batch_size = 0
        if mode == "test":
            batch_size = cfg.TEST.BATCH_SIZE
        else:
            batch_size = cfg.TRAIN.BATCH_SIZE

        # go through each image in the batch
        for batch_index in range(batch_size):
            video_index = video_indices[batch_index]
            
            # make folders to store output images
            slow_folder = os.path.join(output_folder_path, str(video_index), "slow")
            fast_folder = os.path.join(output_folder_path, str(video_index), "fast")
            if not os.path.exists(slow_folder):
                os.makedirs(slow_folder)
            if not os.path.exists(fast_folder):
                os.makedirs(fast_folder)

            # isolate the current slow and fast pathways
            curr_slow_tensor = inputs[0][batch_index, :, :, :, :]
            curr_slow_tensor = torch.unsqueeze(curr_slow_tensor, dim=0)
            curr_fast_tensor = inputs[1][batch_index, :, :, :, :]
            curr_fast_tensor = torch.unsqueeze(curr_fast_tensor, dim=0)
            # revert tensor normalization
            curr_slow_tensor = curr_slow_tensor.permute(0, 2, 3, 4, 1)
            curr_slow_tensor = data_utils.revert_tensor_normalize(
                curr_slow_tensor, cfg.DATA.MEAN, cfg.DATA.STD
            )
            curr_fast_tensor = curr_fast_tensor.permute(0, 2, 3, 4, 1)
            curr_fast_tensor = data_utils.revert_tensor_normalize(
                curr_fast_tensor, cfg.DATA.MEAN, cfg.DATA.STD
            )

            num_slow_frame = curr_slow_tensor.size(dim=1)
            num_fast_frame = curr_fast_tensor.size(dim=1)

            
            # save all slow frames as a jpg
            for slow_frame in range(num_slow_frame):
                # move the tensor to cpu if needed
                if curr_slow_tensor.device != torch.device("cpu"):
                    curr_slow_tensor = curr_slow_tensor.to("cpu")
                # isolate the individual frames from the tensor
                curr_slow_tensor_image = curr_slow_tensor[0, slow_frame, :, :, :].numpy() * 255
                one_based_slow_frame = slow_frame + 1
                slow_name = f"{video_index:03d}_{one_based_slow_frame:06d}.jpg"
                slow_name = os.path.join(slow_folder, slow_name)
                cv2.imwrite(slow_name, curr_slow_tensor_image)

            # save all fast frames as a jpg
            for fast_frame in range(num_fast_frame):
                # move the tensor to cpu if needed
                if curr_fast_tensor.device != torch.device("cpu"):
                    curr_fast_tensor = curr_fast_tensor.to("cpu")
                # isolate the individual frames from the tensor
                curr_fast_tensor_image = curr_fast_tensor[0, fast_frame, :, :, :].numpy() * 255
                one_based_fast_frame = fast_frame + 1
                fast_name = f"{video_index:03d}_{one_based_fast_frame:06d}.jpg"
                fast_name = os.path.join(fast_folder, fast_name)
                cv2.imwrite(fast_name, curr_fast_tensor_image)
    
    # save the input frames as a video
    if save_video:
        for video_index in video_indices:
            video_dir = os.path.join(cfg.OUTPUT_DIR, mode, str(video_index))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            pathways = ["slow", "fast"]
            for pathway in pathways:
                video_name = os.path.join(video_dir, f"{pathway}_{video_index:06d}.mp4")
                if mode == "test":
                    video = cv2.VideoWriter(video_name, fourcc, 25, (cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE))
                else:
                    video = cv2.VideoWriter(video_name, fourcc, 25, (cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE))
                pathway_dir = os.path.join(video_dir, pathway)
                for image_path in os.listdir(pathway_dir):
                    if image_path.endswith(".jpg"):
                        image = cv2.imread(image_path)
                        video.write(image)
                cv2.destroyAllWindows()
                video.release()
