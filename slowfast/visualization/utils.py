#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pdb
import json

import slowfast.utils.logging as logging
import slowfast.datasets.utils as data_utils
from slowfast.datasets.utils import pack_pathway_output, tensor_normalize
from slowfast.datasets import loader

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
                logger.error(
                    "Layer {} does not have weight attribute.".format(layer)
                )
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


def save_inputs(data_loader, cfg, mode):
    """
    Saves the frames and/or videos of the inputs to the model as a .jpg/.mp4

    Inputs:
        data_loader: data loader for the dataset
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        mode(string): a value of either "test", "train", or "val"
    Outputs:
        saves frames and/or videos to the output folder specified
        in config file
    """
    if not (
        cfg.DATA_LOADER.INSPECT.SAVE_FRAMES
        or cfg.DATA_LOADER.INSPECT.SAVE_VIDEO
    ):
        return
    if mode != "train" and mode != "test" and mode != "val":
        raise ValueError("mode must be 'train' or 'test' or 'val'")

    logger.info(f"saving inputs for {mode} mode")

    output_folder_path = os.path.join(cfg.OUTPUT_DIR, f"{mode}_loader_samples")

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    if cfg.DATA_LOADER.INSPECT.SHUFFLE and mode == "train":
        # the val and train loaders have a SequentialSampler which is not
        # supported by loader.shuffle_dataset
        loader.shuffle_dataset(data_loader, 0)

    vid_count = 0

    # go through each batch passed to the model
    for batch, (inputs, labels, index, time, meta) in enumerate(data_loader):
        if vid_count >= cfg.DATA_LOADER.INSPECT.SAVE_SEQ_COUNT:
            break
            # otherwise continue, and break the loop over individual videos if needed

        video_indices = index.numpy()

        # go through each video in the batch
        for i in range(len(labels)):
            if vid_count >= cfg.DATA_LOADER.INSPECT.SAVE_SEQ_COUNT:
                break
            vid_count += 1

            video_index = video_indices[i]

            pathways = ["slow", "fast"]
            for pathway_idx, pathway in enumerate(pathways):
                if cfg.DATA_LOADER.INSPECT.SAVE_FRAMES:
                    # TODO: thereotically could move this down
                    # make folders to store output images
                    pathway_frames_folder = os.path.join(
                        output_folder_path,
                        "frames",
                        f"{video_index:06d}",
                        pathway,
                    )
                    if not os.path.exists(pathway_frames_folder):
                        os.makedirs(pathway_frames_folder)

                # isolate the pathway data
                pathway_tensor = inputs[pathway_idx][i, :, :, :, :]
                pathway_tensor = torch.unsqueeze(pathway_tensor, dim=0)
                # revert tensor normalization
                # permute from (B, C, T, H, W) to (B, T, H, W, C)
                pathway_tensor = pathway_tensor.permute(0, 2, 3, 4, 1)
                pathway_tensor = data_utils.revert_tensor_normalize(
                    pathway_tensor, cfg.DATA.MEAN, cfg.DATA.STD
                )

                num_frames = pathway_tensor.size(dim=1)

                # initialize variables for saving the video
                if cfg.DATA_LOADER.INSPECT.SAVE_VIDEO:
                    video_dir = os.path.join(
                        output_folder_path, "videos", f"{video_index:06d}"
                    )
                    if not os.path.isdir(video_dir):
                        os.makedirs(video_dir)
                    video_name = os.path.join(
                        video_dir, f"{pathway}_{video_index:06d}.mp4"
                    )
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    crop_size = (
                        (
                            cfg.DATA.TEST_CROP_SIZE,
                            cfg.DATA.TEST_CROP_SIZE,
                        )
                        if mode == "test"
                        else (
                            cfg.DATA.TRAIN_CROP_SIZE,
                            cfg.DATA.TRAIN_CROP_SIZE,
                        )
                    )
                    video = cv2.VideoWriter(
                        video_name,
                        fourcc,
                        25,
                        crop_size,
                    )

                # iterate over frames
                for frame in range(num_frames):
                    # move the tensor to cpu if needed
                    if pathway_tensor.device != torch.device("cpu"):
                        pathway_tensor = pathway_tensor.to("cpu")
                    # isolate the individual frames from the tensor (B, T, H, W, C)
                    pathway_np_image = (
                        pathway_tensor[0, frame, :, :, :].numpy() * 255
                    )

                    # save frame, if applicable
                    if cfg.DATA_LOADER.INSPECT.SAVE_FRAMES:
                        one_based_frame = frame + 1
                        frame_fname = (
                            f"{video_index:06d}_{one_based_frame:06d}.jpg"
                        )
                        frame_path = os.path.join(
                            pathway_frames_folder, frame_fname
                        )
                        cv2.imwrite(frame_path, pathway_np_image)

                    # save to videowriter, if applicable
                    if cfg.DATA_LOADER.INSPECT.SAVE_VIDEO:
                        video.write(pathway_np_image.astype("uint8"))

                if cfg.DATA_LOADER.INSPECT.SAVE_VIDEO:
                    cv2.destroyAllWindows()
                    video.release()


def plot_train_val_curves(cfg):
    """Plots the training and validation accuracy and loss over the epochs.

    Data is read from the auto-generated json_stats.log; Plots are saved to the
    configured outputs directory and will be overridden and updated each epoch.
    """
    try:
        save_path = os.path.join(cfg.OUTPUT_DIR, "train_val_curves.jpg")

        # retrieve data from json_stats.log
        stats_path = os.path.join(cfg.OUTPUT_DIR, "json_stats.log")
        train_epochs = []
        train_losses = []
        train_accs = []
        val_epochs = []
        val_losses = []
        val_accs = []

        with open(stats_path, "r") as f:
            for line in f.readlines():
                dict_str = line.rstrip().removeprefix("json_stats: ")
                stats_dict = json.loads(dict_str)
                mode = stats_dict["_type"].removesuffix("_epoch")
                epoch = int(stats_dict["epoch"].split("/")[0])
                loss = float(stats_dict["loss"])
                acc = 1.0 - float(stats_dict["top1_err"])

                if mode == "train":
                    train_epochs.append(epoch)
                    train_losses.append(loss)
                    train_accs.append(acc)
                elif mode == "val":
                    val_epochs.append(epoch)
                    val_losses.append(loss)
                    val_accs.append(acc)
                else:
                    raise ValueError(
                        f"inappropriate _type ({mode}) in json_stats.log"
                    )

        # generate plot
        fig, axs = plt.subplots(1, 2)

        # plot losses
        axs[0].plot(train_epochs, train_losses, "-o", label="train")
        axs[0].plot(val_epochs, val_losses, "-o", label="val")
        axs[0].set_ylim(bottom=0)

        # plot accuracies
        axs[1].plot(train_epochs, train_accs, "-o", label="train")
        axs[1].plot(val_epochs, val_accs, "-o", label="val")
        axs[1].set_ylim(top=1)

        # add labels, titles, etc
        axs[0].set_xlabel("Epochs")
        axs[1].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss")
        axs[1].set_ylabel("Accuracy")
        axs[0].set_title("Loss over training")
        axs[1].set_title("Accuracy over training")

        axs[0].legend()
        axs[1].legend()

        plt.tight_layout()

        plt.savefig(save_path)
    except Exception as e:
        logger.warning(f"Failed to plot train/val curves with Exception {e}")
