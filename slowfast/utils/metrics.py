#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Functions for computing metrics."""

import torch
import numpy as np
from slowfast.visualization.connected_components_utils import (
    load_heatmaps,
    generate_overlay,
)


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k.

    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.

    Returns:
        list containing top k errors as floats btwn 0 and 1
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """Computes the top-k accuracy for each k.

    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.

    Returns:
        list containing top k accuracies as floats btwn 0 and 1
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) for x in num_topks_correct]


def IOU_3D(vol1, vol2):
    """Compute the intersection over union for two 3d arrays with binarized
    volumes.

    Args:
        vol1, vol2: 3d arrays with the same shape. If it contains integers, all non-zero integers are treated as True or 1
    """
    assert vol1.shape == vol2.shape

    vol1 = vol1.astype(bool)
    vol2 = vol2.astype(bool)

    total_volume_1 = vol1.sum()
    total_volume_2 = vol2.sum()

    # compute intersection
    intersect = np.logical_and(vol1, vol2).sum()

    # compute union
    union = total_volume_1 + total_volume_2 - intersect

    # compute IOU
    iou = intersect / union
    return iou


def IOU_heatmap(heatmap_dir, trajectory_dir, thresh=0.2):
    """Compute the intersection over union for a heatmap with its ground-truth
    trajectory.

    Args:
        heatmap_dir (str): path to the directory containing all the heatmap
            frames for a single channel
        trajectory_dir (str): path to the directory containing all the target
            masks for the ground truth trajectory
        thresh (float): float between 0.0 and 1.0 as the percent of the
            maximum value in the heatmap at which the heatmap will be binarized
    """
    # load GT trajectory
    target_volume = load_heatmaps(
        trajectory_dir, t_scale=0.64, s_scale=0.64, mask=True
    )  # t_scale and s_scale are to rescale the time and space dimensions to match the rescaled video outputs

    # load heatmap
    heatmap_volume = load_heatmaps(heatmap_dir)  # shape (T, W, H)

    # binarize the heatmaps using a threshold
    max_intensity = heatmap_volume.max()

    heatmap_volume = np.where(heatmap_volume >= thresh * max_intensity, 1, 0)

    # compute 3D IOU
    iou = IOU_3D(target_volume, heatmap_volume)
    return iou
