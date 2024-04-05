#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Functions for computing metrics."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import copy
from scipy import stats
from slowfast.visualization.connected_components_utils import (
    load_heatmaps,
    generate_overlay,
)

# placeholder
METRIC_FUNCS = [
    "kl_div",
    "mse",
    "covariance",
    "pearson",
    "iou",
]


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


def IOU_3D(target_volume, heatmap_volume):
    """Compute the intersection over union for two 3d arrays with binarized
    volumes.

    Args:
        target_volume, heatmap_volume: 3d arrays with the same shape. If it contains integers,
            all non-zero integers are treated as True or 1
    Returns:
        calculated intersection over union between the volumes
    """
    assert target_volume.shape == heatmap_volume.shape

    target_vol = copy.deepcopy(target_volume)
    heatmap_vol = copy.deepcopy(heatmap_volume)

    target_vol = target_vol.astype(bool)
    heatmap_vol = heatmap_vol.astype(bool)

    total_volume_1 = target_vol.sum()
    total_volume_2 = heatmap_vol.sum()

    # compute intersection
    intersect = np.logical_and(target_vol, heatmap_vol).sum()

    # compute union
    union = total_volume_1 + total_volume_2 - intersect

    # compute IOU
    iou = intersect / union
    return iou


def IOU_frames(target_volume: np.ndarray, heatmap_volume: np.ndarray) -> np.ndarray:
    """Compute the intersection over union for each frame for two 3d arrays with
        binarized volumes

    Args:
        target_volume (np.ndarray): 3d array with shape (time, width, height). If it
            contains integers, all non-zero integers are treated as True or 1
        heatmap_volume (np.ndarray): 3d array with shape (time, width, height). If it
            contains integers, all non-zero integers are treated as True or 1

    Returns:
        np.ndarray: 1d srray of calculated IOUs for each frame
    """
    # shape (time, width, height) for both
    assert target_volume.shape == heatmap_volume.shape

    target_vol = copy.deepcopy(target_volume)
    heatmap_vol = copy.deepcopy(heatmap_volume)

    target_vol = target_vol.astype(bool)
    heatmap_vol = heatmap_vol.astype(bool)

    target_totals = target_vol.sum(axis=0)
    heatmap_totals = heatmap_vol.sum(axis=0)

    # compute intersection
    intersect = np.logical_and(target_vol, heatmap_vol).sum(axis=0)

    # compute union
    union = target_totals + heatmap_totals - intersect

    # compute IOU
    iou = intersect / union
    return iou


def convert_to_prob_dist(target_volume, heatmap_volume):
    """
    Takes in the ground truth heatmap and actual heatmap and converts both
    to probability distributions. the two volumes must have the same
    dimensions

    Args:
        target_volume: ground truth heatmap, numpy array of
            (time, width, height) format
        heatmap_volume: actual heatmap, numpy array of same dimensions
            as target_volume

    Returns:
        converted target and heatmap volumes
    """
    # make copies of both arrays so we can manipulate values
    target_vol = copy.deepcopy(target_volume)
    heatmap_vol = copy.deepcopy(heatmap_volume)

    # check that the shapes of the arrays are equivalent
    assert target_vol.shape == heatmap_vol.shape

    target_total_sum = np.sum(target_vol)
    heatmap_total_sum = np.sum(heatmap_vol)

    conv_target_dist = lambda val: val / target_total_sum
    conv_heatmap_dist = lambda val: val / heatmap_total_sum
    target_vol = conv_target_dist(target_vol)
    heatmap_vol = conv_heatmap_dist(heatmap_vol)

    return target_vol, heatmap_vol


def normalize(target_volume, heatmap_volume):
    """
    Takes in the ground truth heatmap and actual heatmap and normalizes them.
    the two volumes must have the same dimensions

    Args:
        target_volume: ground truth heatmap, numpy array of
            (time, width, height) format
        heatmap_volume: actual heatmap, numpy array of same dimensions
            as target_volume

    Returns:
        normalized target and heatmap volumes
    """
    # make copies of both arrays so we can manipulate values
    target_vol = copy.deepcopy(target_volume)
    heatmap_vol = copy.deepcopy(heatmap_volume)

    # check that the shapes of the arrays are equivalent
    assert target_vol.shape == heatmap_vol.shape

    target_max_val = np.max(target_vol)
    heatmap_max_val = np.max(heatmap_vol)

    conv_target_dist = lambda val: val / target_max_val
    conv_heatmap_dist = lambda val: val / heatmap_max_val
    target_vol = conv_target_dist(target_vol)
    heatmap_vol = conv_heatmap_dist(heatmap_vol)

    return target_vol, heatmap_vol


def KL_div(target_volume, heatmap_volume):
    """
    calculates the pointwise KL-divergence between the normalized activations
    of the target and heatmap volumes. the two volumes must have the
    same dimensions

    Args:
        target_volume: numpy array of (time, width, height) format,
            signifying ground truth heatmap
        heatmap_volume: numpy array of (time, width, height) format,
            signifying actual activation

    Returns:
        floating point representing calculated KL divergence
    """
    target_vol = copy.deepcopy(target_volume)
    heatmap_vol = copy.deepcopy(heatmap_volume)
    assert target_vol.shape == heatmap_vol.shape

    # normalize both arrays
    num_frame_pixels = len(target_vol[0]) * len(target_vol[0][0])
    adjust_vals = lambda val: val + (1 / num_frame_pixels)
    target_vol = adjust_vals(target_vol)
    heatmap_vol = adjust_vals(heatmap_vol)

    # convert value of each point into probability
    target_dist, heatmap_dist = convert_to_prob_dist(target_vol, heatmap_vol)

    flattened_target = np.ndarray.flatten(target_dist)
    flattened_heatmap = np.ndarray.flatten(heatmap_dist)
    kl_div = np.sum(
        np.where(
            flattened_heatmap != 0,
            flattened_heatmap * np.log(flattened_heatmap / flattened_target),
            0,
        )
    )
    return kl_div


def KL_div_frames(target_volume: np.ndarray, heatmap_volume: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        target_volume (np.ndarray): _description_
        heatmap_volume (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    target_vol = copy.deepcopy(target_volume)
    heatmap_vol = copy.deepcopy(heatmap_volume)

    # shape (time, width, height) for both, should be equal over the same video
    assert target_vol.shape == heatmap_vol.shape

    # width * height
    num_frame_pixels = len(target_vol[0]) * len(target_vol[0][0])
    adjust_vals = lambda val: val + (1 / num_frame_pixels)
    target_vol = adjust_vals(target_vol)
    heatmap_vol = adjust_vals(heatmap_vol)

    # convert value of each point into probability
    target_dist, heatmap_dist = convert_to_prob_dist(target_vol, heatmap_vol)

    # reshape from 3d array to 2d array, keep time dimension and flatten width/height
    target_2d = np.reshape(
        target_dist, (target_dist.shape[0], target_dist.shape[1] * target_dist.shape[2])
    )
    heatmap_2d = np.reshape(
        heatmap_dist,
        (heatmap_dist.shape[0], heatmap_dist.shape[1] * heatmap_dist.shape[2]),
    )
    kl_div = np.sum(
        np.where(
            heatmap_2d != 0,
            heatmap_2d * np.log(heatmap_2d / target_2d),
            0,
        ),
        axis=0,
    )
    return kl_div


def MSE(target_volume, heatmap_volume):
    """
    calculates the pixelwise mean squared error in normalized activation
    between two heatmap volumes. the two volumes must have the same dimensions

    Args:
        target_volume: 3d numpy array consisting of ground truth activations
        heatmap_volume: 3d numpy array consisting of actual activations

    Returns:
        mean squared error between the target and heatmap volumes
    """
    time = len(target_volume)
    width = len(target_volume[0])
    height = len(target_volume[0][0])
    num_terms = time * width * height

    target_vol, heatmap_vol = normalize(target_volume, heatmap_volume)

    result = 0

    for t in range(time):
        for w in range(width):
            for h in range(height):
                activation_diff = (heatmap_vol[t][w][h] - target_vol[t][w][h]) ** 2
                result += activation_diff
    result = result / num_terms
    return result


def MSE_frames(target_volume: np.ndarray, heatmap_volume: np.ndarray) -> np.ndarray:
    """Calculates the pixelwise mean squared error by frame in normalized activation
    between two heatmap videos of the same dimension

    Args:
        target_volume (np.ndarray): Numpy array with shape (time, width, height) of
            ground truth activations
        heatmap_volume (np.ndarray): Numpy array with shape (time, width, height) of
            actual activations

    Returns:
        np.ndarray: Numpy array with shape (time) containing the mean squared error
            between each frame of the target and heatmap volumes
    """
    # shape (time, width, height) for both, should be equal over the same video
    assert target_volume.shape == heatmap_volume.shape
    time = target_volume.shape[0]
    width = target_volume.shape[1]
    height = target_volume.shape[2]
    num_terms_per_frame = width * height

    target_vol, heatmap_vol = normalize(target_volume, heatmap_volume)

    mses = np.zeros((time))

    for t in range(time):
        for w in range(width):
            for h in range(height):
                activation_diff = (heatmap_vol[t][w][h] - target_vol[t][w][h]) ** 2
                mses[t] += activation_diff
    mses = mses / num_terms_per_frame
    return mses


def covariance(target_volume, heatmap_volume):
    """
    calculates the pixelwise covariance between the normalized activations of
    the target volume and observed activations. the two volumes must have the
    same dimensions

    Args:
        target_volume: numpy array of (time, width, height) format,
            signifying ground truth heatmap
        heatmap_volume: numpy array of (time, width, height) format,
            signifying actual activation

    Returns:
        floating point representing pixelwise covariance
    """
    time = len(target_volume)
    width = len(target_volume[0])
    height = len(target_volume[0][0])
    num_terms = time * width * height

    target_vol, heatmap_vol = normalize(target_volume, heatmap_volume)

    target_vol_mean = np.mean(target_vol)
    heatmap_vol_mean = np.mean(heatmap_vol)

    summation = 0
    for t in range(time):
        for w in range(width):
            for h in range(height):
                heatmap_diff = heatmap_vol[t][w][h] - heatmap_vol_mean
                target_vol_diff = target_vol[t][w][h] - target_vol_mean
                summation += heatmap_diff * target_vol_diff
    covariance = summation / (num_terms - 1)
    return covariance


def covariance_frames(
    target_volume: np.ndarray, heatmap_volume: np.ndarray
) -> np.ndarray:
    """Calculates the frame by frame pixelwise covariance between the normalized
    activations of the target volume and observed activations.

    Args:
        target_volume: numpy array of (time, width, height) format,
            signifying ground truth heatmap
        heatmap_volume: numpy array of (time, width, height) format,
            signifying actual activation

    Returns:
        np.ndarray: numpy array of shape (time) with entries representing the pixelwise
            covariance for each frame
    """
    assert target_volume.shape == heatmap_volume.shape

    time = target_volume.shape[0]
    width = target_volume.shape[1]
    height = target_volume.shape[2]
    num_terms_per_frame = width * height

    target_vol, heatmap_vol = normalize(target_volume, heatmap_volume)

    target_vol_means = np.mean(target_vol, axis=0)
    heatmap_vol_means = np.mean(heatmap_vol, axis=0)

    summation = 0
    sums = np.zeros((time))
    for t in range(time):
        for w in range(width):
            for h in range(height):
                heatmap_diff = heatmap_vol[t][w][h] - heatmap_vol_means[t]
                target_vol_diff = target_vol[t][w][h] - target_vol_means[t]
                sums[t] += heatmap_diff * target_vol_diff
    covariances = sums / (num_terms_per_frame - 1)
    return covariances


def pearson_correlation(target_volume, heatmap_volume):
    """
    calculates the pixelwise pearson correlation between the normalized
    activations of the target volume and observed activations. the two
    volumes must have the same dimensions

    Args:
        target_volume: numpy array of (time, width, height) format,
            signifying ground truth heatmap
        heatmap_volume: numpy array of (time, width, height) format,
            signifying actual activation

    Returns:
        floating point representing pixelwise pearson correlation
    """
    cov = covariance(target_volume, heatmap_volume)
    target_std = np.std(target_volume)
    heatmap_std = np.std(heatmap_volume)
    pearson = cov / (target_std * heatmap_std)
    return pearson


def pearson_correlation_frames(
    target_volume: np.ndarray, heatmap_volume: np.ndarray
) -> np.ndarray:
    """Calculates the pixelwise pearson corellation by frame between the normalized
    activations of the target volume and the observed activations.

    Args:
        target_volume: numpy array of (time, width, height) format,
            signifying ground truth heatmap
        heatmap_volume: numpy array of (time, width, height) format,
            signifying actual activation

    Returns:
        np.ndarray: numpy array of shape (time) with entries representing the pixelwise
            pearson correlation
    """
    covs = covariance_frames(target_volume, heatmap_volume)
    target_stds = np.std(target_volume, axis=0)
    heatmap_stds = np.std(heatmap_volume, axis=0)
    pearsons = covs / (target_stds * heatmap_stds)
    return pearsons


def heatmap_metrics(
    heatmap_dir: str,
    trajectory_dir: str,
    metrics: List[str],
    pathway: str,
    thresh: float = 0.2,
) -> dict:
    """Compute the values for a list of given metric functions on a heatmap
        with its ground-truth trajectory.

    Args:
        heatmap_dir (str): path to the directory containing all the heatmap
            frames for a single channel
        trajectory_dir (str): path to the directory containing all the target
            masks for the ground truth trajectory
        metrics (List[str]): list of metric function names. must be a valid
            element of METRIC_FUNCS
        pathway (string): indicates input pathway
        thresh (float, optional): float between 0.0 and 1.0 as the percent of the
            maximum value in the heatmap at which the heatmap will be binarized

    Returns:
        dict: dictionary containing the metric names and values, where the value is
        a single element list
    """
    assert set(metrics).issubset(set(METRIC_FUNCS))

    # load ground truth trajectory
    if pathway == "rgb":
        target_volume = load_heatmaps(
            trajectory_dir, t_scale=0.64, s_scale=0.64, mask=True
        )  # t_scale and s_scale are to rescale the time and space dimensions to match the rescaled video outputs
    elif pathway == "slow":
        # should be 0.16 to get 50 frames to 8
        target_volume = load_heatmaps(
            trajectory_dir, t_scale=0.16, s_scale=0.64, mask=True
        )  # t_scale and s_scale are to rescale the time and space dimensions to match the rescaled video outputs
    elif pathway == "fast":
        # tscale fine here
        target_volume = load_heatmaps(
            trajectory_dir, t_scale=0.64, s_scale=0.64, mask=True
        )  # t_scale and s_scale are to rescale the time and space dimensions to match the rescaled video outputs
    else:
        raise NotImplementedError("Add in logic to retrieve the correct target volume")

    # load activation heatmap (heatmap volume)
    heatmap_volume = load_heatmaps(heatmap_dir)  # shape (time, width, height)

    # create binarized version of heatmap volume
    max_intensity = heatmap_volume.max()
    binarized_heatmap = copy.deepcopy(heatmap_volume)
    binarized_heatmap = np.where(binarized_heatmap >= thresh * max_intensity, 1, 0)

    metric_results = {}

    # iterate through list of metrics, computing the values
    for metric_name in metrics:
        if metric_name == "kl_div":
            result = KL_div(target_volume, heatmap_volume)
        elif metric_name == "mse":
            result = MSE(target_volume, heatmap_volume)
        elif metric_name == "covariance":
            result = covariance(target_volume, heatmap_volume)
        elif metric_name == "pearson":
            result = pearson_correlation(target_volume, heatmap_volume)
        elif metric_name == "iou":
            result = IOU_3D(target_volume, binarized_heatmap)
        else:
            raise NotImplementedError(
                "Unrecognized metric; implement metric and add logic"
            )

        metric_results[metric_name] = result

    return metric_results
