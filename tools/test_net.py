#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import pdb
import cv2
import json
import copy

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
import slowfast.datasets.utils as data_utils
import pandas as pd
from slowfast.datasets import loader
from slowfast.visualization.utils import save_inputs
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter
from slowfast.utils.metrics import heatmap_metrics
from scripts import output_idx_to_input, get_exp_and_root_dir, class_int_to_string


logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.

    NOTE: the multi-view parameters are configured in the data loader class for a particular dataset. The above description is true for most datasets except the synthetic dataset of generated videos. For the synthetic videos, the test_loader is configured to return entire videos, not subsampled ones.

    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.

            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            metadata = metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()


            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time)
            train_labels = (
                model.module.train_labels
                if hasattr(model, "module")
                else model.train_labels
            )
            yd, yi = model(inputs, video_idx, time)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)
        else:
            # Perform the forward pass.
            preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(preds.detach(), labels.detach(), video_idx.detach())
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
            new_preds = list(all_preds)
            predictions = []
            for pred in range(len(new_preds)):
                new_preds[pred] = list(new_preds[pred])
                num = max(new_preds[pred])
                p = new_preds[pred].index(num)
                predictions += [p]
            #     for p in range(len(new_preds[pred])):
            #         new_preds[pred][p] = float(new_preds[pred][p] - num)
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info("Successfully saved prediction results to {}".format(save_path))
            pdb.set_trace()

    test_meter.finalize_metrics()
    return test_meter


@torch.no_grad()
def run_heatmap_metrics(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Run metric computations over existing heatmaps. This will retrieve predictions for each video and compute the metrics declared in the config.

    NOTE: this is only designed to work for the SyntheticMotion dataset for now as it uses the full testing videos and does not subsample multiple testing videos like the standard testing loader/pipeline are configured to do.

    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # get the heatmap root dir and json containing the list of test data 
    heatmaps_root_dir = os.path.join(
        cfg.OUTPUT_DIR,
        "heatmaps",
        cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.METHOD,
        "post_softmax"
        if cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.POST_SOFTMAX
        else "pre_softmax",
    )
    input_json_path = os.path.join(
        cfg.DATA.PATH_TO_DATA_DIR, "synthetic_motion_test.json"
    )
    label_json_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, "synthetic_motion_labels.json")

    # get details of the experiment, including experiment number and nonlocal
    exp, experiment_root_dir = get_exp_and_root_dir(cfg.DATA.PATH_TO_DATA_DIR)

    print("heatmaps root directory:", heatmaps_root_dir)
    print("input json path:", input_json_path)
    print("experiment:", exp)

    # Enable eval mode.
    model.eval()
    # test_meter.iter_tic()

    # get experiment properties 
    metrics = cfg.METRICS.FUNCS

    nonlocal_location = np.array(cfg.NONLOCAL.LOCATION, dtype = "object")
    nonlocal_location = nonlocal_location.flatten("F")
    non_empty_elems = [x for x in nonlocal_location if len(x) != 0] 
    # TODO ^ is this able to handle nested lists of arbitrary depth? 
    print("cfg.NONLOCAL.LOCATION", cfg.NONLOCAL.LOCATION)
    print("nonlocal_location after flatten", nonlocal_location)
    print("non_empty_elems", non_empty_elems)

    if len(non_empty_elems) != 0:
        is_nonlocal = True
    else:
        is_nonlocal = False


    # Create results dictionary (which will later be converted to a dataframe
    # and exported to csv)
    # the keys will be the various properties, e.g. video index, label, experiment parameters, etc. and the values are lists, where the ith value of each list corresponds to the ith video we iterate over, aka the ith row in the dataframe after we convert it 
    dataset_size = len(test_loader.dataset)
    if cfg.MODEL.ARCH == "slowfast":
        num_channels = 2
    elif cfg.MODEL.ARCH == "i3d":
        num_channels = 1
    else:
        raise NotImplementedError("add in number of channels for architecture")

    data_dict = {
        # experiment params are the same for all videos in the dataset
        "experiment": [exp] * dataset_size * num_channels, 
        "model": [cfg.MODEL.ARCH] * dataset_size * num_channels,
        "nonlocal": [is_nonlocal] * dataset_size * num_channels,
        "gradcam_variant": (
            [cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.METHOD] * dataset_size * num_channels),
        "post_softmax": (
            [cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.POST_SOFTMAX] * dataset_size * num_channels
        ),
        # video features
        "input_vid_idx": [],
        "channel": [],
        "label": [],
        "label_numeric": [],
        "pred": [],
        "pred_numeric": [],
        "correct": [],
    }
    for metric in metrics:
        data_dict[metric] = []
    print("beginning iterations through testing loader")
    for cur_iter, (inputs, labels, video_ids, time, meta) in enumerate(test_loader):
        print("current iteration", cur_iter)
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_ids = video_ids.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        # test_meter.data_toc()

        # Perform the forward pass.
        preds = model(inputs)

        # log results for each heatmap to the results dict
        # iterate over each video in the batch
        for label, pred, output_vid_idx in zip(labels, preds, video_ids):
            # retrieve the input vid index
            target_class, input_vid_idx = output_idx_to_input(input_json_path, output_vid_idx)

            # other option instead of using target_class is to convert the label (which is the integer encoding of the class) to the string. but that's annoying 

            # get the actual prediction (preds has the cross entropy scores for all classes) 
            pred_numeric = torch.flatten(torch.argmax(pred, dim=-1)).item()
            pred_class = class_int_to_string(pred_numeric, label_json_path)

            if cfg.MODEL.ARCH not in ["slowfast", "i3d"]:
                raise NotImplementedError("add in logic retrieving channels for this architecture")
            
            if len(inputs) == 1:
                channel_list = ["rgb"]
            else:
                channel_list = ["slow", "fast"]

            label_numeric = label.item()
            for channel in channel_list:
                
                # heatmap_info = {
                #     "input_vid_idx": [input_vid_idx],
                #     "label": [label_val],
                #     "pred": [pred],
                #     "correct": [(pred == label_val)],
                #     "channel": [channel],
                # }

                # check that the heatmaps exist
                # (let's just check the first frame)
                heatmap_frames_dir = os.path.join(
                    heatmaps_root_dir,
                    "frames",
                    f"{target_class}",
                    f"{target_class}_{input_vid_idx:06d}",
                    channel,
                )
                trajectory_frames_dir = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, "test", "target_masks", f"{target_class}", f"{target_class}_{input_vid_idx:06d}")
                # "/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_1/test/target_masks/circle/circle_000000"
                # "/research/cwloka/data/action_attn/synthetic_motion_experiments/experiment_1/i3d_output/heatmaps/eigen_cam/post_softmax/frames/circle/circle_000000/rgb/circle_000000_rgb_000001.jpg"

                sample_heatmap_frame_path = os.path.join(
                    heatmap_frames_dir,
                    f"{target_class}_{input_vid_idx:06d}_{channel}_000001.jpg",

                )
                # print(heatmap_frames_dir)
                # print(sample_heatmap_frame_path)
                # pdb.set_trace()

                # TODO decide if it should be a fatal error or just a warning if it doesn't exist? 
                assert os.path.exists(sample_heatmap_frame_path), f"could not find heatmaps {sample_heatmap_frame_path}; cannot compute metrics if heatmaps do not exist"
                

                # Compute metrics over the heatmap
                # pdb.set_trace()
                metric_results = heatmap_metrics(
                    heatmap_dir=heatmap_frames_dir, 
                    trajectory_dir=trajectory_frames_dir, 
                    metrics=metrics,
                    pathway=channel, 
                    thresh=0.2)
                
                # update video features in the data dictionary 
                data_dict["input_vid_idx"].append(input_vid_idx)
                data_dict["channel"].append(channel)
                data_dict["label"].append(target_class)
                data_dict["pred"].append(pred_class)
                data_dict["label_numeric"].append(label_numeric)
                data_dict["pred_numeric"].append(pred_numeric)
                data_dict["correct"].append((label_numeric == pred_numeric))

                # update metric results in the data dictionary
                for metric in metrics:
                    data_dict[metric].append(metric_results[metric])

                # print(data_dict)
                # pdb.set_trace()
                # # create dictionary to merge the other dictionaries together
                # curr_results = {}
                # curr_results.update(metric_results)
                # curr_results.update(experiment_params)
                # curr_results.update(heatmap_info)
                # # handle storage of results for multi-channel architectures, where the different
                # # channels will have different results
                # if input_vid_idx in results:
                #     for key, value in curr_results.items():
                #         results[input_vid_idx][key].append(value)
                # else:
                #     results[input_vid_idx] = copy.deepcopy(curr_results)


        # # Gather all the predictions across all the devices to perform ensemble.
        # if cfg.NUM_GPUS > 1:
        #     preds, labels, video_ids = du.all_gather([preds, labels, video_ids])
        # if cfg.NUM_GPUS:
        #     preds = preds.cpu()
        #     labels = labels.cpu()
        #     video_ids = video_ids.cpu()

        # test_meter.iter_toc()
        # # Update and log stats.
        # test_meter.update_stats(preds.detach(), labels.detach(), video_ids.detach())
        # test_meter.log_iter_stats(cur_iter)

        # test_meter.iter_tic()

    # TODO: may run into issues if any video index overwriting happens, want it to enum the entries
    # print(data_dict)

    print("check the length of the results dataframe, make sure all entries have same length")
    pdb.set_trace()
    results_dataframe = pd.DataFrame.from_dict(data_dict)
    if os.path.exists(cfg.METRICS.CSV_PATH):
        results_dataframe.to_csv(cfg.METRICS.CSV_PATH, mode='a', index=False, header=False)
    else:
        results_dataframe.to_csv(cfg.METRICS.CSV_PATH, index=False)


    # TODO log testing stats e.g. accuracy - figure out a nice way to do this 



def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg.NUM_GPUS, cfg.SHARD_ID)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    out_str_prefix = "lin" if cfg.MODEL.DETACH_FINAL_FC else ""

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    if (
        cfg.TASK == "ssl"
        and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.KNN_ON
    ):
        train_loader = loader.construct_loader(cfg, "train")
        out_str_prefix = "knn"
        if hasattr(model, "module"):
            model.module.init_knn_labels(train_loader)
        else:
            model.init_knn_labels(train_loader)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    save_inputs(test_loader, cfg, "test")

    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES
            if not cfg.TASK == "ssl"
            else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    if cfg.METRICS.ENABLE:
        test_meter = run_heatmap_metrics(test_loader, model, test_meter, cfg, writer)
        logger.info("metric calculations done")


    # Perform multi-view test on the entire dataset.
    if cfg.TEST.ENABLE: 
        test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
        if writer is not None:
            writer.close()
        result_string = (
            "_a{}{}{} Top1 Acc: {} Top5 Acc: {} MEM: {:.2f} dataset: {}{}"
            "".format(
                out_str_prefix,
                cfg.TEST.DATASET[0],
                test_meter.stats["top1_acc"],
                test_meter.stats["top1_acc"],
                test_meter.stats["top5_acc"],
                misc.gpu_mem_usage(),
                cfg.TEST.DATASET[0],
                cfg.MODEL.NUM_CLASSES,
            )
        )
        logger.info("testing done: {}".format(result_string))

        return result_string
