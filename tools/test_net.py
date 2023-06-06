#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import torchvision
import pdb
import cv2

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
import torchvision.transforms as transforms
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter


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

    for cur_iter, (inputs, labels, video_idx, time,
                   meta) in enumerate(test_loader):
        #TODO: grab inputs!
        # print("test_net")
        # print(type(inputs))
        # print(len(inputs))
        # print(type(inputs[0]))
        # print(inputs[0].size())
        # print(inputs[1].size())
        # print(type(inputs[1]))
        # print(video_idx)
        # print(labels.size())
        # print(video_idx.shape)
        # print(video_idx)
        # print(video_idx.item())
        # pdb.set_trace()
        


        # naming convention: input_[vid name]_[slow/fast]
        if cfg.TEST.SAVE_INPUT_VIDEO:
            test_folder_path = os.path.join(cfg.VIS_MODEL_INPUT_DIR, "test")
            slow_folder = os.path.join(test_folder_path, "slow")
            fast_folder = os.path.join(test_folder_path, "fast")
                
            if not os.path.exists(cfg.VIS_MODEL_INPUT_DIR):
                os.makedirs(cfg.VIS_MODEL_INPUT_DIR)
            if not os.path.exists(test_folder_path):
                os.makedirs(test_folder_path)
            if not os.path.exists(slow_folder):
                os.makedirs(slow_folder)
            if not os.path.exists(fast_folder):
                os.makedirs(fast_folder)

            for batch in range(cfg.TEST.BATCH_SIZE):
                slow_tensor = inputs[0]
                fast_tensor = inputs[1]
                
                num_slow_frame = slow_tensor.size(dim=2)
                num_fast_frame = fast_tensor.size(dim=2)
                
                
                for slow_frame in range(num_slow_frame):
                    slow_tensor_image = slow_tensor[batch, :, slow_frame, :,:].numpy()*255
                    slow_tensor_image = np.moveaxis(slow_tensor_image, 0, -1)
                    slow_name = "input_slow_vid"+ str(video_idx.item())+ "frame" +str(slow_frame) + ".jpg"
                    slow_name = os.path.join(slow_folder, slow_name)
                    cv2.imwrite(slow_name, slow_tensor_image)
                for fast_frame in range(num_fast_frame):
                    fast_tensor_image = fast_tensor[batch, :, fast_frame, :,:].numpy()*255
                    fast_tensor_image = np.moveaxis(fast_tensor_image, 0, -1)
                    fast_name = "input_fast_vid"+ str(video_idx.item())+ "frame" +str(fast_frame) + ".jpg"
                    fast_name = os.path.join(fast_folder, fast_name)
                    cv2.imwrite(fast_name, fast_tensor_image)

            # isolate each image in the batch
            # find out whether bgr or rgb
            # save as frame in video writer
            # pdb.set_trace()

        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list, )):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list, )):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.

            # TODO: grab inputs
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (ori_boxes.detach().cpu()
                         if cfg.NUM_GPUS else ori_boxes.detach())
            metadata = (metadata.detach().cpu()
                        if cfg.NUM_GPUS else metadata.detach())

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes),
                                      dim=0)
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
            train_labels = (model.module.train_labels if hasattr(
                model, "module") else model.train_labels)
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
            preds, labels, video_idx = du.all_gather(
                [preds, labels, video_idx])
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(preds.detach(), labels.detach(),
                                video_idx.detach())
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
            # print(new_preds)
            print(all_labels)
            print(predictions)
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR,
                                     cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info("Successfully saved prediction results to {}".format(
                save_path))

    test_meter.finalize_metrics()
    return test_meter


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

    if (cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            and cfg.CONTRASTIVE.KNN_ON):
        train_loader = loader.construct_loader(cfg, "train")
        out_str_prefix = "knn"
        if hasattr(model, "module"):
            model.module.init_knn_labels(train_loader)
        else:
            model.init_knn_labels(train_loader)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos %
            (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS) == 0)
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            test_loader.dataset.num_videos //
            (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES if not cfg.TASK == "ssl" else
            cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
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
        ))
    logger.info("testing done: {}".format(result_string))

    return result_string
