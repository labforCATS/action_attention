from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.datasets.utils as data_utils

import numpy as np
import torch
import cv2
import os
import pdb

logger = logging.get_logger(__name__)

from slowfast.datasets import loader


def save_dataloader_samples(cfg, n_batches=1):
    """Constructs a dataloader and saves data outputs as videos for `n_batches`.

    This allows us to check if the dataloader works properly. We can manually
    inspect the videos to examine:
        * spatial cropping and resizing
        * temporal cropping
        * data augmentation, if any
        * frame reordering, if any

    Args:
        cfg: config object
        n_batches (int): number of batches to iterate over and save
    """
    # Set up environment.
    du.init_distributed_training(cfg.NUM_GPUS, cfg.SHARD_ID)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # create dataloader, load data
    train_loader = loader.construct_loader(cfg, "train")
    loader.shuffle_dataset(train_loader, cur_epoch=0)

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # prepare the output directory
    video_dir = os.path.join(cfg.OUTPUT_DIR, "test_dataloader")
    if not os.path.isdir(video_dir):
        os.makedirs(video_dir)

    # get some videos from the dataloader
    for batch, (inputs, labels, index, time, meta) in enumerate(train_loader):
        if batch >= n_batches:
            break

        # inputs is a list containing 2 channels, each with shape BCTHW
        # permute input from BCTHW to BTHWC, then undo normalization transform
        slow_tensor = inputs[0].permute(0, 2, 3, 4, 1)
        slow_tensor = data_utils.revert_tensor_normalize(
            slow_tensor, cfg.DATA.MEAN, cfg.DATA.STD
        )
        fast_tensor = inputs[1].permute(0, 2, 3, 4, 1)
        fast_tensor = data_utils.revert_tensor_normalize(
            fast_tensor, cfg.DATA.MEAN, cfg.DATA.STD
        )

        # scale from [0, 1] to [0, 255]
        slow_np = (slow_tensor.detach().numpy() * 255).astype("uint8")
        fast_np = (fast_tensor.detach().numpy() * 255).astype("uint8")

        for ch_idx, channel in enumerate([slow_np, fast_np]):
            for i in range(cfg.TRAIN.BATCH_SIZE):
                video_path = os.path.join(
                    video_dir,
                    f"channel_{ch_idx}_batch_{batch}_vid_{i}.mp4",
                )
                # note that in the constructor the size must be passed in as
                # (cols, rows) i.e. (width, height)
                # however, when passing in the actual frames it should be in
                # (rows, cols)
                video = cv2.VideoWriter(
                    video_path,
                    fourcc,
                    fps=25,
                    frameSize=(
                        channel.shape[-2],  # W
                        channel.shape[-3],  # H
                    ),
                    isColor=True,
                )

                for frame_num in range(channel.shape[1]):
                    frame = channel[i, frame_num, :, :, :]  # BTHWC
                    video.write(frame.astype("uint8"))

                cv2.destroyAllWindows()
                video.release()


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        # We added the line below
        cfg.NUM_GPUS = 1

    save_dataloader_samples(cfg, n_batches=1)
