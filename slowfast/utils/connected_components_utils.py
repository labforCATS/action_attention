import cc3d
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

from datetime import datetime
import os
import logging
import pdb

logger = logging.getLogger(__name__)


def load_heatmaps(heatmaps_dir, t_scale=1.0, s_scale=1.0):
    """Load heatmaps for a single video.

    Args:
        heatmaps_dir (str): path to the folder containing all and only the
            heatmaps for a single video
            (e.g. "/.../heatmaps/grad_cam/6AyQsS4WC4A/slow"; see below for
            example structure)
        t_scale (float): factor for rescaling the time dimension.
            < 1.0 to downsample, > 1.0 to upsample
        s_scale (float): factor for rescaling the spatial dimensions.
            < 1.0 to downsample, > 1.0 to upsample

        Example directory of heatmaps:
            |-heatmaps/
                |-<grad_cam_type>/
                    |-<videoidx>/
                        |-<stream_type_0>/ (optional dir, only for the SlowFast model)
                            |-<videoidx>_000001.jpg
                            |-<videoidx>_000002.jpg
                            |-<videoidx>_000003.jpg
                        ...
                        |-<stream_type_1>/ (optional dir, only for the SlowFast model)
                            |-<videoidx>_000001.jpg
                            |-<videoidx>_000002.jpg
                            |-<videoidx>_000003.jpg
                        ...
                    |-<videoidx>/
                        |-...
                    |-<videoidx>/
                        |-...
                    ...

    Output:
        Returns a stacked and resized 3d array.
    """
    # load in heatmap data
    img_paths_list = os.listdir(heatmaps_dir)
    img_paths_list.sort()

    img_list = []

    for fname in img_paths_list:
        path = os.path.join(heatmaps_dir, fname)
        img = np.asarray(Image.open(path))
        img_list.append(img)

    img_stack = np.stack(img_list, axis=0)  # dimensions (T, H, W)

    # downsample the image_stack
    img_stack = zoom(img_stack, (t_scale, s_scale, s_scale))

    return img_stack


def plot_heatmap(
    volume,
    fpath,
    surface_count=8,
    t_scale=1.0,
    s_scale=1.0,
    slider=False,
    isomin=None,
    isomax=None,
):
    """Plots a heatmap in 3D and saves as an interactive html file.

    Args:
        volume (3D np.array): the 3d heatmap volume to be plotted
        fpath (str): folder to which the output html file will be saved.
        surface_count (int): number of 'surfaces' to plot in the 3d volume.
            needs to be large enough for good volume rendering but not too large
            that it hinders viewing
        t_scale (float): factor for rescaling the time dimension.
            < 1.0 to downsample, > 1.0 to upsample
        s_scale (float): factor for rescaling the spatial dimensions.
            < 1.0 to downsample, > 1.0 to upsample
        slider (bool): if True, creates interactive slider that allows you to
            adjust the heatmap's minimum displaying threshold. if False, creates
            the heatmap with a minimum display threshold of 10%
        isomin (int/float): optional minimum display threshold for the heatmap,
            only affects plot when slider==False.
        isomax (int/float): optional maximum display threshold for the heatmap.
            By default, this is just the max value in the heatmap volume.

    Output:
        interactive html file, saved to the given location.
    """
    if isomax is None:
        isomax = volume.max()

    # prepare the X, Y, and Z axes
    t_scale_step = int(1 / t_scale)
    s_scale_step = int(1 / s_scale)

    X, Y, Z = np.mgrid[
        0 : volume.shape[0] * t_scale_step : t_scale_step,
        0 : volume.shape[1] * s_scale_step : s_scale_step,
        0 : volume.shape[2] * s_scale_step : s_scale_step,
    ]

    start = datetime.now()

    if slider:
        # Create figure
        fig = go.Figure()

        # Add traces, one for each slider step threshold
        for step in np.arange(0, 1, 0.1):
            fig.add_trace(
                go.Volume(
                    visible=False,
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=volume.flatten(),
                    isomin=int(step * volume.max()),
                    isomax=isomax,
                    opacity=0.1,  # needs to be small to see through all surfaces
                    surface_count=surface_count,
                )
            )

        # Set the default visible plot
        fig.data[0].visible = True

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[
                    {"visible": [False] * len(fig.data)},
                    {
                        "title": "Slider switched to threshold: {:.1f}".format(
                            i / 10
                        )
                    },
                ],  # layout attribute
            )
            step["args"][0]["visible"][
                i
            ] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [
            dict(
                active=10,
                currentvalue={"prefix": "Threshold: "},
                pad={"t": 50},
                steps=steps,
            )
        ]

        fig.update_layout(sliders=sliders)

    else:
        if isomin is None:
            isomin = 0.1 * volume.max()

        fig = go.Figure(
            data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=volume.flatten(),
                isomin=isomin,
                isomax=isomax,
                opacity=0.1,  # needs to be small to see through all surfaces
                surface_count=surface_count,
            )
        )

    # add labels
    fig.update_layout(
        scene=dict(
            xaxis_title="Frames -->",
            yaxis_title="Image Width",
            zaxis_title="Image Height",
        ),
    )

    # save interactive figure
    if not fpath.endswith(".html"):
        fpath += ".html"
    fdir, fname = os.path.split(fpath)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    fig.write_html(fpath)
    logger.info("heatmap volume generated and saved", datetime.now() - start)


def plot_components(volume, output_dir, thresh=0.0, t_scale=1.0, s_scale=1.0):
    """Plots and saves a set of 3D heatmaps, one for each 3D connected
        component in the heatmap volume.

    Args:
        volume (3D np.array): the 3d heatmap volume to be plotted
        output_dir (str): folder to which the output html files will be saved.
        thresh (float): float between 0.0 and 1.0 indicating the lower threshold
            for heatmap data to be displayed
        t_scale (float): factor for rescaling the time dimension.
            < 1.0 to downsample, > 1.0 to upsample
        s_scale (float): factor for rescaling the spatial dimensions.
            < 1.0 to downsample, > 1.0 to upsample

    Output:
        multiple interactive html files, saved to the given output directory.

    """
    # binarize the heatmaps using a threshold
    max_intensity = volume.max()
    volume_bin = np.where(volume >= thresh * max_intensity, 1, 0)

    # extract 3d connected components
    connectivity = 26  # 26, 18, and 6 (3D) are allowed
    result = cc3d.connected_components(volume_bin, connectivity=connectivity)
    stats = cc3d.statistics(result)

    # get the indices of the volumes, sorted
    # we don't want index 0, which is the component for the 'background'
    # (i.e. under the threshold)
    sorted_idx = np.argsort(stats["voxel_counts"][1:])

    # visualize components in order of largest to smallest volume
    for i, component_idx in enumerate(sorted_idx[::-1]):
        component_idx += 1
        # increment the component index, since we're ignoring the background

        fpath = os.path.join(output_dir, f"component_{i:03d}.html")

        # extract voxels from this specific component
        cc = np.where(result == component_idx, volume, 0)

        logger.info(f"generating 3d plot for component {i}")
        plot_heatmap(
            cc,
            fpath,
            t_scale=t_scale,
            s_scale=s_scale,
            slider=False,
            isomin=thresh * max_intensity,
            isomax=max_intensity,
        )


def plot_all_heatmaps(
    heatmaps_root_dir,
    output_root_dir,
    model_arch,
    surface_count=8,
    thresh=0.2,
    t_scale=1.0,
    s_scale=1.0,
):
    """Plots and saves a set of 3D heatmaps for all videos in a directory.

        Note: this is currently only implemented for a file structure following the slowfast architecture

    Args:
        heatmaps_root_dir (str): path to the folder containing video
            subdirectories (e.g. "/.../heatmaps/grad_cam/"; see below for
            example structure)
        output_root_dir (str): output folder that 3d heatmaps will be saved to
        model_arch (str): model architecture, indicating what file structure
            exists. currently this function is only implemented for
            model_arch == "slowfast"
        surface_count (int): number of 'surfaces' to plot in the 3d volume.
            needs to be large enough for good volume rendering but not too large
            that it hinders viewing
        thresh (float): float between 0.0 and 1.0 indicating the lower threshold
        t_scale (float): factor for rescaling the time dimension.
            < 1.0 to downsample, > 1.0 to upsample
        s_scale (float): factor for rescaling the spatial dimensions.
            < 1.0 to downsample, > 1.0 to upsample

        Example directory of heatmaps:
            |-heatmaps/
                |-grad_cam/ # or its variants
                    |-<videoidx>/
                        |-<stream_type_0>/ (optional dir, only for the SlowFast model)
                            |-<videoidx>_000001.jpg
                            |-<videoidx>_000002.jpg
                            |-<videoidx>_000003.jpg
                        ...
                        |-<stream_type_1>/ (optional dir, only for the SlowFast model)
                            |-<videoidx>_000001.jpg
                            |-<videoidx>_000002.jpg
                            |-<videoidx>_000003.jpg
                        ...
                    |-<videoidx>/
                        |-...
                    |-<videoidx>/
                        |-...
                    ...

    Output:
        multiple interactive html files, saved to the given output directory.

    """
    if model_arch != "slowfast":
        raise NotImplementedError

    vid_ids = os.listdir(heatmaps_root_dir)
    vid_ids.sort()

    for video_idx in vid_ids:
        if model_arch == "slowfast":
            for stream in ["slow", "fast"]:
                heatmaps_dir = os.path.join(
                    heatmaps_root_dir, str(video_idx), stream
                )
                logger.info("generating heatmap volumes to " + heatmaps_dir)
                output_dir = os.path.join(
                    output_root_dir, str(video_idx), stream
                )

                img_stack = load_heatmaps(heatmaps_dir, t_scale, s_scale)

                plot_heatmap(
                    img_stack,
                    os.path.join(
                        output_dir, "heatmap_volume_with_slider.html"
                    ),
                    surface_count,
                    t_scale,
                    s_scale,
                    slider=True,
                )

                plot_components(
                    img_stack,
                    output_dir=output_dir,
                    thresh=thresh,
                    t_scale=t_scale,
                    s_scale=s_scale,
                )


def heatmap_stats(volume, thresh=0.2):
    """Compute and plot basic statistics for each connected component in the heatmap volume, including
        * histograms for temporal depth and spatial area
            * mean, median, and mode
        * sorted temporal and spatial area bar graphs
            * normalized area under curves for comparison

    Args:
        volume (3D np.array): the 3d heatmap volume to be plotted
        thresh (float): float between 0.0 and 1.0 indicating the lower threshold
            for heatmap data to be displayed

    Output:
        multiple interactive html files, saved to the given output directory.

    """
    # binarize the heatmaps using a threshold
    max_intensity = volume.max()

    volume = np.where(volume >= thresh * max_intensity, 1, 0)

    # extract 3d connected components
    connectivity = 26  # 26, 18, and 6 (3D) are allowed
    result = cc3d.connected_components(volume, connectivity=connectivity)
    stats = cc3d.statistics(result)
    n_components = (
        len(stats["voxel_counts"]) - 1
    )  # ignoring "background" element

    # TODO:
    raise NotImplementedError

    for component_idx in range(n_components):
        pass
        # plot spatial histogram

        # plot temporal histogram

        # plot spatial AUC

        # plot temporal AUC
