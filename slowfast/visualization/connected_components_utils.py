import cc3d
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import cv2

from datetime import datetime
import os
import logging
import pdb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_heatmaps(heatmaps_dir, t_scale=1.0, s_scale=1.0, mask=False):
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
        mask (bool): If heatmaps_dir is a directory of mask images, set to True

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
        Returns a stacked and resized 3d array *with shape (T, W, H)*.
    """
    # load in heatmap data
    img_paths_list = os.listdir(heatmaps_dir)
    img_paths_list.sort()

    img_list = []

    for fname in img_paths_list:
        path = os.path.join(heatmaps_dir, fname)
        img = np.asarray(Image.open(path))

        if mask:
            img = img[:, :, 0]  # check why we need this?
        img_list.append(img)

    img_stack = np.stack(img_list, axis=0)  # dimensions (T, H, W)

    # permute the volume to be (T, W, H) so that we get a more intuitive
    # visualization with frames on the x-axis, width on the y-axis, and height
    # on the z-axis
    img_stack = np.swapaxes(img_stack, 1, 2)

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
    overlay=None,  # TODO: rename to something more descriptive
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
        overlay (3D np.array) the 3d heatmap volume for a ground truth volume to
            be overlayed on top of the volume specified in volume arg

    Output:
        interactive html file, saved to the given location.
    """
    if isomax is None:
        isomax = volume.max()
    elif overlay is not None:
        isomax = max(volume.max(), overlay.max())

    custom_colorscale = [
        [0, "rgb(0, 100, 100)"],  # Green
        [1, "rgb(0, 140, 140)"],  # Blue
    ]

    # prepare the X, Y, and Z axes
    t_scale_step = int(1 / t_scale)
    s_scale_step = int(1 / s_scale)

    X, Y, Z = np.mgrid[
        0 : volume.shape[0] * t_scale_step : t_scale_step,  # frames
        0 : volume.shape[1] * s_scale_step : s_scale_step,  # width
        0 : volume.shape[2] * s_scale_step : s_scale_step,  # height
    ]
    if overlay is not None:
        overlay_X, overlay_Y, overlay_Z = np.mgrid[
            0 : overlay.shape[0] * t_scale_step : t_scale_step,  # frames
            0 : overlay.shape[1] * s_scale_step : s_scale_step,  # width
            0 : overlay.shape[2] * s_scale_step : s_scale_step,  # height
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
                ],
                # layout attribute
            )
            step["args"][0]["visible"][
                i
            ] = True  # Toggle ith trace to "visible"

            steps.append(step)

        sliders = [
            dict(
                active=10,
                currentvalue={"prefix": "Threshold: "},
                pad={"t": 50},
                steps=steps,
            )
        ]

        fig.update_layout(
            sliders=sliders,
        )

    else:
        if isomin is None:
            isomin = 0.1 * volume.max()

        if overlay is not None:
            fig = go.Figure(
                data=go.Volume(
                    x=overlay_X.flatten(),
                    y=overlay_Y.flatten(),
                    z=overlay_Z.flatten(),
                    value=overlay.flatten(),
                    isomin=isomin,
                    isomax=isomax,
                    opacity=0.08,  # needs to be small to see through all surfaces
                    surface_count=8,
                    colorscale=custom_colorscale,
                )
            )

            fig.add_trace(
                go.Volume(
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
        else:
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

    # add labels and fix default axis view
    fig.update_layout(
        scene={
            "xaxis_title": "Frames",
            "yaxis_title": "Image Width",
            "zaxis_title": "Image Height",
            "xaxis": {
                "autorange": "reversed",
                "showticklabels": True,
                "dtick": 20,
            },  # frames axis
            "zaxis": {
                "autorange": "reversed",
                "showticklabels": True,
                "dtick": 120,
            },  # height axis
            "yaxis": {"showticklabels": True, "dtick": 120},
        },
        font=dict(size=16),
        # margin=dict(l=100, r=100, t=100, b=100),
    )

    # save interactive figure
    if not fpath.endswith(".html"):
        fpath += ".html"
    fdir, fname = os.path.split(fpath)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    fig.write_html(fpath)


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
    component_volumes = get_components(volume, thresh)
    max_intensity = volume.max()

    # visualize components in order of largest to smallest volume
    for i, cc in enumerate(component_volumes):
        fpath = os.path.join(output_dir, f"component_{i:03d}.html")

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
    plot_indiv_components=True,
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
        plot_indiv_components (bool): whether or not to also plot individual
            3d components

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
                    os.path.join(output_dir, "heatmap_volume_with_slider.html"),
                    surface_count,
                    t_scale,
                    s_scale,
                    slider=True,
                )

                if plot_indiv_components:
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


def get_components(volume, thresh=0.0):
    """Extract the 3d heatmap connected components for a video.

    Args:
        volume (3D np.array): a 3d heatmap volume
        thresh (float): float between 0.0 and 1.0 indicating the lower threshold
            for heatmap data to be displayed

    Output:
        returns a list of component volumes (each are 3d np arrays), sorted
        by largest to smallest volume
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

    component_volumes = []

    # visualize components in order of largest to smallest volume
    for component_idx in sorted_idx[::-1]:
        # increment the component index, since we're ignoring the background
        component_idx += 1

        # extract voxels from this specific component
        cc = np.where(result == component_idx, volume, 0)
        component_volumes.append(cc)

    return component_volumes


def get_3d_measurements(component_volume):
    """Given a single connected component volume, compute the areas over each
    frame, and the temporal depth over each pixel.

    Args:
        component_volume should have dimensions (T, H, W) and is assumed to be a single continuous component

    Output:
        returns a dictionary with two keys, "spatial_area" and
        "temporal_depth", each having a list in the key containing the
        unsorted area and depth values. (Note that these lists are not
        necessarily the same size, but do have the same sum.)
    """
    # binarize the component
    component_volume = np.where(component_volume != 0, 1, 0)

    temp_depths = []
    spat_areas = []

    # iter over each pixel in the spatial frame
    for r in range(component_volume.shape[1]):
        for c in range(component_volume.shape[2]):
            # compute temporal depth. if the pixel has non-continous segments of
            # the heatmap, count each segment separately

            # pad the pixel's temporal array with zeros (in case the heatmap
            # values reach the edge, i.e. first/last frame of the video)
            time_px = np.pad(
                component_volume[:, r, c],
                pad_width=1,
                mode="constant",
                constant_values=0,
            )

            # get the indices of temporally continuous segments
            segment_start_idxs = np.where(np.diff(time_px) == 1)[0]
            segment_end_idxs = np.where(np.diff(time_px) == -1)[0]

            assert segment_start_idxs.shape == segment_end_idxs.shape
            segment_durations = segment_end_idxs - segment_start_idxs

            temp_depths += list(segment_durations)

    # iter over each frame in the video
    for t in range(component_volume.shape[0]):
        # compute area
        frame = component_volume[t]

        # get the connected components in the frame (even though it is one
        # continous 3d volume, a 2d slice may have multiple disconnected
        # components, or no components)
        (
            n_components,
            labels,
            stats,
            _,
        ) = cv2.connectedComponentsWithStats(
            frame.astype(np.uint8), connectivity=8
        )

        # iter over components (ignore component 0, which is the background)
        for c in range(1, n_components):
            area = stats[c, cv2.CC_STAT_AREA]
            spat_areas.append(area)

    assert sum(temp_depths) == sum(spat_areas)

    return {"temp_depths": temp_depths, "spat_areas": spat_areas}


def generate_stats(component_volume):
    """
    calculates the mean, median, and mode for the spatial area and
    temporal depths of each component

    Args:
        component_volume(tuple): should have dimensions (T, H, W) and
        is assumed to be a single continuous component
            T: temporal depth, H: height, W: width
    Returns:
        6-element tuple in the specific order of spatial mean, spatial median,
        spatial mode, temporal mean, temporal median, temporal mode
    """

    vol_area_dict = get_3d_measurements(component_volume)
    spatial_key = "spat_area"
    temporal_key = "temp_depth"

    spatial_area_info = vol_area_dict[spatial_key]
    temporal_depth_info = vol_area_dict[temporal_key]

    spatial_mean = np.mean(spatial_area_info)
    spatial_median = np.median(spatial_area_info)
    spatial_mode = np.mode(spatial_area_info)

    temporal_mean = np.mean(temporal_depth_info)
    temporal_median = np.median(temporal_depth_info)
    temporal_mode = np.mode(temporal_depth_info)

    return (
        spatial_mean,
        spatial_median,
        spatial_mode,
        temporal_mean,
        temporal_median,
        temporal_mode,
    )


def generate_overlay(
    heatmaps_root_dir, output_path, overlay_dir, t_scale, s_scale
):
    """
    Generates heatmaps with another heatmap overlayed on top of it
    Args:
        heatmaps_root_dir (str): path to the folder containing all and only the
            heatmaps for a single video
            (e.g. "/.../heatmaps/grad_cam/6AyQsS4WC4A/slow"; see below for
            example structure)
        output_path (str): desired path name for the generated output 3d
            heatmap. must end with ".html"
        overlay_dir (str): path to the folder containing all and only the
            heatmaps to be overlayed over heatmaps specified in heatmaps_root_dir.
            Follows the same structure as heatmaps_root_dir
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

    """
    # set up output folder
    assert output_path.endswith(".html")
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load in heatmap of model prediction
    heatmap = load_heatmaps(heatmaps_root_dir, t_scale, s_scale)

    # # load in heatmap to overlay on top
    overlay = load_heatmaps(
        overlay_dir,
        t_scale * 0.64,  # scale of dimensions of overlay to model heatmap
        s_scale * 0.64,
        mask=True,  # if overlay_dir is a directory of masks, set to true
    )
    # TODO: slider for overlay is not currently implemented
    # plots final overlayed heatmap
    plot_heatmap(heatmap, output_path, slider=False, overlay=overlay)


if __name__ == "__main__":
    # heatmaps_root_dir = "/research/cwloka/projects/hannah_sandbox/outputs/synthetic_vids/dataset_1000_subset/ispy_0.1_9/output/grad_cam/heatmaps/grad_cam/"
    # output_root_dir = "/research/cwloka/projects/hannah_sandbox/outputs/synthetic_vids/dataset_1000_subset/ispy_0.1_9/output/grad_cam/heatmaps/grad_cam_volumes/"
    # model_arch = "slowfast"
    # t_scale = 0.25
    # s_scale = 1 / 8

    # print("plotting all heatmaps")

    # plot_all_heatmaps(
    #     heatmaps_root_dir=heatmaps_root_dir,
    #     output_root_dir=output_root_dir,
    #     model_arch=model_arch,
    #     surface_count=8,
    #     thresh=0.2,
    #     t_scale=t_scale,
    #     s_scale=s_scale,
    # )

    generate_overlay(
        "/research/cwloka/data/action_attn/synthetic_motion_7_classes/slowfast_outputs/epoch_45_outputs/heatmaps/grad_cam/pre_softmax/frames/000501/fast",
        "/research/cwloka/projects/rohit_sandbox/heatmap",
        "/research/cwloka/data/action_attn/synthetic_motion_7_classes/ispy_0.1_9/test/target_masks/triangle/triangle_000004",
        0.75,
        1 / 2,
    )
