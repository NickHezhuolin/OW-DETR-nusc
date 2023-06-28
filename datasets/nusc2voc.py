import os
import shutil
from typing import Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.axes import Axes
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
import pdb

def get_color(category_name: str) -> Tuple[int, int, int]:
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """

    return nusc.colormap[category_name]

def render(boxes,
            axis: Axes,
            view: np.ndarray = np.eye(3),
            normalize: bool = False,
            colors: Tuple = ('b', 'r', 'k'),
            linewidth: float = 2) -> None:
    """
    Renders the box in the provided Matplotlib axis.
    :param axis: Axis onto which the box should be drawn.
    :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
    :param normalize: Whether to normalize the remaining coordinate.
    :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
        back and sides.
    :param linewidth: Width in pixel of the box sides.
    """
    corners = view_points(boxes.corners(), view, normalize=normalize)[:2, :]

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                    [corners.T[i][1], corners.T[i + 4][1]],
                    color=colors[2], linewidth=linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0])
    draw_rect(corners.T[4:], colors[1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    axis.plot([center_bottom[0], center_bottom_forward[0]],
                [center_bottom[1], center_bottom_forward[1]],
                color=colors[0], linewidth=linewidth)
    
    return corners

# Initialize NuScenes. Requires the dataset to be stored on disk.
nusc = NuScenes(version='v1.0-mini', dataroot='/data/dataset2tssd/nuscenes-mini', verbose=True)

# Specify a path to save the images and annotations.
save_annotations_path = '/data/dataset2tssd/OWDETR/nusc_voc_dataset/Annotations'

# Make sure the output directories exist.
os.makedirs(save_annotations_path, exist_ok=True)

# All sensor list

CAMERA_SENSOR = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    # Iterate over all the samples in the dataset.
for sensor_name in CAMERA_SENSOR:
    for sample in nusc.sample:
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample['data'][sensor_name], box_vis_level = BoxVisibility.ANY)
        image = ET.Element("image", filename=data_path)
        size = ET.SubElement(image, "size")
        ET.SubElement(size, "width").text = str(1600)
        ET.SubElement(size, "height").text = str(900)
        ET.SubElement(size, "depth").text = str(3)
        
        # 读入token
        camera_token = sample['data'][sensor_name]
        pointsensor_token = sample['data']['LIDAR_TOP']
        cam = nusc.get('sample_data', camera_token)
        pointsensor = nusc.get('sample_data', pointsensor_token)
        
        data = Image.open(data_path)
        
        # Init axes.
        ax = None
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 16))

        # Show image.
        ax.imshow(data)

        # Show boxes.
        for box in boxes:
            # 3DBoxCorner translate to 2DCorner
            c = np.array(get_color(box.name)) / 255.0
            corner = render(box, ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))
            
            # Filter 2D
            x_coords = corner[0, :]  # 选择所有的x坐标
            y_coords = corner[1, :]  # 选择所有的y坐标

            x_min = np.min(x_coords)
            y_min = np.min(y_coords)
            x_max = np.max(x_coords)
            y_max = np.max(y_coords)
            
            # Create VOC bounding box
            ob = ET.SubElement(image, "object")
            ET.SubElement(ob, "name").text = box.name
            ET.SubElement(ob, "pose").text = "Unspecified"
            ET.SubElement(ob, "truncated").text = "0"
            ET.SubElement(ob, "difficult").text = "0"
            bbox = ET.SubElement(ob, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(x_min)
            ET.SubElement(bbox, "ymin").text = str(y_min)
            ET.SubElement(bbox, "xmax").text = str(x_max)
            ET.SubElement(bbox, "ymax").text = str(y_max)

        # Limit visible range.
        ax.set_xlim(0, data.size[0])
        ax.set_ylim(data.size[1], 0)        

        # Save to XML
        xml_str = ET.tostring(image)
        root = ET.fromstring(xml_str)
        tree = ET.ElementTree(root)
        tree.write(os.path.join(save_annotations_path, sample['token']+'_ann.xml'))
        
