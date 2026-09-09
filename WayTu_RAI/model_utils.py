import yaml
import os
import robotic as ry
import numpy as np
import time
import math
import csv
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation as R
import torch.nn as nn
import pandas as pd 
import random
import torch
import matplotlib.pyplot as plt
import open3d as o3d
import copy
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from scipy.spatial import cKDTree



# from WayTu_RAI.GenerateEnvironment import GenerateEnvironment


# Read the configuration file
def get_config(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return data

# def test(cfg):
#     num_trials = cfg['test-num-trials']

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     predictions = model(graph_batch.to(device))



def visualize_point_cloud_with_corners(points, labels, corner_points, corner_colors):
    """
    Visualize a labeled point cloud with bounding box corners as additional points.

    Parameters:
        point_cloud_with_labels (numpy.ndarray): Nx4 array of point cloud with labels.
        corner_points (numpy.ndarray): Mx3 array of bounding box corner points.
        corner_colors (numpy.ndarray): Mx3 array of RGB colors for corner points.
    """

    # Create Open3D point cloud for labeled points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Generate a random color for each label
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    colormap = plt.get_cmap('tab20', num_labels)

    colors = np.zeros((len(labels), 3))
    for label in unique_labels:
        color = colormap(label / num_labels)[:3]  # Normalize label index
        colors[labels == label] = color

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a point cloud for corners
    corner_pcd = o3d.geometry.PointCloud()
    corner_pcd.points = o3d.utility.Vector3dVector(corner_points)
    corner_pcd.colors = o3d.utility.Vector3dVector(corner_colors)

    # Visualize the labeled points and corners
    o3d.visualization.draw_geometries([pcd, corner_pcd])

def visualize_pc_with_labels(points, labels=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if labels is not None: 
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        colormap = plt.get_cmap('viridis', num_labels)

        normalized_labels = (labels - labels.min()) / (labels.max() - labels.min())
        colors = colormap(normalized_labels.flatten())[:, :3] 
        colors = np.ascontiguousarray(colors)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        for i, label in enumerate(unique_labels):
            color = colormap(i / (num_labels - 1))[:3]  # Get color
            print(f"Label {label}: RGB {tuple(int(c * 255) for c in color)}")

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

def train_validation_split(cfg):
    csv_path = os.path.join(cfg['dataset-train-path'], 'dataset_info.csv')
    dataset = pd.read_csv(csv_path)

    train, test = train_test_split(dataset, test_size=0.2, random_state=11)
    return train, test

def get_dataset(cfg, name):
    csv_path = os.path.join(cfg[name], 'dataset_info.csv')
    dataset = pd.read_csv(csv_path)

    return dataset
          

# ############# FOR CREATING ENVIRONMENT ############# # 

# A general function for adding mesh objects to the environment
def add_mesh_object(C, frame_name, mesh_object, parent=None, mass = 0.05,joint = False,):
    
    if parent is not None: 
        mesh_frame = C.addFrame(frame_name, parent)
    else: 
        mesh_frame = C.addFrame(frame_name)
    
    vertices = np.array(mesh_object.vertices, dtype=float)
    faces = np.array(mesh_object.faces, dtype=int)
    mesh_frame.setMesh(vertices, faces, True)

    mesh_frame.setContact(1) 
    mesh_frame.setMass(1.0)

    if joint == True: 
        mesh_frame.setJoint(ry.JT.rigid)

        
def add_different_target(C, target_name, frame_name, parent):
    if target_name == "cube":
        alpha = random.uniform(-0.01, 0.01)
        add_shape(C= C, 
                  frame_name=frame_name, 
                  joint= True ,
                  parent = parent, 
                  shape=[0.05 + alpha, 0.05 + alpha, 0.05 + alpha, 0.00], 
                  color = [0, 0, 1.0], mass= 0.00001
                  ) 
    elif target_name == "puck":
        alpha = random.uniform(-0.001, 0.001)
        beta = random.uniform(-0.001, 0.001)
        add_shape(C= C, 
                frame_name=frame_name, 
                joint= True ,
                parent = parent, 
                shape=[0.05 + alpha, 0.05 + alpha, 0.025, 0.015], 
                color = [0, 0, 1.0], mass= 0.00001
                ) 
    elif target_name == "bottle":
        alpha = random.uniform(-0.003, 0.003)
        beta = random.uniform(-0.003, 0.003)

        body_width = 0.045 + alpha
        body_height = 0.045 + beta
        shoulder_height = 0.016
        neck_height = 0.018
        overlap = 0.002

        add_shape(
            C=C,
            frame_name=frame_name,
            joint=True,
            parent=parent,
            shape=[body_width, body_width, body_height, 0.010],
            color=[0, 0, 1.0],
            mass=0.000007
        )

        shoulder_z = body_height / 2 + shoulder_height / 2 - overlap

        add_shape(
            C=C,
            frame_name=f"{frame_name}-shoulder",
            parent=frame_name,
            relative_position=[0.0, 0.0, shoulder_z],
            shape=[0.034 + alpha, 0.034 + alpha,
                shoulder_height, 0.006],
            color=[0, 0, 1.0],
            mass=0.000002
        )

        neck_z = (
            body_height / 2
            + shoulder_height
            + neck_height / 2
            - 2 * overlap
        )

        add_shape(
            C=C,
            frame_name=f"{frame_name}-neck",
            parent=frame_name,
            relative_position=[0.0, 0.0, neck_z],
            shape=[0.022, 0.022, neck_height, 0.005],
            color=[0, 0, 1.0],
            mass=0.000001
        )
    
        
        


# A general function for adding frames to the environment 
def add_shape(C, frame_name, shape, color = [0.8, 0.8, 0.8],  
                  mass = 0.1, joint = False, parent = None, position = None, relative_position = None, relative_quaternion = None):
    
    if parent is not None: 
        obj = C.addFrame(frame_name, parent)
    else: 
        obj = C.addFrame(frame_name)
    
    if joint == True: 
        obj.setJoint(ry.JT.rigid)
        # obj.setJoint(ry.JT.free)
    
    obj.setShape(ry.ST.ssBox, shape)
    if position is not None: 
        obj.setPosition(position)
        
    if relative_position is not None: 
        obj.setRelativePosition(relative_position)

    if relative_quaternion is not None: 
        obj.setRelativeQuaternion(relative_quaternion)
    
    obj.setContact(True)
    obj.setMass(mass)
    obj.setColor(color)

def add_waypoint(C, frame_name, position, orientation):
        tool_waypoint = C.addFrame(frame_name)
        tool_waypoint.setShape(ry.ST.marker, size=[.1])
        tool_waypoint.setQuaternion(orientation)
        tool_waypoint.setPosition(position)


# Calculating the bounding boxes for primitive training object parts
def create_primitive_bb(C, obj_names):
    bounding_boxes = []

    for name in obj_names: 
        # print(name)
        obj = C.getFrame(name)
        center = obj.getPosition()
        info = obj.info()

        if info["shape"] == 'ssBox':
            size = info['size'][:3]
            half_size = np.array(size) / 2
            bounding_box_min = center - half_size
            bounding_box_max = center + half_size
        else :
            raise Exception("The shape mode was not found or is currently not implemented.") 
        
        bounding_box = {
                'min': bounding_box_min,
                'max': bounding_box_max
            }
        
        bounding_boxes.append(bounding_box)
    
    return bounding_boxes

# Combining the bounding boxes for primitive training object parts for an object 
def combine_primitive_bb(bounding_boxes):
    global_min = np.array([float('inf'), float('inf'), float('inf')])
    global_max = np.array([-float('inf'), -float('inf'), -float('inf')])

    for bbox in bounding_boxes:
        global_min = np.minimum(global_min, bbox['min'])
        global_max = np.maximum(global_max, bbox['max'])
    
    combined_bounding_box = {
            'min': global_min,
            'max': global_max
        }
    
    # print(combined_bounding_box)
    
    return combined_bounding_box

def create_bounding_box(half_len, center, quaternion):
    x_half, y_half,z_half = map(float, half_len)
    # Takes th half of the lenghts of axis's of bounding boxes
    min_point = [center[0] - x_half, center[1] - y_half, center[2] - z_half]
    max_point = [center[0] + x_half, center[1] + y_half, center[2] + z_half]
    
    bounding_box = {
        'min': np.array(min_point),
        'max': np.array(max_point),
        'center' : np.array(center),
        'quaternion' : np.array(quaternion)
    }
    return bounding_box

def create_bounding_box_tool(C, half_len, obj_name, quaternion):
    # Assume all the tools has only two parts
    handle_center = C.getFrame(obj_name + "-base").getPosition()
    head_center = C.getFrame(obj_name + "-head").getPosition()

    handle_shape = C.getFrame(obj_name + "-base").info()["size"]
    head_shape = C.getFrame(obj_name + "-head").info()["size"]

    x_weight_handle, x_weight_head = handle_shape[0], head_shape[0]
    y_weight_handle, y_weight_head = handle_shape[1], head_shape[1]

    center_x = (handle_center[0] * x_weight_handle + head_center[0] * x_weight_head) / (x_weight_handle + x_weight_head)
    center_y = (handle_center[1] * y_weight_handle + head_center[1] * y_weight_head) / (y_weight_handle + y_weight_head)
    center_z = handle_center[2]

    center = np.array([center_x, center_y, center_z])

    x_half, y_half,z_half = map(float, half_len)
    # Takes th half of the lenghts of axis's of bounding boxes
    min_point = [center[0] - x_half, center[1] - y_half, center[2] - z_half]
    max_point = [center[0] + x_half, center[1] + y_half, center[2] + z_half]
    
    bounding_box = {
        'min': np.array(min_point),
        'max': np.array(max_point),
        'center' : np.array(center),
        'quaternion' : np.array(quaternion)
    }
    return bounding_box

def create_mesh_bounding_box(C, obj_name, mesh):
    vertices = np.array(mesh.vertices, dtype=float)

    center_position = C.getFrame(obj_name).getPosition()
    min_point = vertices.min(axis=0) + center_position 
    max_point = vertices.max(axis=0) + center_position

    bounding_box = {
        'min': min_point,
        'max': max_point
    }
    return bounding_box

def create_primitive_bounding_box(C, obj_part_names):
    bounding_boxes = create_primitive_bb(C, obj_part_names)
    combined_bb = combine_primitive_bb(bounding_boxes)

    return combined_bb



# For debugging bounding boxes 
def draw_bounding_box(C, bounding_box, quaternion,name):
    # print(type(bounding_box['max']))
    size = bounding_box['max'] - bounding_box['min']
    
    center_position = (bounding_box['min'] + bounding_box['max']) / 2

    # Add transparent bounding box 
    frame = C.addFrame(name, "table")
    frame.setShape(ry.ST.ssBox, size.tolist() + [0.01])  # Adding a small radius for rounded edges
    frame.setPosition(center_position)
    frame.setQuaternion(quaternion)
    frame.setColor([1,1,0,.5])

    C.view()
    time.sleep(1.0)

def draw_transformed_courners(C, tranformed_courners, name):
    i= 0
    for pos in tranformed_courners:
        corner = C.addFrame("corner_" + str(name) + str(i))
        corner.setShape(ry.ST.marker, size=[.1])
        corner.setPosition(pos)
        i += 1

# ############### FOR DATASET CREATION ############### #

def label_points(labels, points, bounding_boxes):
    numeric_labels = np.full(len(points), labels["background"])

    for obj_name, bbox in bounding_boxes.items():
        x_min, y_min, z_min = bbox["min"]
        x_max, y_max, z_max = bbox["max"]
        
        # Create masks for checking if points are inside the bounding box
        in_x = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
        in_y = (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        in_z = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        
        # Combine masks to get a final mask for points inside the bounding box
        inside_bbox = in_x & in_y & in_z
        
        # Assign the corresponding numeric label to all points that are inside the bounding box
        numeric_labels[inside_bbox] = labels[obj_name]


def farthest_point_sampling(pcl, k):
    n,d = pcl.shape

    if d == 4: 
        pc = pcl[:, :3]
        labels = pcl[:, 3]
    else: 
        pc = pcl
    
    idx = random.randint(0, n - 1)
    downsampled_pc = [pc[idx]]
    
    if d == 4: 
        downsampled_labels = [labels[idx]]

    distances = np.ones(n) * np.inf
    
    for i in range(k-1): 
        last_added = downsampled_pc[-1]
        dist_to_last = np.linalg.norm(pc - last_added, axis =1)
        distances = np.minimum(distances, dist_to_last)
        farthest_idx = np.argmax(distances)

        downsampled_pc.append(pc[farthest_idx])
        if d ==4 :
            downsampled_labels.append(labels[farthest_idx])
    
    if d ==4 :
        downsampled_pc_with_labels = np.column_stack((downsampled_pc, downsampled_labels))
        return np.array(downsampled_pc_with_labels)
    else : 
        return np.array(downsampled_pc)



def adaptive_farthest_point_sampling(pcl, k):
    n, d = pcl.shape
    assert d == 4, "Input point cloud must have 4 columns (XYZ + Labels)"

    pc = pcl[:, :3]
    labels = pcl[:, 3]

    # Count unique labels and their point counts
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_labels)

    # Calculate the target number of points per class
    points_per_class = k // num_classes
    remaining_points = k % num_classes  # Handle rounding issues

    sampled_points = []
    for label in unique_labels:
        class_mask = labels == label
        class_points = pc[class_mask]
        class_labels = labels[class_mask]

        # Assign remaining points to some classes to make total = k
        extra_point = 1 if remaining_points > 0 else 0
        num_samples = min(points_per_class + extra_point, len(class_points))  # Don't oversample
        remaining_points -= extra_point  # Reduce remaining points

        if len(class_points) > 0:
            sampled = farthest_point_sampling(np.column_stack((class_points, class_labels)), num_samples)
            sampled_points.append(sampled)

    # Merge results
    downsampled_pc_with_labels = np.vstack(sampled_points) if sampled_points else np.array([])

    return downsampled_pc_with_labels

def uniform_object_point_sampling(pcl, points_per_object):
    """
    Args:
        pcl: numpy array of shape (N, 4), with XYZ + label
        points_per_object: int, number of points for *each* object/class

    Returns:
        sampled_pcl: numpy array of shape (num_objects * points_per_object, 4)
    """
    pc = pcl[:, :3]
    labels = pcl[:, 3]
    unique_labels = np.unique(labels)

    sampled_points = []
    for label in unique_labels:
        mask = labels == label
        class_points = pc[mask]
        class_labels = labels[mask]

        class_pc_with_labels = np.column_stack((class_points, class_labels))

        if len(class_pc_with_labels) >= points_per_object:
            sampled = farthest_point_sampling(class_pc_with_labels, points_per_object)
        else:
            # Upsample with replacement if not enough points
            indices = np.random.choice(len(class_pc_with_labels), points_per_object, replace=True)
            sampled = class_pc_with_labels[indices]

        sampled_points.append(sampled)

    return np.vstack(sampled_points)


def interpolate_points(pc, selected_pc, upsample_factor = 2):
    upsampled_points = []

    for i in range(selected_pc.shape[0]):
        distances = np.linalg.norm(pc - selected_pc[i], axis=1)

        # Exclude the same point
        same_point_mask = np.all(pc == selected_pc[i], axis=1)
        distances[same_point_mask] = np.inf

        nearest_idx = np.argmin(distances)
        nearest_point = pc[nearest_idx]

        for j in range(upsample_factor):
            alpha = (j + 1) / (upsample_factor + 1) 
            new_point = (1 - alpha) * selected_pc[i] + alpha * nearest_point
            upsampled_points.append(new_point)
        
    return np.vstack((selected_pc, np.array(upsampled_points)))


def upsample_point_clouds(pc,upsample_factor = 2):
    fps_points = farthest_point_sampling(pc, pc.shape[0]//2)
    upsampled_points = interpolate_points(pc, fps_points, upsample_factor=upsample_factor)
    print(f"Upsampled point cloud shape: ", upsampled_points.shape)
    return upsampled_points

def normalize_pc_and_bb (point_cloud, bounding_boxes):
    min_vals = point_cloud.min(axis=0)
    max_vals = point_cloud.max(axis=0)
    range_vals = max_vals - min_vals

    normalized_points = (point_cloud - min_vals) / range_vals

    print("IN NORMALIZE PC")
    print(bounding_boxes.keys())

    normalized_bboxes = {}
    for class_name, bbox in bounding_boxes.items():
        normalized_min = (bbox['min'] - min_vals) / range_vals
        normalized_max = (bbox['max'] - min_vals) / range_vals
        print(bbox)
        normalized_bboxes[class_name] = {
            'min': normalized_min,
            'max': normalized_max,
            'center' : bounding_boxes[class_name]['center'],
            'quaternion' : bounding_boxes[class_name]['quaternion'],

        }

    return normalized_points, normalized_bboxes

# Not tested yet! 
def filter_pc_one_bb(pc, bb, buffer):
    min_point = bb["min"]
    max_point = bb["max"]
    quaternion = bb['orientation']

    rotation_matrix = R.from_quat(quaternion).as_matrix()

    center = (min_point + max_point)/2.0 

    points_translated = pc - center
    points_local = np.dot(points_translated, rotation_matrix.T)

    min_xyz_buffered = min_point - buffer
    max_xyz_buffered = max_point + buffer

    mask = np.all((points_local >= min_xyz_buffered) & (points_local <= max_xyz_buffered), axis=1)
    filtered_points = pc[mask]

    return filtered_points


def filter_pc_all_bb(pc, bbs, buffer = 0.3):
    all_points = []
    for obj,bb in bbs.items():
        all_points.append(pc, bb, buffer)
    
    if all_points:
        all_points = np.vstack(all_points)
    
    return all_points

# Check this function!!
def label_pointcloud(cfg, point_cloud, bounding_boxes, threshold = 0.01):
    labels = np.full(len(point_cloud), -1)

    for object_name, bbox in bounding_boxes.items():
        min_coords = np.array(bbox['min'])
        max_coords = np.array(bbox['max'])

        min_coords -= threshold
        max_coords += threshold

        class_label = cfg["labels-list"].index(object_name)
        print(f"object name: {object_name}, class label {class_label}")

        inside_bbox = np.where(
        (point_cloud[:, 0] >= min_coords[0]) & (point_cloud[:, 0] <= max_coords[0]) &
        (point_cloud[:, 1] >= min_coords[1]) & (point_cloud[:, 1] <= max_coords[1]) &
        (point_cloud[:, 2] >= min_coords[2]) & (point_cloud[:, 2] <= max_coords[2])
        )[0]

        labels[inside_bbox] = class_label
    
    background_label = cfg["labels-list"].index("background")
    labels[labels == -1] = background_label

    labels = labels.reshape(-1, 1)
    point_cloud_with_labels = np.concatenate([point_cloud, labels], axis=1)

    return point_cloud_with_labels

def visualize_point_cloud_with_bboxes(point_cloud, bounding_boxes):
    import open3d as o3d
    from scipy.spatial.transform import Rotation as R

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Prepare geometries for visualization
    geometries = [pcd]

    for object_name, bbox in bounding_boxes.items():
        center = np.array(bbox['center'])
        quaternion = np.array(bbox['quaternion']) / np.linalg.norm(bbox['quaternion'])
        min_coords = np.array(bbox['min'])
        max_coords = np.array(bbox['max'])

        # Compute rotation matrix from quaternion
        rotation = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        rotation_matrix = rotation.as_matrix()

        # Compute extent (dimensions of the box)
        extent = max_coords - min_coords

        # Create Open3D OrientedBoundingBox
        obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extent)
        obb.color = [1, 0, 0]  # Red for bounding box

        geometries.append(obb)

    # Visualize the point cloud with bounding boxes
    o3d.visualization.draw_geometries(geometries)

def label_pointcloud_obb_quaternion(cfg, point_cloud, bounding_boxes, threshold=0.01):
    """
    Labels a point cloud based on oriented bounding boxes (OBBs) with debugging for slim boxes.

    Parameters:
        cfg (dict): Configuration dictionary containing a "labels-list".
        point_cloud (numpy.ndarray): Nx3 array of point coordinates.
        bounding_boxes (dict): Dictionary of bounding box data.
        threshold (float): Threshold to expand the bounding boxes (default: 0.01).

    Returns:
        numpy.ndarray: Labeled point cloud of shape Nx4 (x, y, z, label).
    """
    labels = np.full(len(point_cloud), -1)

    for object_name, bbox in bounding_boxes.items():
        center = np.array(bbox['center'])
        quaternion = np.array(bbox['quaternion'])
        min_coords_world = np.array(bbox['min'])
        max_coords_world = np.array(bbox['max'])

        # Normalize quaternion
        quaternion = quaternion / np.linalg.norm(quaternion)

        # Compute rotation matrix
        rotation = R.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]], scalar_first=True)
        rotation_matrix = rotation.as_matrix()

        # Debugging outputs
        print(f"Object: {object_name}")
        print(f"Quaternion: {quaternion}")
        print(f"Rotation Matrix:\n{rotation_matrix}")

        # Transform bounding box center and edges to local frame
        min_coords_local = np.dot(min_coords_world - center, rotation_matrix)
        max_coords_local = np.dot(max_coords_world - center, rotation_matrix)

        # Ensure min and max are consistent
        min_coords_local, max_coords_local = np.minimum(min_coords_local, max_coords_local), np.maximum(min_coords_local, max_coords_local)

        # Debugging: Print dimensions
        print(f"Bounding Box (local frame): Min: {min_coords_local}, Max: {max_coords_local}")

        # Transform points to the local frame of the bounding box
        points_local = np.dot(point_cloud - center, rotation_matrix)

        # Check which points lie inside the bounding box in the local frame
        inside_bbox = np.where(
            (points_local[:, 0] >= min_coords_local[0]) & (points_local[:, 0] <= max_coords_local[0]) &
            (points_local[:, 1] >= min_coords_local[1]) & (points_local[:, 1] <= max_coords_local[1]) &
            (points_local[:, 2] >= min_coords_local[2]) & (points_local[:, 2] <= max_coords_local[2])
        )[0]

        # Debugging: Check number of points inside
        print(f"Points inside bounding box for {object_name}: {len(inside_bbox)}")

        # Assign the label to these points
        if len(inside_bbox) > 0:
            class_label = cfg["labels-list"].index(object_name)
            labels[inside_bbox] = class_label

    # Assign background label to unlabeled points
    background_label = cfg["labels-list"].index("background")
    labels[labels == -1] = background_label

    # Combine the point cloud with the labels
    labels = labels.reshape(-1, 1)
    point_cloud_with_labels = np.concatenate([point_cloud, labels], axis=1)
    print("Final labeled point cloud shape: ", point_cloud_with_labels.shape)

    return point_cloud_with_labels


def compute_rotation_matrix_z(angle_degrees):
    """
    Computes the 3D rotation matrix for a rotation around the Z-axis.

    Parameters:
        angle_degrees (float): Rotation angle in degrees.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,          0,         1]
    ])


def label_pointcloud_obb(cfg, point_cloud, bounding_boxes, threshold=0.01):
    """
    Labels a point cloud based on oriented bounding boxes (OBBs) with rotation angles.

    Parameters:
        cfg (dict): Configuration dictionary containing a "labels-list".
        point_cloud (numpy.ndarray): Nx3 array of point coordinates.
        bounding_boxes (dict): Dictionary of bounding box data.
        threshold (float): Threshold to expand the bounding boxes (default: 0.01).

    Returns:

        numpy.ndarray: Labeled point cloud of shape Nx4 (x, y, z, label).
    """
    labels = np.full(len(point_cloud), -1)

    for object_name, bbox in bounding_boxes.items():
        center = np.array(bbox['center'])
        min_coords_world = np.array(bbox['min'])
        max_coords_world = np.array(bbox['max'])
        rotation_angle_z = bbox['rotation_z']  # Rotation angle in degrees

        # Compute rotation matrix
        rotation_matrix = compute_rotation_matrix_z(- rotation_angle_z)

        # Debugging: Print rotation matrix
        print(f"Object: {object_name}")
        print(f"Rotation Angle (degrees): {rotation_angle_z}")
        print(f"Rotation Matrix:\n{rotation_matrix}")

        # Transform bounding box center and edges to local frame
        min_coords_local = np.dot(min_coords_world - center, rotation_matrix.T)
        max_coords_local = np.dot(max_coords_world - center, rotation_matrix.T)

        # Ensure min and max are consistent
        min_coords_local, max_coords_local = np.minimum(min_coords_local, max_coords_local), np.maximum(min_coords_local, max_coords_local)

        # Debugging: Print bounding box dimensions
        print(f"Bounding Box (local frame): Min: {min_coords_local}, Max: {max_coords_local}")

        # Transform points to the local frame of the bounding box
        points_local = np.dot(point_cloud - center, rotation_matrix.T)

        # Check which points lie inside the bounding box in the local frame
        inside_bbox = np.where(
            (points_local[:, 0] >= min_coords_local[0]) & (points_local[:, 0] <= max_coords_local[0]) &
            (points_local[:, 1] >= min_coords_local[1]) & (points_local[:, 1] <= max_coords_local[1]) &
            (points_local[:, 2] >= min_coords_local[2]) & (points_local[:, 2] <= max_coords_local[2])
        )[0]

        # Debugging: Check number of points inside
        print(f"Points inside bounding box for {object_name}: {len(inside_bbox)}")

        # Assign the label to these points
        if len(inside_bbox) > 0:
            class_label = cfg["labels-list"].index(object_name)
            labels[inside_bbox] = class_label

    # Assign background label to unlabeled points
    background_label = cfg["labels-list"].index("background")
    labels[labels == -1] = background_label

    # Combine the point cloud with the labels
    labels = labels.reshape(-1, 1)
    point_cloud_with_labels = np.concatenate([point_cloud, labels], axis=1)
    print("Final labeled point cloud shape: ", point_cloud_with_labels.shape)

    return point_cloud_with_labels















def label_pointcloud_obb_original(cfg, point_cloud, bounding_boxes, threshold=0.01):
   
    # Initialize all labels as -1 (unlabeled)
    labels = np.full(len(point_cloud), -1)
    print("IN LABELING")
    print(bounding_boxes.keys())
    # Iterate through each bounding box
    for object_name, bbox in bounding_boxes.items():
        # Extract bounding box properties
        min_coords = np.array(bbox['min'])
        max_coords = np.array(bbox['max'])
        center = np.array(bbox['center'])
        quaternion = np.array(bbox['quaternion'])

        # Normalize quaternion to avoid numerical issues
        # quaternion = quaternion / np.linalg.norm(quaternion)

        # Compute rotation matrix from quaternion
        rotation = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])  # [w, x, y, z] -> [x, y, z, w]
        rotation_matrix = rotation.as_matrix()

        # Transform min and max coords to the local frame
        min_coords_local = np.dot(min_coords - center, rotation_matrix.T)
        max_coords_local = np.dot(max_coords - center, rotation_matrix.T)

        # Add threshold to expand the bounding box in the local frame
        # min_coords_local -= threshold
        # max_coords_local += threshold

        # Debugging: Print intermediate results
        print(f"Object: {object_name}")
        print(f"Min (local): {min_coords_local}, Max (local): {max_coords_local}")
        print(f"Center: {center}, Quaternion: {quaternion}")
        print(f"Rotation Matrix:\n{rotation_matrix}")

        # Transform points to the local frame of the bounding box
        points_local = np.dot(point_cloud - center, rotation_matrix.T)

        # Check which points lie inside the bounding box in the local frame
        inside_bbox = np.where(
            (points_local[:, 0] >= min_coords_local[0]) & (points_local[:, 0] <= max_coords_local[0]) &
            (points_local[:, 1] >= min_coords_local[1]) & (points_local[:, 1] <= max_coords_local[1]) &
            (points_local[:, 2] >= min_coords_local[2]) & (points_local[:, 2] <= max_coords_local[2])
        )[0]

        # Assign the label to these points
        class_label = cfg["labels-list"].index(object_name)
        print(f"Points inside {object_name}: {len(inside_bbox)}, Assigned label: {class_label}")
        labels[inside_bbox] = class_label

        # if debug == 1: 
        #     raise Exception
        # debug += 1

    # Assign background label to unlabeled points
    background_label = cfg["labels-list"].index("background")
    labels[labels == -1] = background_label

    # Combine the point cloud with the labels
    labels = labels.reshape(-1, 1)
    point_cloud_with_labels = np.concatenate([point_cloud, labels], axis=1)
    print("Final labeled point cloud shape: ", point_cloud_with_labels.shape)


    return point_cloud_with_labels

# ################## MODEL FUNCTIONS ################## # 

# A function for Xavier Initialization of MLP layers
def xavier_initialization(layer_in, layer_out):
        layer = nn.Linear(layer_in,layer_out)
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

        return layer

# A function for calculating the IoU score for segmentation
def calculate_IoU(pred, target, num_classes):
    batch_IoUs = []
    # print(pred.shape)
    pred = torch.argmax(pred, dim=2) # .view(-1)
    target = target # .view(-1)

    IoUs = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum(dim=1).float()
        union = (pred_mask | target_mask).sum(dim=1).float()

        iou = torch.where(union > 0, intersection / union, torch.tensor(float('nan')).to(intersection.device))
        IoUs.append(iou)
    
    iou_per_class = torch.stack(IoUs, dim=1)
    mean_iou = torch.nanmean(iou_per_class, dim=1)

    return mean_iou.mean()

        
# ############### QUATERNION OPERATIONS ############### # 

# Convert a 3x3 rotation matrix to a quaternion (w, x, y, z). 
def rotation_matrix_to_quaternion(R):
    q_w = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    q_x = (R[2, 1] - R[1, 2]) / (4 * q_w)
    q_y = (R[0, 2] - R[2, 0]) / (4 * q_w)
    q_z = (R[1, 0] - R[0, 1]) / (4 * q_w)
    return np.array([q_w, q_x, q_y, q_z])

# Directly from  version 1
# Takes the quaternion, degree of rotation, and the axis of rotation. Returns the rotated quaternion 
def quaternion_rotation(quaternion, angle_degrees, axis):
    # Convert angle to radians and calculate half angle
    angle_radians = math.radians(angle_degrees)
    half_angle = angle_radians / 2
    cos_half_angle = math.cos(half_angle)
    sin_half_angle = math.sin(half_angle)

    # Normalize the axis
    axis_length = math.sqrt(sum([x**2 for x in axis]))
    normalized_axis = tuple(x/axis_length for x in axis)

    # Rotation quaternion for the arbitrary axis
    rot_quaternion = (
        cos_half_angle,
        normalized_axis[0] * sin_half_angle,
        normalized_axis[1] * sin_half_angle,
        normalized_axis[2] * sin_half_angle,
    )

    # Quaternion multiplication (rot_quaternion * quaternion)
    w1, x1, y1, z1 = rot_quaternion
    w2, x2, y2, z2 = quaternion
    rotated_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    rotated_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    rotated_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    rotated_z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return (rotated_w, rotated_x, rotated_y, rotated_z)

# ############# CAMERA RELATED OPERTATION  ############# # 
# def cam_to_world(point_cloud, cam_frame):
#         t = cam_frame.getPosition() 
#         R = cam_frame.getRotationMatrix()
#         points_camera_frame = point_cloud

#         # Add homogeneous coordinates for the points in camera frame
#         points_camera_frame_homogeneous = np.hstack((points_camera_frame, np.ones((points_camera_frame.shape[0], 1))))
#         # Transformation matrix (combining rotation and translation)
#         transformation_matrix = np.vstack((np.hstack((R, t.reshape(-1, 1))), np.array([0, 0, 0, 1])))
#         # Transform all points to world frame
#         points_world_frame_homogeneous = np.dot(transformation_matrix, points_camera_frame_homogeneous.T).T
#         # Extract the 3D coordinates in the world frame
#         points_world_frame = points_world_frame_homogeneous[:, :3]
#         return points_world_frame

def get_rgb_images(C, camera_names=["camera1", "camera2", "camera3"], cam_frame = None, filter_pcl= False):
    images = []
    depthes = []
    pcls = []

    table_threshold = C.getFrame("table").getPosition()[2] + C.getFrame("table").info()["size"][2]/2  + 0.005
    robot_threshold = C.getFrame("l_gripper").getPosition()[2] - 0.04

    for name in camera_names:
        cam = ry.CameraView(C)
        cam.setCamera(name)
        rgb, depth = cam.computeImageAndDepth(C)
        print(f"rgb size: {rgb.shape}")
        images.append(rgb)
        depthes.append(depth)
        pcl = ry.depthImage2PointCloud(depth, cam.getFxycxy())
        # pcl = cam_to_world(pcl.reshape(-1, 3), cam_frame)
        print(f"pcl size: {pcl.shape}")
        
        if filter_pcl == True:
            table_mask = pcl[:,2] > table_threshold
            pcl = pcl[table_mask]
            robot_mask = pcl[:,2] < robot_threshold
            pcl = pcl[robot_mask]
            table_mask = pcl[:,1] < 0.6
            pcl = pcl[table_mask]
            table_mask = pcl[:,1] > 0
            pcl = pcl[table_mask]

        pcls.append(pcl) 
    
    return images, depthes, pcls

def cam_to_world(point_cloud, cam_frame):
    """
    Transforms a point cloud from the camera frame to the world frame, 
    considering both position and orientation of the camera.

    Parameters:
        point_cloud (numpy.ndarray): Nx3 array of points in the camera frame.
        cam_frame (object): Camera frame object with methods:
            - getPosition() -> numpy.ndarray: Returns the camera position as a 1x3 array.
            - getRotationMatrix() -> numpy.ndarray: Returns the 3x3 rotation matrix of the camera.

    Returns:
        numpy.ndarray: Nx3 array of points transformed to the world frame.
    """
    # Get camera position and rotation matrix
    t = cam_frame.getPosition()  # Camera position in world frame (1x3 array)
    R = cam_frame.getRotationMatrix()  # Camera rotation matrix (3x3)

    # Ensure the point cloud has homogeneous coordinates
    points_camera_frame_homogeneous = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Construct the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R  # Set the rotation part
    transformation_matrix[:3, 3] = t   # Set the translation part

    # Transform points to world frame
    points_world_frame_homogeneous = np.dot(transformation_matrix, points_camera_frame_homogeneous.T).T
    points_world_frame = points_world_frame_homogeneous[:, :3]

    return points_world_frame

# ## HELPER FUNCTIONS FOR GRASPING AND MANIPULATION ## # 
def find_grasping_part(C, selected_tool, tool_waypoint):
    if selected_tool in ["hammer", "L-ruler", "spatula", "screwdriver"]:
        possible_parts = [selected_tool + '-base', selected_tool + '-head']
    elif selected_tool in ["book", "thin-stick", "ball"]:
        possible_parts = [selected_tool + '-base']
    elif selected_tool in ["ring", "pipe-hammer"]:
        possible_parts = [selected_tool + '-base', selected_tool + '-head1',  selected_tool + '-head2',  selected_tool + '-head']
    elif selected_tool in ["U-tool", "asymmetric-L-ruler", ]:
        possible_parts = [selected_tool + '-base', selected_tool + '-head',  selected_tool + '-head2']
    elif selected_tool in ["fork-spatula"]:
        possible_parts = [selected_tool + '-base', selected_tool + '-head', selected_tool + '-tine-1', selected_tool + '-tine-2',selected_tool + '-tine-3', selected_tool + '-tine-0']
    else: 
        raise Exception("The name is wrong")
    closest_part = None
    min_distance = float('inf')

    for part in possible_parts:
        print("****** part: ", part)
        center = C.getFrame(part).getPosition()
        size = C.getFrame(part).info()['size'][:3]

        min_coords = np.array(center) - np.array(size) / 2
        max_coords = np.array(center) + np.array(size) / 2

        # Check if waypoint is inside the bounding box
        is_inside = np.all(tool_waypoint[:3] >= min_coords) and np.all(tool_waypoint[:3] <= max_coords)
        if is_inside:
            return part  # If inside, return immediately

        # Compute distance to the closest point on the bounding box
        closest_point = np.clip(tool_waypoint[:3], min_coords, max_coords)
        distance = np.linalg.norm(tool_waypoint[:3] - closest_point)

        if distance < min_distance:
            min_distance = distance
            closest_part = part

    return closest_part

def reparent_tool(C, selected_tool, selected_part, other_tools):
    for tool in other_tools: 
        C.getFrame(tool + '-base').setJoint(ry.JT.rigid)

    if "base" in selected_part:
        C.getFrame(selected_part).setJoint(ry.JT.rigid)
        print("I added the joint to the base.")
    elif "head" in selected_part:
        C.getFrame(selected_part).unLink()
        C.getFrame(selected_part).setParent(C.getFrame("table"), True)

        C.getFrame(selected_tool + '-base').unLink()
        C.getFrame(selected_tool + '-base').setParent(C.getFrame(selected_part), True)

        C.getFrame(selected_part).setJoint(ry.JT.rigid)

# Grasp Score Calculation over stability: 
def compute_grasp_stability(position_histories, goal_position, goal_tolerance=0.05, fail_threshold = 0.01):
    stability_weight = 0.3

    displacements = []
    for position in position_histories: 
        displacements.append(np.linalg.norm(position[-1] - position[0]))
    
    displacement_score = np.clip(np.mean(displacements) / fail_threshold, 0, 1)
    
    consistency_score = np.exp(-np.nan_to_num(np.std(displacements)))
    velocities = [np.diff(positions, axis=0) for positions in position_histories]
    velocity_magnitudes = np.array([np.linalg.norm(vel, axis=1).mean() for vel in velocities])

    max_velocity = np.max(velocity_magnitudes) if np.max(velocity_magnitudes) > 0 else 1
    # smoothness_score = 1 - (np.std(velocity_magnitudes) / max_velocity)
    
    # Smoothness Score Update because of nan: 
    if np.all(velocity_magnitudes == 0):  
        smoothness_score = 1  # If no movement, assume perfectly smooth
    else:
        max_velocity = np.max(velocity_magnitudes) if np.max(velocity_magnitudes) > 0 else 1
        smoothness_score = 1 - (np.nan_to_num(np.std(velocity_magnitudes)) / max_velocity)

    initial_distance = np.mean([
        np.linalg.norm(positions[0] - goal_position) for positions in position_histories
    ])
    final_distance = np.mean([
        np.linalg.norm(positions[-1] - goal_position) for positions in position_histories
    ])

    improvement = (initial_distance - final_distance) / (initial_distance + 1e-6)  # How much closer it got
    goal_penalty = 1 / (1 + np.exp((final_distance - goal_tolerance) * 10))  # Penalize stopping too far

    goal_score = improvement * goal_penalty  # Final goal contribution

    # Final stability score combines all factors
    stability_score = (displacement_score * 
                       (consistency_score ** stability_weight) * 
                       (smoothness_score ** stability_weight) * 
                       goal_score)
    
    if np.isnan(stability_score):
        raise ValueError(f"NaN detected in final stability_score: {stability_score}\n"
                         f"Displacement Score: {displacement_score}\n"
                         f"Consistency Score: {consistency_score}\n"
                         f"Smoothness Score: {smoothness_score}\n"
                         f"Goal Score: {goal_score}")

    return np.clip(stability_score, 0, 1)
    # if all(d < fail_threshold for d in displacements):
    #     return 0
    
    # velocities = [np.diff(positions, axis=0) for positions in position_histories]
    # velocity_variances = [np.var(vel, axis=0) for vel in velocities]

    # instability_factor = 0
    # for i in range(len(velocity_variances)):
    #     for j in range(i + 1, len(velocity_variances)):  
    #         diff = np.linalg.norm(velocity_variances[i] - velocity_variances[j]) 
    #         instability_factor += diff

    # accelerations = [np.diff(vel, axis=0) for vel in velocities]
    # acceleration_variances = [np.var(acc, axis=0) for acc in accelerations]
    
    # smoothness_factor = 0
    # for i in range(len(acceleration_variances)):
    #     for j in range(i + 1, len(acceleration_variances)): 
    #         diff = np.linalg.norm(acceleration_variances[i] - acceleration_variances[j])  # Compute difference
    #         smoothness_factor += diff
    
    # stability_score = 1 / (1 + instability_factor + smoothness_factor)
    # return stability_score

# #################### NOT READY #################### # 
def IK_old(C, qHome, target):
    q0 = C.getJointState()
    komo = ry.KOMO(C, 2, 10, 2, False)
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q0)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.sos)
    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', target], ry.OT.eq, [1e2])
    
    ret = ry.NLP_Solver(komo.nlp(), verbose=0) .solve()
    return [komo.getPath()[0], ret]

def IK(C, target, step, phase = 10, ta = [0.0,0.0,0.0], interaction = False):
    q0 = C.getJointState()
    komo = ry.KOMO(C, step, phase, 1, False) 
    komo.addControlObjective([], 0, 1e-0)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q0) 
    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', target], ry.OT.eq, [1e1], target=ta)
    komo.addObjective([], ry.FS.quaternionDiff, ['l_gripper', target], ry.OT.eq, [1e0])
    komo.addObjective([], ry.FS.vectorZ, ['l_gripper'], ry.OT.sos, [1e-1], target=[0, 0, -1])
    
    if interaction: 
        v_eps = 0.00  
        komo.addObjective([],
                        ry.FS.position, ['l_gripper'],
                        ry.OT.ineq,
                        scale=[0,0,1], target=[0,0,v_eps],
                        order=1)

    ret = ry.NLP_Solver(komo.nlp(), verbose=0) .solve()
    
    return [komo.getPath()[0], ret]


def botop_move(C, bot, q):
    bot.moveTo(q) 
    while bot.getTimeToEnd()>0:
        bot.sync(C, .1)

def botop_move_stability(C, bot, q, selected_tool):
    # print(f"###DEBUG### selected tool: {selected_tool}")
    head_positions = []
    base_positions = []

    bot.moveTo(q) 
    while bot.getTimeToEnd()>0:
        if selected_tool in ["hammer", "screwdriver", "ring","L-ruler", "spatula"]:
            head_positions.append(C.getFrame(selected_tool + "-head").getPosition())
        base_positions.append(C.getFrame(selected_tool + "-base").getPosition())

        bot.sync(C, .1)
    
    # print("The position changes: ")
    # print(head_positions)
    # print(base_positions)
    
    # print(f"The grasp score: {grasp_score}")

    return head_positions, base_positions

def botop_move_target(C, bot, q, target_history, environment):

    bot.moveTo(q) 
    while bot.getTimeToEnd()>0:
        # print(target_history.keys())
        for key in target_history.keys():
            target_history[key].append(C.getFrame(key).getPosition())
        # target_history["end_point_1"].append(C.getFrame("end-point-1").getPosition())
        # target_history["end_point_2"].append(C.getFrame("end-point-2").getPosition())
        # target_history["center"].append(C.getFrame("lifting-obj").getPosition())
        # print(C.getCollisions())
        bot.sync(C, .1)
    
    # print("The position changes: ")
    # print(head_positions)
    # print(base_positions)

    return target_history

def manipulation_with_komo(C):
    qHome = C.getJointState()

    bot = ry.BotOp(C, useRealRobot=False)
    bot.home(C) 


    bot.gripperMove(ry._left, width = 0.08, speed = 0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)

    path, ret = IK(C, target="tool-waypoint", step=1, phase=1)
    botop_move(C,bot,path)

    bot.gripperClose(ry._left)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)

    # bot.home(C)

    path, ret = IK(C, target="initial-waypoint", step=2)
    botop_move(C,bot,path)

    path, ret = IK(C, target="goal-waypoint", step=2)
    botop_move(C,bot,path)


    del bot 

def AdaptiveManipulationWithKOMO(C, selected_tool, environment):
    qHome = C.getJointState()
    # The before grasping can stay same for adaptive manipulation, it works well 
    C.addFrame("before-grasping")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(C.getFrame("tool-waypoint").getPosition() + [0.0, 0.0, 0.05])\
            .setQuaternion(C.getFrame("tool-waypoint").getQuaternion())
    
    tool_waypoint = C.getFrame("tool-waypoint").getPosition()
    initial_waypoint = C.getFrame("initial-waypoint").getPosition()
    goal_waypoint = C.getFrame("goal-waypoint").getPosition()


    # Post grasp
    # if abs(tool_waypoint[2] - initial_waypoint[2])< ...:
    #     ... 

    # The movement with Bot-OP
    bot = ry.BotOp(C, useRealRobot=False)
    bot.home(C)

    # Open Gripper: 
    bot.gripperMove(ry._left, width = 0.08, speed = 0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)

    path, ret = IK(C, target="before-grasping", step=1, phase=1)
    botop_move(C,bot,path)

    path, ret = IK(C, target="tool-waypoint", step=1, phase=1)
    botop_move(C,bot,path)

    bot.gripperMove(ry._left, width=0.005, speed=0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)

# This is a good function for testing :) 25.07
def ManipulationWithKOMO_minigolf(C, selected_tool, environment):
    qHome = C.getJointState()
    C.addFrame("before-grasping")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(C.getFrame("tool-waypoint").getPosition() + [0.0, 0.0, 0.05])\
            .setQuaternion(C.getFrame("tool-waypoint").getQuaternion())
    
    C.addFrame("after-grasping")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(C.getFrame("tool-waypoint").getPosition() + [0.0, 0.0, 0.1])\
            .setQuaternion(C.getFrame("tool-waypoint").getQuaternion())
    

    
    before_initial_position = [C.getFrame("tool-waypoint").getPosition()[0], C.getFrame("tool-waypoint").getPosition()[1], C.getFrame("initial-waypoint").getPosition()[2]]
    C.addFrame("before-initial")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(before_initial_position)\
            .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())
    
    mid_position = (C.getFrame("before-initial").getPosition() +C.getFrame("initial-waypoint").getPosition())/2
    # mid_position[2] = C.getFrame("initial-waypoint").getPosition()[2]
    C.addFrame("mid-initial")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(mid_position)\
            .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())
    
    initial_up_position = [C.getFrame("mid-initial").getPosition()[0], C.getFrame("initial-waypoint").getPosition()[1], C.getFrame("before-initial").getPosition()[2]]
    C.addFrame("initial-up")\
        .setShape(ry.ST.marker, [.1])\
        .setPosition(initial_up_position)\
        .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())
    
    
    mid_goal_position = (C.getFrame("initial-waypoint").getPosition() +C.getFrame("goal-waypoint").getPosition())/2
    # mid_position[2] = C.getFrame("initial-waypoint").getPosition()[2]
    C.addFrame("goal-mid")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(mid_goal_position)\
            .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())
    
    target_history = environment.target_setup()
    C.view()

    bot = ry.BotOp(C, useRealRobot=False)
    bot.home(C)

    # Open Gripper: 
    bot.gripperMove(ry._left, width = 0.08, speed = 0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)

    path, ret = IK(C, target="before-grasping", step=1, phase=1)
    botop_move(C,bot,path)

    path, ret = IK(C, target="tool-waypoint", step=1, phase=1)
    botop_move(C,bot,path)

    bot.gripperMove(ry._left, width=0.005, speed=0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)
    
    # C.attach(selected_part, "l_gripper")
    
    path, ret = IK(C, target="after-grasping", step=1, phase=1)
    botop_move(C,bot,path)

    
    bot.home(C)
    # path, ret = IK(C, target="mid-point", step=2)
    # head_position_half, base_position_half  = botop_move_stability(C,bot,path, selected_tool)
    head_positions_half, base_positions_half = [], []
    path, ret = IK(C, target="before-initial", step=10, interaction=True)
    head_positions_half, base_positions_half  = botop_move_stability(C,bot,path, selected_tool)
    
    head_positions_mid, base_positions_mid = [], []
    path, ret = IK(C, target="mid-initial", step=10, interaction=True)
    head_positions_mid, base_positions_mid  = botop_move_stability(C,bot,path, selected_tool)

    head_positions_up, base_positions_up = [], []
    path, ret = IK(C, target="initial-up", step=10, interaction=True)
    head_positions_up, base_positions_up  = botop_move_stability(C,bot,path, selected_tool)

    path, ret = IK(C, target="initial-waypoint", step=5, interaction=True)
    head_positions_in, base_positions_in  = botop_move_stability(C,bot,path, selected_tool)
    
    head_positions = head_positions_half + head_positions_mid + head_positions_up + head_positions_in 
    base_positions = base_positions_half + base_positions_mid + base_positions_up + base_positions_in 
    position_history = [base_positions]
    if selected_tool in ["hammer", "screwdriver", "ring","L-ruler", "spatula"]:
        position_history.append(head_positions)

    grasp_score = compute_grasp_stability(position_history, C.getFrame("initial-waypoint").getPosition()) 
    print(f"Grasp Score: {grasp_score}")

    # path, ret = IK(C, target="mid-waypoint", step=2)
    # target_history = botop_move_target(C,bot,path,target_history, environment)
    path, ret = IK(C, target="goal-mid", step=2, interaction=True)
    target_history = botop_move_target(C,bot,path,target_history, environment)
    
    path, ret = IK(C, target="goal-waypoint", step=2, interaction=True)
    target_history = botop_move_target(C,bot,path,target_history, environment)
    
    task_score = environment.compute_task_score(target_history)
    print(f"Task Score: {task_score}")

    score = grasp_score + task_score
    print(f"All Score: {score}")
    score_info = {
        "score" : score,
        "grasp_score": grasp_score,
        "task_score": task_score
    }
    return score_info
    
def ManipulationWithKOMO_lifting(C, selected_tool, environment):
    qHome = C.getJointState()
    C.addFrame("before-grasping")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(C.getFrame("tool-waypoint").getPosition() + [0.0, 0.0, 0.05])\
            .setQuaternion(C.getFrame("tool-waypoint").getQuaternion())
    
    C.addFrame("after-grasping")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(C.getFrame("tool-waypoint").getPosition() + [0.0, 0.0, 0.1])\
            .setQuaternion(C.getFrame("tool-waypoint").getQuaternion())
    
    mid_position = (C.getFrame("initial-waypoint").getPosition() +C.getFrame("goal-waypoint").getPosition())/2
    # mid_position[2] = C.getFrame("initial-waypoint").getPosition()[2]
    # C.addFrame("mid-waypoint")\
    #         .setShape(ry.ST.marker, [.1])\
    #         .setPosition(mid_position)\
    #         .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())
    
    before_initial_position = [C.getFrame("tool-waypoint").getPosition()[0],C.getFrame("tool-waypoint").getPosition()[1], C.getFrame("initial-waypoint").getPosition()[2]]
    C.addFrame("before-initial")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(before_initial_position)\
            .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())
    
    target_history = environment.target_setup()
    C.view()

    bot = ry.BotOp(C, useRealRobot=False)
    bot.home(C)

    # Open Gripper: 
    bot.gripperMove(ry._left, width = 0.08, speed = 0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)

    path, ret = IK(C, target="before-grasping", step=1, phase=1)
    botop_move(C,bot,path)

    path, ret = IK(C, target="tool-waypoint", step=1, phase=1)
    botop_move(C,bot,path)

    bot.gripperMove(ry._left, width=0.005, speed=0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)
    
    # C.attach(selected_part, "l_gripper")
    
    path, ret = IK(C, target="after-grasping", step=1, phase=1)
    botop_move(C,bot,path)

    
    # bot.home(C)
    # path, ret = IK(C, target="mid-point", step=2)
    # head_position_half, base_position_half  = botop_move_stability(C,bot,path, selected_tool)
    
    path, ret = IK(C, target="before-initial", step=10)
    head_positions_half, base_positions_half  = botop_move_stability(C,bot,path, selected_tool)
    
    path, ret = IK(C, target="initial-waypoint", step=10)
    head_positions_in, base_positions_in  = botop_move_stability(C,bot,path, selected_tool)
    
    head_positions = head_positions_half + head_positions_in
    base_positions = base_positions_half + base_positions_in
    position_history = [base_positions]
    if selected_tool in ["hammer", "screwdriver", "ring","L-ruler", "spatula"]:
        position_history.append(head_positions)
    grasp_score = compute_grasp_stability(position_history, C.getFrame("initial-waypoint").getPosition()) 
    print(f"Grasp Score: {grasp_score}")

    # path, ret = IK(C, target="mid-waypoint", step=2)
    # target_history = botop_move_target(C,bot,path,target_history, environment)
    
    path, ret = IK(C, target="goal-waypoint", step=2)
    # botop_move(C,bot,path)
    target_history = botop_move_target(C,bot,path,target_history, environment)
    
    task_score = environment.compute_task_score(target_history)
    print(f"Task Score: {task_score}")

    score = grasp_score + task_score
    print(f"All Score: {score}")
    score_info = {
        "score" : score,
        "grasp_score": grasp_score,
        "task_score": task_score
    }
    return score_info

def ManipulationWithKOMO_hammering(C, selected_tool, environment, env_class=None):
    qHome = C.getJointState()
    home = C.getFrame("l_gripper").getPosition()
    if env_class is not None: 
        data_path = os.path.join("hammering-all-steps", "start.g")
        env_class.save_environment_path(data_path)

    C.addFrame("before-grasping")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(C.getFrame("tool-waypoint").getPosition() + [0.0, 0.0, 0.05])\
            .setQuaternion(C.getFrame("tool-waypoint").getQuaternion())
    
    C.addFrame("after-grasping")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(C.getFrame("tool-waypoint").getPosition() + [0.0, 0.0, 0.1])\
            .setQuaternion(C.getFrame("tool-waypoint").getQuaternion())
    
    mid_position = (C.getFrame("initial-waypoint").getPosition() +C.getFrame("goal-waypoint").getPosition())/2
    # mid_position[2] = C.getFrame("initial-waypoint").getPosition()[2]
    # C.addFrame("mid-waypoint")\
    #         .setShape(ry.ST.marker, [.1])\
    #         .setPosition(mid_position)\
    #         .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())
    
    before_initial_position = [home[0],home[1], C.getFrame("initial-waypoint").getPosition()[2]]
    C.addFrame("before-initial")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(before_initial_position)\
            .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())
    
    target_history = environment.target_setup()
    C.view()

    bot = ry.BotOp(C, useRealRobot=False)
    bot.home(C)

    # Open Gripper: 
    bot.gripperMove(ry._left, width = 0.08, speed = 0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)

    path, ret = IK(C, target="before-grasping", step=1, phase=1)
    botop_move(C,bot,path)

    path, ret = IK(C, target="tool-waypoint", step=1, phase=1)
    botop_move(C,bot,path)

    if env_class is not None: 
        data_path = os.path.join("hammering-all-steps", "grasping.g")
        env_class.save_environment_path(data_path)

    bot.gripperMove(ry._left, width=0.005, speed=0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)
    
    # C.attach(selected_part, "l_gripper")
    
    path, ret = IK(C, target="after-grasping", step=1, phase=1)
    botop_move(C,bot,path)
    
    
    bot.home(C)
    # path, ret = IK(C, target="mid-point", step=2)
    # head_position_half, base_position_half  = botop_move_stability(C,bot,path, selected_tool)
    
    head_positions_half, base_positions_half = [], []
    path, ret = IK(C, target="before-initial", step=10)
    head_positions_half, base_positions_half  = botop_move_stability(C,bot,path, selected_tool)
    
    if env_class is not None: 
        data_path = os.path.join("hammering-all-steps", "after_grasping.g")
        env_class.save_environment_path(data_path)

    path, ret = IK(C, target="initial-waypoint", step=10)
    head_positions_in, base_positions_in  = botop_move_stability(C,bot,path, selected_tool)
    
    if env_class is not None: 
        data_path = os.path.join("hammering-all-steps", "at_initial.g")
        env_class.save_environment_path(data_path)

    head_positions = head_positions_half + head_positions_in
    base_positions = base_positions_half + base_positions_in
    position_history = [base_positions]
    if selected_tool in ["hammer", "screwdriver", "ring","L-ruler", "spatula"]:
        position_history.append(head_positions)
    grasp_score = compute_grasp_stability(position_history, C.getFrame("initial-waypoint").getPosition()) 
    print(f"Grasp Score: {grasp_score}")

    # path, ret = IK(C, target="mid-waypoint", step=2)
    # target_history = botop_move_target(C,bot,path,target_history, environment)
    
    path, ret = IK(C, target="goal-waypoint", step=2)
    # botop_move(C,bot,path)
    target_history = botop_move_target(C,bot,path,target_history, environment)

    if env_class is not None: 
        data_path = os.path.join("hammering-all-steps", "after_goal.g")
        env_class.save_environment_path(data_path)
    
    task_score = environment.compute_task_score(target_history)
    print(f"Task Score: {task_score}")

    score = grasp_score + task_score
    print(f"All Score: {score}")
    score_info = {
        "score" : score,
        "grasp_score": grasp_score,
        "task_score": task_score
    }
    return score_info
# This is the KOMO that used in the data collection
def ManipulationWithKOMO_reaching(C, selected_tool, environment):
    qHome = C.getJointState()
    HomePose = C.getFrame("l_gripper").getPosition()
    C.addFrame("before-grasping")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(C.getFrame("tool-waypoint").getPosition() + [0.0, 0.0, 0.05])\
            .setQuaternion(C.getFrame("tool-waypoint").getQuaternion())
    
    C.addFrame("after-grasping")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(C.getFrame("tool-waypoint").getPosition() + [0.0, 0.0, 0.1])\
            .setQuaternion(C.getFrame("tool-waypoint").getQuaternion())
    
    mid_position = (C.getFrame("initial-waypoint").getPosition() +C.getFrame("goal-waypoint").getPosition())/2
    # mid_position[2] = C.getFrame("initial-waypoint").getPosition()[2]
    # C.addFrame("mid-waypoint")\
    #         .setShape(ry.ST.marker, [.1])\
    #         .setPosition(mid_position)\
    #         .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())
    
    bef_init_x = (C.getFrame("l_gripper").getPosition()[0]*5 + C.getFrame("initial-waypoint").getPosition()[0])/6
    bef_init_y = (C.getFrame("l_gripper").getPosition()[1]*5 + C.getFrame("initial-waypoint").getPosition()[1])/6
    before_initial_position = [bef_init_x,bef_init_y, C.getFrame("initial-waypoint").getPosition()[2]]
    C.addFrame("before-initial")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(before_initial_position)\
            .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())

    bef_init_x2 = (C.getFrame("l_gripper").getPosition()[0]*2 + C.getFrame("initial-waypoint").getPosition()[0]*4)/6
    bef_init_y2 = (C.getFrame("l_gripper").getPosition()[1]*2 + C.getFrame("initial-waypoint").getPosition()[1]*4)/6
    before_initial_position2 = [bef_init_x2,bef_init_y2, C.getFrame("initial-waypoint").getPosition()[2]]
    C.addFrame("before-initial2")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(before_initial_position2)\
            .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())
    
    target_history = environment.target_setup()
    C.view()

    bot = ry.BotOp(C, useRealRobot=False)
    bot.home(C)

    # Open Gripper: 
    bot.gripperMove(ry._left, width = 0.08, speed = 0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)

    path, ret = IK(C, target="before-grasping", step=1, phase=1)
    botop_move(C,bot,path)

    path, ret = IK(C, target="tool-waypoint", step=1, phase=1)
    botop_move(C,bot,path)

    bot.gripperMove(ry._left, width=0.005, speed=0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)
    
    # C.attach(selected_part, "l_gripper")
    
    path, ret = IK(C, target="after-grasping", step=1, phase=1)
    botop_move(C,bot,path)

    
    bot.home(C)
    # path, ret = IK(C, target="mid-point", step=2)
    # head_position_half, base_position_half  = botop_move_stability(C,bot,path, selected_tool)
    
    path, ret = IK(C, target="before-initial", step=10)
    head_positions_half, base_positions_half  = botop_move_stability(C,bot,path, selected_tool)

    path, ret = IK(C, target="before-initial2", step=10)
    head_positions_half, base_positions_half  = botop_move_stability(C,bot,path, selected_tool)
    
    path, ret = IK(C, target="initial-waypoint", step=10)
    head_positions_in, base_positions_in  = botop_move_stability(C,bot,path, selected_tool)
    
    head_positions = head_positions_half + head_positions_in
    base_positions = base_positions_half + base_positions_in
    position_history = [base_positions]
    if selected_tool in ["hammer", "screwdriver", "ring","L-ruler", "spatula"]:
        position_history.append(head_positions)
    grasp_score = compute_grasp_stability(position_history, C.getFrame("initial-waypoint").getPosition()) 
    print(f"Grasp Score: {grasp_score}")

    # path, ret = IK(C, target="mid-waypoint", step=2)
    # target_history = botop_move_target(C,bot,path,target_history, environment)
    
    path, ret = IK(C, target="goal-waypoint", step=2)
    # botop_move(C,bot,path)
    target_history = botop_move_target(C,bot,path,target_history, environment)
    
    task_score = environment.compute_task_score(target_history)
    print(f"Task Score: {task_score}")

    score = grasp_score + task_score
    print(f"All Score: {score}")
    score_info = {
        "score" : score,
        "grasp_score": grasp_score,
        "task_score": task_score
    }
    return score_info
# This is the KOMO that used in the data collection
def ManipulationWithKOMO_original(C, selected_tool, environment):
    qHome = C.getJointState()
    C.addFrame("before-grasping")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(C.getFrame("tool-waypoint").getPosition() + [0.0, 0.0, 0.05])\
            .setQuaternion(C.getFrame("tool-waypoint").getQuaternion())
    
    C.addFrame("after-grasping")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(C.getFrame("tool-waypoint").getPosition() + [0.0, 0.0, 0.1])\
            .setQuaternion(C.getFrame("tool-waypoint").getQuaternion())
    
    mid_position = (C.getFrame("initial-waypoint").getPosition() +C.getFrame("goal-waypoint").getPosition())/2
    # mid_position[2] = C.getFrame("initial-waypoint").getPosition()[2]
    # C.addFrame("mid-waypoint")\
    #         .setShape(ry.ST.marker, [.1])\
    #         .setPosition(mid_position)\
    #         .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())
    
    before_initial_position = [C.getFrame("tool-waypoint").getPosition()[0],C.getFrame("tool-waypoint").getPosition()[1], C.getFrame("initial-waypoint").getPosition()[2]]
    C.addFrame("before-initial")\
            .setShape(ry.ST.marker, [.1])\
            .setPosition(before_initial_position)\
            .setQuaternion(C.getFrame("initial-waypoint").getQuaternion())
    
    target_history = environment.target_setup()
    C.view()

    bot = ry.BotOp(C, useRealRobot=False)
    bot.home(C)

    # Open Gripper: 
    bot.gripperMove(ry._left, width = 0.08, speed = 0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)

    path, ret = IK(C, target="before-grasping", step=1, phase=1)
    botop_move(C,bot,path)

    path, ret = IK(C, target="tool-waypoint", step=1, phase=1)
    botop_move(C,bot,path)

    bot.gripperMove(ry._left, width=0.005, speed=0.1)
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)
    
    # C.attach(selected_part, "l_gripper")
    
    path, ret = IK(C, target="after-grasping", step=1, phase=1)
    botop_move(C,bot,path)

    
    # bot.home(C)
    # path, ret = IK(C, target="mid-point", step=2)
    # head_position_half, base_position_half  = botop_move_stability(C,bot,path, selected_tool)
    
    path, ret = IK(C, target="before-initial", step=10)
    head_positions_half, base_positions_half  = botop_move_stability(C,bot,path, selected_tool)
    
    path, ret = IK(C, target="initial-waypoint", step=10)
    head_positions_in, base_positions_in  = botop_move_stability(C,bot,path, selected_tool)
    
    head_positions = head_positions_half + head_positions_in
    base_positions = base_positions_half + base_positions_in
    position_history = [base_positions]
    if selected_tool in ["hammer", "screwdriver", "ring","L-ruler", "spatula"]:
        position_history.append(head_positions)
    grasp_score = compute_grasp_stability(position_history, C.getFrame("initial-waypoint").getPosition()) 
    print(f"Grasp Score: {grasp_score}")

    # path, ret = IK(C, target="mid-waypoint", step=2)
    # target_history = botop_move_target(C,bot,path,target_history, environment)
    
    path, ret = IK(C, target="goal-waypoint", step=2)
    # botop_move(C,bot,path)
    target_history = botop_move_target(C,bot,path,target_history, environment)
    
    task_score = environment.compute_task_score(target_history)
    print(f"Task Score: {task_score}")

    score = grasp_score + task_score
    print(f"All Score: {score}")
    score_info = {
        "score" : score,
        "grasp_score": grasp_score,
        "task_score": task_score
    }
    return score_info

# ########### ROTATION RELATED FUNCTIONS ############ # 

# This function is used for rotation of cameras
def euler_to_quaternion(euler_list):
    combined_quaternion = np.array([0, 0, 0, 1]) # --> [x, y, z, w]

    for euler in euler_list:

        angle = euler[0]
        axis = np.array(euler[1:4])
        axis = axis / np.linalg.norm(axis)

        rotation = R.from_rotvec(np.deg2rad(angle) * axis)
        quaternion = rotation.as_quat() 

        combined_rotation = R.from_quat(combined_quaternion) * R.from_quat(quaternion)
        combined_quaternion = combined_rotation.as_quat()
    
    return [combined_quaternion[3], combined_quaternion[0], combined_quaternion[1], combined_quaternion[2]]

    
# For testing the model, graph creation from the point cloud
def getEnvironmentGraph(pc, radius):
    pc = farthest_point_sampling(pc, 4096)

    # Directly same with the grasp construction in Dataset.py
    pc_tensor = torch.tensor(pc, dtype=torch.float)
    edge_index = edge_index = radius_graph(pc_tensor, r=radius)
    pc_graph = Data(x=pc_tensor, edge_index=edge_index)

    return pc_graph


# --- A small post processing ---
def grasping_post_process(parameters):
    grasp_quaternion = post_quaternion(parameters["tool-qua"], parameters["predicted-qua"])
    grasp_position = post_posititon(parameters["tool-pc"], parameters["predicted-pos"])
    is_inside_tool = is_inside_pointcloud(parameters["tool-pc"], parameters["predicted-pos"])
    print(f"**DEBUG** is inside: {is_inside_tool}")
    if True:
        grasp_position =  fix_grasp_pos(parameters["tool-pc"], parameters["predicted-pos"])
        grasp_position = shift_toward_handle_center(grasp_position, parameters["tool-handle-center"])
    return grasp_quaternion, grasp_position


def post_process_waypoints_for_tasks(
    task,
    C,
    selected_tool,
    selected_part,
    best_wp,
    tool_points,
    target_center,
    lifting_side_fix,
    hammering_flag,
):
    if task in ["minigolf", "pushing"]:
        tool_quaternion = C.getFrame(selected_tool + "-base").getQuaternion()
        grasp_quaternion, grasp_position = grasping_post_process(
            {
                "tool-qua": tool_quaternion,
                "tool-handle-center": C.getFrame(selected_tool + "-base").getPosition(),
                "predicted-qua": best_wp["qua"][0],
                "predicted-pos": best_wp["pos"][0],
                "tool-pc": tool_points,
            }
        )
        best_wp["qua"][0] = grasp_quaternion
        best_wp["pos"][0] = grasp_position
        best_wp["pos"][0][2] = C.getFrame(selected_tool + "-base").getPosition()[2]

    elif task == "lifting":
        tool_quaternion = C.getFrame(selected_tool + "-base").getQuaternion()
        print(f"Tool Quaternion: {tool_quaternion}")
        if "base" in selected_part:
            tool_handle_center = C.getFrame(selected_tool + "-base").getPosition()
        elif "head" in selected_part:
            tool_handle_center = C.getFrame(selected_tool + "-head").getPosition()
        else:
            raise Exception

        if lifting_side_fix:
            best_wp["pos"][1][0] = -best_wp["pos"][1][0]
            best_wp["pos"][2][0] = -best_wp["pos"][2][0]
            best_wp["qua"][1] = rotate_quat_yaw_180(best_wp["qua"][1])
            best_wp["qua"][2] = rotate_quat_yaw_180(best_wp["qua"][2])

        def as_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x, dtype=np.float64)

        p1 = as_np(best_wp["pos"][1])
        tc = as_np(C.getFrame("lifting-obj").getPosition())
        best_wp["pos"][1] = (p1 + 0.3 * (tc - p1)).tolist()

        p2 = as_np(best_wp["pos"][2])
        best_wp["pos"][2] = (p2 + 0.3 * (tc - p2)).tolist()

        grasp_quaternion, grasp_position = grasping_post_process(
            {
                "tool-qua": tool_quaternion,
                "tool-handle-center": tool_handle_center,
                "predicted-qua": best_wp["qua"][0],
                "predicted-pos": best_wp["pos"][0],
                "tool-pc": tool_points,
            }
        )
        best_wp["qua"][0] = grasp_quaternion
        best_wp["qua"][0] = tool_quaternion
        print("AAA")
        best_wp["pos"][0] = grasp_position
        best_wp["pos"][0][2] = C.getFrame(selected_tool + "-base").getPosition()[2]

        best_wp["qua"][1] = rotate_quat_yaw_180(best_wp["qua"][1])
        best_wp["qua"][2] = rotate_quat_yaw_180(best_wp["qua"][2])

        manipulation_height = best_wp["pos"][2][2] - best_wp["pos"][1][2]
        if manipulation_height < 0.2:
            print(f"I predicted too short lifting: {manipulation_height}")
            best_wp["pos"][2][2] += 0.1

        target_height_difference = target_center[2] - best_wp["pos"][1][2]
        print(
            f"**DEBUG** the difference between initial and target center: {target_height_difference}"
        )
        if target_height_difference < 0.06:
            best_wp["pos"][1][2] -= 0.03

    elif task in ["hammering", "reaching"]:
        tool_quaternion = C.getFrame(selected_tool + "-base").getQuaternion()
        if "base" in selected_part:
            tool_handle_center = C.getFrame(selected_tool + "-base").getPosition()
        elif "head" in selected_part:
            tool_handle_center = C.getFrame(selected_tool + "-head").getPosition()
        else:
            raise Exception

        if hammering_flag:
            print("here")
            best_wp["pos"][1][0] = -best_wp["pos"][1][0]
            best_wp["pos"][2][0] = -best_wp["pos"][2][0]

        def as_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x, dtype=np.float64)

        target_obj_name = task + "-obj"
        tc = as_np(C.getFrame(target_obj_name).getPosition())

        p2 = as_np(best_wp["pos"][2])
        # best_wp["pos"][2] = (p2 + 0.5 * (tc - p2)).tolist()

        grasp_quaternion, grasp_position = grasping_post_process(
            {
                "tool-qua": tool_quaternion,
                "tool-handle-center": tool_handle_center,
                "predicted-qua": best_wp["qua"][0],
                "predicted-pos": best_wp["pos"][0],
                "tool-pc": tool_points,
            }
        )
        best_wp["qua"][0] = grasp_quaternion
        best_wp["qua"][0] = tool_quaternion
        best_wp["pos"][0] = grasp_position
        best_wp["pos"][0][2] = C.getFrame(selected_tool + "-base").getPosition()[2]

        if task == "hammering":
            best_wp["qua"][1] = rotate_quat_yaw_angle(best_wp["qua"][1], -np.pi / 4)
            best_wp["qua"][2] = rotate_quat_yaw_angle(best_wp["qua"][2], -np.pi / 4)
        else:
            print("*******here*******")
            best_wp["qua"][1] = rotate_quat_yaw_angle(best_wp["qua"][1], np.pi / 8)
            best_wp["qua"][2] = rotate_quat_yaw_angle(best_wp["qua"][2], np.pi / 8)

        initial_goal = best_wp["pos"][2][0] - best_wp["pos"][1][0]
        print("initial_goal: ", initial_goal)

        # best_wp["pos"][1][2] = target_center[2]
        # best_wp["pos"][2][2] = target_center[2]

        if hammering_flag:
            best_wp["qua"][1] = rotate_quat_yaw_180(best_wp["qua"][1])
            best_wp["qua"][2] = rotate_quat_yaw_180(best_wp["qua"][2])

    return best_wp


def run_manipulation_for_task(
    task,
    C,
    selected_tool,
    task_environment,
    environment_controller=None,
):
    if task == "lifting":
        real_score = ManipulationWithKOMO_lifting(C, selected_tool, task_environment)
    elif task in ["minigolf", "pushing"]:
        real_score = ManipulationWithKOMO_minigolf(C, selected_tool, task_environment)
    elif task in ["hammering"]:
        real_score = ManipulationWithKOMO_hammering(
            C,
            selected_tool,
            task_environment,
            environment_controller,
        )
    elif task in ["reaching"]:
        real_score = ManipulationWithKOMO_reaching(C, selected_tool, task_environment)

    return real_score

# For DEBUG 
def shift_toward_handle_center(p, handle_center, alpha=0.2):
    return p + alpha * (handle_center - p)

def fix_grasp_pos(tool_pts, p, R=0.035, belt_w=0.01, eps_in=0.004, min_pts=12):
    tree = cKDTree(tool_pts)
    _, i0 = tree.query(p, k=1)
    idx_patch = tree.query_ball_point(tool_pts[i0], r=R)
    P = tool_pts[idx_patch]
    if len(P) < min_pts:  # fallback
        return tool_pts[i0]

    C = P - P.mean(0)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    U, V, N = Vt[0], Vt[1], Vt[2]         # in-plane, in-plane, normal
    # right-handed
    if np.linalg.det(np.stack([U, V, N], 1)) < 0:
        V = -V

    # project to local 2D
    Pu = C @ U; Pv = C @ V
    Cp = p - P.mean(0)
    pu = float(Cp @ U); pv = float(Cp @ V)

    # decide along vs thickness by spread
    su, sv = Pu.std(), Pv.std()
    if su >= sv:
        a_axis, t_axis = ('u','v')
        a_vals, t_vals = Pu, Pv
        a_wp,  t_wp  = pu, pv
    else:
        a_axis, t_axis = ('v','u')
        a_vals, t_vals = Pv, Pu
        a_wp,  t_wp  = pv, pu

    # belt around the waypoint along-part coordinate
    belt = np.abs(a_vals - a_wp) <= belt_w
    if belt.sum() < min_pts//2:
        # widen once
        belt = np.abs(a_vals - a_wp) <= (2*belt_w)
        if belt.sum() < min_pts//3:
            return tool_pts[i0]  # fallback

    t_min, t_max = t_vals[belt].min(), t_vals[belt].max()
    t_mid = 0.5 * (t_min + t_max)

    # rebuild 3D with new thickness coord, same along-part coord
    if a_axis == 'u':
        pu_new, pv_new = a_wp, t_mid
    else:
        pu_new, pv_new = t_mid, a_wp

    p_mid = P.mean(0) + pu_new*U + pv_new*V - eps_in*N
    return p_mid

def is_inside_pointcloud( tool_points, waypoint, radius=0.005, min_dir_count=6):
    tree = cKDTree(tool_points)
    d, _ = tree.query(waypoint, k=1)
    if d > radius:
        return False
    
    idxs = tree.query_ball_point(waypoint, r=radius)
    if not idxs:
        return False
    rel_vecs = tool_points[idxs] - waypoint
    dirs = np.sign(rel_vecs)
    unique_dirs = set(tuple(v) for v in dirs)
    
    return len(unique_dirs) >= min_dir_count

def post_quaternion(original_qua, predicted_qua):
    original_qua = original_qua / np.linalg.norm(original_qua)
    predicted_qua = predicted_qua / np.linalg.norm(predicted_qua) 
    dot = abs(np.dot(original_qua, predicted_qua))  # dot product
    dot = np.clip(dot, -1.0, 1.0)
    angle = 2 * np.arccos(dot)  # in radians
    angle_diff = np.degrees(angle)
    print(f"**DEBUG** angle difference: {angle_diff}")
    # if angle_diff > 20.0:
    #     print(f"**DEBUG** post quaternion processing is done")
    #     q_final = original_qua
    # else:
    #     print(f"**DEBUG** post quaternion processing is not done")
    #     q_final = predicted_qua
    # return q_final    
    if angle_diff < 20.0 or 60 <= angle_diff <= 120:
        print(f"**DEBUG** post quaternion processing is done")
        q_final = original_qua
    else:
        print(f"**DEBUG** post quaternion processing is not done")
        q_final = predicted_qua
        
    return q_final 

def yaw_to_quaternion_scalar_first(yaw):
    """
    yaw: radians
    returns quaternion [w, x, y, z]
    """
    half = yaw / 2.0
    q = np.array([
        np.cos(half),   # w
        0.0,            # x
        0.0,            # y
        np.sin(half)    # z
    ])
    return q   

def post_posititon(tool_pc, pred_pos, normal_thresh_deg=15, dist_thresh=0.1): 
    tool_normals = compute_normals(tool_pc)
    # 1. Find the closest point to the predicted waypoint
    tree = cKDTree(tool_pc)
    dist, idx = tree.query(pred_pos)
    contact_point = tool_pc[idx]
    contact_normal = tool_normals[idx]

    # 2. Find potential opposite points based on normals
    dot_prods = tool_normals @ (-contact_normal)  # cosine of angle between normals
    angle_thresh = np.cos(np.radians(normal_thresh_deg))
    mask = dot_prods > angle_thresh

    # 3. Distance check
    candidates = tool_pc[mask]
    dists = np.linalg.norm(candidates - contact_point, axis=1)
    print(dists)
    valid_mask = dists < dist_thresh

    if np.any(valid_mask):
        print(f"**DEBUG** post position processing is done")
        opposite_points = candidates[valid_mask]
        # Pick the farthest one for better stability
        best_opposite = opposite_points[np.argmax(dists[valid_mask])]

        # 4. Midpoint between contacts
        new_pos = (contact_point + best_opposite) / 2.0
        return new_pos
    else:
        print(f"**DEBUG** post position processing is not done")
        return pred_pos     

def compute_normals(tool_pc_np, radius=0.01, max_nn=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tool_pc_np)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=max_nn)

    normals = np.asarray(pcd.normals)
    return normals


def min_dist_if_tool_at_initial(tool_pc_world,              # (N,3)
                                p_tool_w, q_tool_wxyz,     # tool waypoint (world)
                                p_init_w, q_init_wxyz,     # initial env waypoint (world)
                                target_center_world):      # (3,)
    
    # Ensure all inputs are numpy arrays
    def to_np(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.array(x, dtype=np.float32)

    tool_pc_world      = to_np(tool_pc_world)
    p_tool_w           = to_np(p_tool_w)
    q_tool_wxyz        = to_np(q_tool_wxyz)
    p_init_w           = to_np(p_init_w)
    q_init_wxyz        = to_np(q_init_wxyz)
    target_center_world = to_np(target_center_world)
    
    # quat: [w,x,y,z] -> [x,y,z,w]
    qT_xyzw = [q_tool_wxyz[1], q_tool_wxyz[2], q_tool_wxyz[3], q_tool_wxyz[0]]
    qI_xyzw = [q_init_wxyz[1], q_init_wxyz[2], q_init_wxyz[3], q_init_wxyz[0]]
    R_tool = R.from_quat(qT_xyzw).as_matrix()
    R_init = R.from_quat(qI_xyzw).as_matrix()

    # rigid transform that maps (p_tool, R_tool) -> (p_init, R_init)
    R_delta = R_init @ R_tool.T
    Xc = tool_pc_world - p_tool_w[None, :]        # center at tool waypoint
    X_new = (R_delta @ Xc.T).T + p_init_w[None, :]  # move whole tool

    dists = np.linalg.norm(X_new - target_center_world[None, :], axis=1)
    i = int(np.argmin(dists))
    return float(dists[i]), X_new[i] 


def most_rotated_quaternion(q_ref, q_a, q_b):
    """
    Selects the quaternion (q_a or q_b) that represents a larger rotation from q_ref.
    Quaternions are scalar-first [w, x, y, z].
    Handles both NumPy arrays and PyTorch tensors.
    """

    # --- Tensor → NumPy conversion ---
    import torch
    def to_numpy(q):
        if isinstance(q, torch.Tensor):
            return q.detach().cpu().numpy()
        return np.array(q, dtype=np.float64)

    q_ref = to_numpy(q_ref)
    q_a   = to_numpy(q_a)
    q_b   = to_numpy(q_b)

    # --- Normalize quaternions ---
    def normalize(q):
        return q / np.linalg.norm(q)

    q_ref = normalize(q_ref)
    q_a = normalize(q_a)
    q_b = normalize(q_b)

    # --- Rotation angle between quaternions ---
    def rotation_angle(q1, q2):
        dot = np.abs(np.dot(q1, q2))  # abs handles q and -q equivalence
        dot = np.clip(dot, -1.0, 1.0)
        return 2 * np.arccos(dot)  # radians

    angle_a = rotation_angle(q_ref, q_a)
    angle_b = rotation_angle(q_ref, q_b)

    return q_a if angle_a > angle_b else q_b

def rotate_quat_yaw_180(q):
    """
    Rotate a scalar-first quaternion q by 180 degrees (pi radians) around the Z axis.
    q: array-like [w, x, y, z] (scalar-first)
    Returns: rotated quaternion [w, x, y, z]
    """
    q = np.array(q, dtype=np.float64)

    # If tensor -> convert to numpy
    if hasattr(q, 'detach'):
        q = q.detach().cpu().numpy()

    # Rotation quaternion for 180° yaw about Z
    half_angle = np.pi / 2  # π/2 radians in half-angle form
    q_rot = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)], dtype=np.float64)

    # Quaternion multiplication (scalar-first)
    w1, x1, y1, z1 = q_rot
    w2, x2, y2, z2 = q
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    q_new = np.array([w, x, y, z], dtype=np.float64)

    # Normalize to avoid drift
    return q_new / np.linalg.norm(q_new)


def rotate_quat_yaw_angle(q, angle):
    """
    Rotate a scalar-first quaternion q by 180 degrees (pi radians) around the Z axis.
    q: array-like [w, x, y, z] (scalar-first)
    Returns: rotated quaternion [w, x, y, z]
    """
    q = np.array(q, dtype=np.float64)

    # If tensor -> convert to numpy
    if hasattr(q, 'detach'):
        q = q.detach().cpu().numpy()

    # Rotation quaternion for 180° yaw about Z
    # half_angle = np.pi / 2  # π/2 radians in half-angle form
    q_rot = np.array([np.cos(angle), 0.0, 0.0, np.sin(angle)], dtype=np.float64)

    # Quaternion multiplication (scalar-first)
    w1, x1, y1, z1 = q_rot
    w2, x2, y2, z2 = q
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    q_new = np.array([w, x, y, z], dtype=np.float64)

    # Normalize to avoid drift
    return q_new / np.linalg.norm(q_new)

def rot180_inplace_torch(points, axis='z'):
    """
    points: (N,3) float tensor (CPU or CUDA). Rotates IN-PLACE by 180°.
    """
    if axis == 'z':
        points[:, 0].mul_(-1.0)
        points[:, 1].mul_(-1.0)
    elif axis == 'x':
        points[:, 1].mul_(-1.0)
        points[:, 2].mul_(-1.0)
    elif axis == 'y':
        points[:, 0].mul_(-1.0)
        points[:, 2].mul_(-1.0)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

def rotate_quaternion(axis, quaternion, degrees, local_axis=True):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    current_rotation = R.from_quat(
        quaternion,
        scalar_first=True,
    )

    additional_rotation = R.from_rotvec(
        axis * np.deg2rad(degrees)
    )

    if local_axis:
        rotated = current_rotation * additional_rotation
    else:
        rotated = additional_rotation * current_rotation

    return rotated.as_quat(scalar_first=True)
