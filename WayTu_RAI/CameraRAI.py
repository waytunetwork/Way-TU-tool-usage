import robotic as ry
import matplotlib.pyplot as plt
import time
import numpy as np
import open3d as o3d
import os

import WayTu_RAI.model_utils as mutils

class CameraRAI: 
    def __init__(self, cfg, C):
        self.cfg = cfg
        self.C = C

        print(self.C.getFrame("table").info())

        # self.table_threshold = 0.652 # Taken from old version
        self.table_threshold = self.C.getFrame("table").getPosition()[2] + self.C.getFrame("table").info()["size"][2]/2  + 0.005
        print("table threshold: ", self.table_threshold) 

        self.robot_threshold = self.C.getFrame("l_gripper").getPosition()[2] - 0.04

        table_pos = self.C.getFrame("table").getPosition()
        self.create_camera("camera1", table_pos + [0.0, 0.9, 0.62], mutils.euler_to_quaternion([(180, 0, 1, 0), (60, 1, 0, 0)])) # [0, 0, -0.92, 0.38]
        self.create_camera("camera2", table_pos + [-0.6, -0.3, 0.62], mutils.euler_to_quaternion([(-120, 1, 0, 0), (45, 0, 1, 0)])) #  [-0.38, 0.92, 0, 0]
        self.create_camera("camera3", table_pos + [0.6, -0.3, 0.62], mutils.euler_to_quaternion([(-120, 1, 0, 0), (-45, 0, 1, 0)])) # [-0.38, 0.92, 0, 0]
        # self.C.addFile("WayTu_RAI/cameras.g")
        self.create_camera("camera4", table_pos + [0.0, 0.00, 0.83], mutils.euler_to_quaternion([(180, 0, 1, 0)])) # Top-down view
        # self.create_camera("camera5", table_pos + [0.0, 0.95, 0.6], mutils.euler_to_quaternion([(135, 1, 0, 0), (180, 0, 0, 1)])) # Front view, looking at the scene
        new_pos = table_pos + [0.0, 0.7, 1.0]
        new_orientation = mutils.euler_to_quaternion([(150, 1, 0, 0), (180, 0, 0, 1)])
        self.create_camera("camera5", new_pos, new_orientation)
        self.create_camera("camera6", table_pos + [-1.2, 0.5, 0.9], mutils.euler_to_quaternion([(135, 1, 0, 0), (180, 0, 0, 1)])) # Side view

        # For VLM minigolf: 
        # camera 4 position: [0.0, 0.0, 0.83]


        print("camera height: ", self.C.getFrame("camera1").getPosition())

   

    def create_camera(self, name, position, quaternion):
        cam = self.C.addFrame(name)
        cam.setShape(type=ry.ST.marker, size=[.05])
        cam.setAttribute('focalLength', .5) # wide angle
        cam.setAttribute('width', 500)
        cam.setAttribute('height', 500)

        cam.setPosition(position)
        cam.setQuaternion(quaternion)
    
    def getPC_from_cam(self, name): 
        cam = ry.CameraView(self.C)
        cam.setCamera(name)
        rgb, depth = cam.computeImageAndDepth(self.C)
        pcl = ry.depthImage2PointCloud(depth, cam.getFxycxy())

        return pcl, rgb, depth


    
    def getPCLInfo(self, pcl, depth = None, rgb= None):
        print(f"fycxy: {self.C.view_fxycxy()}")
        print(f"pcl shape:  {pcl.shape}")

        if depth is not None: 
            print(f"depth shape:  {depth.shape}")
        if rgb is not None: 
            print(f"rgb shape:  {rgb.shape}")

    def draw_in_simulation(self, pcl, name, color, rgb, depth=None):
        # fig = plt.figure()
        # fig.add_subplot(1,2,1)
        # plt.imshow(rgb)
        # fig.add_subplot(1,2,2)
        # plt.imshow(depth)
        # plt.show()
        print(color)
        f = self.C.addFrame('pcl', name)
        f.setPointCloud(points= pcl, colors = color)

        # self.C.view()
    
    def getPointCloud(self):
        pcls, rgbs, depths = [], [], []
        for cam in self.cfg["cameras"]:
            cam_frame = self.C.getFrame(cam)
            pcl, rgb, depth = self.getPC_from_cam(cam)   
            pcl = mutils.cam_to_world(pcl.reshape(-1, 3), cam_frame)

            pcls.append(pcl)
            rgbs.append(rgb.reshape(-1, 3))
            depths.append(depth.reshape(-1))
        
        all_pcl = np.concatenate(pcls)
        all_rgb = np.concatenate(rgbs)
        all_depth = np.concatenate(depths)
        # The dimension of depth is different then others.


        return all_pcl, all_rgb, all_depth
    
    def cleanPointClouds(self, pcl, rgb):
        
        # For z dimension 
        table_mask = pcl[:,2] > self.table_threshold
        pcl = pcl[table_mask]
        rgb = rgb[table_mask]

        robot_mask = pcl[:,2] < self.robot_threshold
        pcl = pcl[robot_mask]
        rgb = rgb[robot_mask]

        # For y dimension
        table_mask = pcl[:,1] < 0.6
        pcl = pcl[table_mask]
        rgb = rgb[table_mask]

        table_mask = pcl[:,1] > 0
        pcl = pcl[table_mask]
        rgb = rgb[table_mask]



        print(rgb.shape)
        print(pcl.shape)

        rgb = rgb/256

        # self.visualize_pc_from_array(pcl, rgb)
        self.draw_in_simulation(pcl, "world", [0,0,255], rgb)



        return pcl, rgb
    
    def filter_by_color(self, points, colors, target_color, tolerance=0.1):
        mask = np.all(np.abs(colors - target_color) < tolerance, axis=1)
        return points[mask], colors[mask]
    
    def visualize_pc_with_label(self, point_cloud_with_labels):

        print("In visualize pc with label function")
        points = point_cloud_with_labels[:, :3]  # XYZ coordinates
        print(points.shape)
        print(points[:5])

        min_values = np.min(points, axis=0)
        max_values = np.max(points, axis=0)
        points = (points - min_values) / (max_values - min_values)

        labels = point_cloud_with_labels[:, 3].astype(int)

        num_classes = len(self.cfg["labels-list"])
        color_map = np.random.rand(num_classes, 3)
        colors = color_map[labels]

        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
        point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([point_cloud_o3d], window_name="Labeled Point Cloud")

    def visualize_pc_with_label_and_bb(self, point_cloud_with_labels, bounding_boxes):
        """
        Visualize a point cloud with labels and bounding boxes in Open3D.

        Args:
            point_cloud_with_labels (np.array): Nx4 array where the first three columns are XYZ coordinates and the last column is labels.
            bounding_boxes (dict): Dictionary of bounding boxes with keys 'min', 'max', and optionally 'color'.
        """
        # Extract points (XYZ) and labels
        points = point_cloud_with_labels[:, :3]  # XYZ coordinates
        labels = point_cloud_with_labels[:, 3].astype(int)

        # Normalize points
        min_values = np.min(points, axis=0)
        max_values = np.max(points, axis=0)
        points = (points - min_values) / (max_values - min_values)

        # Normalize bounding boxes
        normalized_bounding_boxes = {}
        for object_name, bbox in bounding_boxes.items():
            min_coords = np.array(bbox['min'])
            max_coords = np.array(bbox['max'])

            # Normalize min and max coordinates
            min_coords_normalized = (min_coords - min_values) / (max_values - min_values)
            max_coords_normalized = (max_coords - min_values) / (max_values - min_values)

            min_coords_normalized = np.clip(min_coords_normalized, 0.0, 1.0)
            max_coords_normalized = np.clip(max_coords_normalized, 0.0, 1.0)

            normalized_bounding_boxes[object_name] = {
                "min": min_coords_normalized,
                "max": max_coords_normalized,
                "color": bbox.get('color', [1, 0, 0])  # Default to red if no color is provided
            }
            print(normalized_bounding_boxes[object_name])
        


        # Map labels to colors
        num_classes = len(self.cfg["labels-list"])
        color_map = np.random.rand(num_classes, 3)  # Random colors for labels
        colors = color_map[labels]

        # Create Open3D PointCloud object
        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
        point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)

        # Create LineSets for normalized bounding boxes
        line_sets = []
        for object_name, bbox in normalized_bounding_boxes.items():
            min_coords = bbox['min']
            max_coords = bbox['max']
            color = bbox['color']
            line_set = self.create_bounding_box_lines(min_coords, max_coords, color)
            line_sets.append(line_set)

        # Visualize the point cloud and bounding boxes
        o3d.visualization.draw_geometries([point_cloud_o3d, *line_sets], window_name="Normalized Point Cloud with Bounding Boxes")

    def create_bounding_box_lines(self, min_coords, max_coords, color=[1, 0, 0]):
        """
        Create a LineSet for a bounding box.

        Args:
            min_coords (np.array): Minimum x, y, z coordinates of the bounding box.
            max_coords (np.array): Maximum x, y, z coordinates of the bounding box.
            color (list): RGB color for the bounding box lines (default is red).

        Returns:
            open3d.geometry.LineSet: LineSet representing the bounding box.
        """
        # Define the 8 corners of the bounding box
        corners = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],  # Bottom-back-left
            [min_coords[0], min_coords[1], max_coords[2]],  # Bottom-back-right
            [min_coords[0], max_coords[1], min_coords[2]],  # Bottom-front-left
            [min_coords[0], max_coords[1], max_coords[2]],  # Bottom-front-right
            [max_coords[0], min_coords[1], min_coords[2]],  # Top-back-left
            [max_coords[0], min_coords[1], max_coords[2]],  # Top-back-right
            [max_coords[0], max_coords[1], min_coords[2]],  # Top-front-left
            [max_coords[0], max_coords[1], max_coords[2]]   # Top-front-right
        ])

        # Define the edges of the bounding box (pairs of point indices)
        edges = [
            [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
            [4, 5], [5, 7], [7, 6], [6, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines
        ]

        # Create a LineSet object for the bounding box
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)

        # Assign the same color to all lines
        colors = [color for _ in range(len(edges))]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set
    

    # This function needs some editing to work with labeling. 
    def visualize_pc_from_array_with_color(self, pc, rgb=None):
        pcl = o3d.geometry.PointCloud()
        pcl.points =  o3d.utility.Vector3dVector(pc)
        o3d.visualization.draw_geometries([pcl])

        

        points = np.asarray(pcl.points)
        
        if rgb is not None: 
            pcl.colors = o3d.utility.Vector3dVector(rgb)
            colors = np.asarray(pcl.colors)
            # Define color thresholds for green, red, blue, and black
            green_threshold = np.array([0, 1, 0])
            red_threshold = np.array([1, 0, 0])
            blue_threshold = np.array([0, 0, 1])
            black_threshold = np.array([0, 0, 0])

            # Define a tolerance for color matching
            tolerance = 0.3

            # Function to filter points by color

            # Filter points for each color
            green_points, green_colors = self.filter_by_color(points, colors, green_threshold, tolerance)
            red_points, red_colors = self.filter_by_color(points, colors, red_threshold, tolerance)
            blue_points, blue_colors = self.filter_by_color(points, colors, blue_threshold, tolerance)
            black_points, black_colors = self.filter_by_color(points, colors, black_threshold, tolerance)

            # Combine all filtered points and colors
            filtered_points = np.concatenate((green_points, red_points, blue_points, black_points), axis=0)
            filtered_colors = np.concatenate((green_colors, red_colors, blue_colors, black_colors), axis=0)

            # Create a new point cloud with filtered points and colors
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

            # Visualize the filtered point cloud
            o3d.visualization.draw_geometries([filtered_pcd])

        # o3d.visualization.draw_geometries([pcl])
    
