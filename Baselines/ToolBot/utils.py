import robotic as ry
import numpy as np
import os

CAMERA_NAMES = ["camera1", "camera2", "camera3" ]

def create_dataset(root_dir, save_dir):
    environments = [
        os.path.join(root_dir, file_name)
        for file_name in os.listdir(root_dir)
        if file_name.lower().endswith('.g')
    ]
    for i in range(len(environments)):
        C = ry.Config()
        C.addFile(environments[i])
        pcl, rgb, depth = getPointCloud(C)

def getPC_from_cam(self, name): 
    cam = ry.CameraView(self.C)
    cam.setCamera(name)
    rgb, depth = cam.computeImageAndDepth(self.C)
    pcl = ry.depthImage2PointCloud(depth, cam.getFxycxy())

    return pcl, rgb, depth

def getPointCloud(C):
    pcls, rgbs, depths = [], [], []
    for cam in CAMERA_NAMES:
        cam_frame = C.getFrame(cam)
        pcl, rgb, depth = getPC_from_cam(cam)   
        pcl = cam_to_world(pcl.reshape(-1, 3), cam_frame)

        pcls.append(pcl)
        rgbs.append(rgb.reshape(-1, 3))
        depths.append(depth.reshape(-1))
    
    all_pcl = np.concatenate(pcls)
    all_rgb = np.concatenate(rgbs)
    all_depth = np.concatenate(depths)
    # The dimension of depth is different then others.

    return all_pcl, all_rgb, all_depth

def cam_to_world(point_cloud, cam_frame):
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

    # white_mask = np.any(rgb != np.array([250,250,250]), axis=1)
    # rgb = rgb[white_mask]
    # pcl = pcl[white_mask]

    print(rgb.shape)
    print(pcl.shape)

    rgb = rgb/256

    # self.visualize_pc_from_array(pcl, rgb)
    self.draw_in_simulation(pcl, "world", [0,0,255], rgb)

    # raise Exception

    return pcl, rgb


