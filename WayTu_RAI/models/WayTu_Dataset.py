import torch 
import os 
import numpy as np
import json
import open3d as o3d
from torch.utils.data import Dataset

from torch_geometric.nn import radius_graph
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
import WayTu_RAI.model_utils as mutils


class WayTuDataset(Dataset):
    def __init__(self, cfg, data_df, device):
        self.cfg = cfg
        self.device = device

        data_df.reset_index(drop=True, inplace=True)
        self.data_df = data_df
        # self.data_df = data_df.iloc[-100:].reset_index(drop=True)
        # self.data_df = data_df.reset_index(drop=True) 
        # self.data_df = data_df.sort_values(by="score", ascending=False).head(100).reset_index(drop=True)


        self.normalize = cfg['normalize']
        self.data_augmentation = cfg['data-augmentation']
        self.radius = cfg['radius']

        self.label_list = cfg["label-list-all"]
        self.background_label = len(self.label_list)-1
        self.num_points = cfg['num-training-points']

        # For Debugging the model: 
        environmental_mask = np.char.find(cfg["label-list-all"], "platform") != -1
        tool_mask = ~environmental_mask
        self.tool_related_label_idx = np.where(tool_mask)[0]

    
    def __len__(self):
        return self.data_df.shape[0]
    
    def __getitem__(self, idx):
        # Get path of sample folder 
        data_number = self.data_df.iloc[idx]["path"]
        
        tool_waypoint = torch.tensor(np.array(json.loads(self.data_df.iloc[idx]["tool-waypoint"])), dtype=torch.float) 
        initial_waypoint = torch.tensor(np.array(json.loads(self.data_df.iloc[idx]["initial-waypoint"])), dtype=torch.float) 
        goal_waypoint = torch.tensor(np.array(json.loads(self.data_df.iloc[idx]["goal-waypoint"])), dtype=torch.float) 
        score = torch.tensor(self.data_df.iloc[idx]["score"], dtype=torch.float)  
        
        # Additional Info
        task = torch.tensor(self.cfg['label-list-all'].index(self.data_df.iloc[idx]['task']), dtype=torch.float)
        tool = torch.tensor(self.cfg['label-list-all'].index(self.data_df.iloc[idx]['selected_tool']), dtype=torch.float)

        # Get point cloud 
        data_path = os.path.join(self.cfg["dataset-train-path"], "data_" + str(data_number) + "_pc.npy")
        pcl = np.load(data_path)



        pcl = mutils.adaptive_farthest_point_sampling(pcl, self.cfg["pc-size"])


        point_cloud = pcl[:, :3]
        labels = pcl[:, 3]
        labels = torch.tensor(labels, dtype=torch.float) 

        normals, curvatures = self.compute_normals_and_curvature(point_cloud)
        eigen_features = self.compute_eigenvalue_features(point_cloud)
        # pc_graph = self.construct_graph(point_cloud)

        if self.normalize: 
            # Normalize the positions of the point clouds:
            point_cloud = point_cloud - np.mean(point_cloud, axis=0)
            scale = np.max(np.linalg.norm(point_cloud, axis=1))
            point_cloud = point_cloud / scale

            # Normalize the normals of the point cloud
            norm = np.linalg.norm(normals, axis=1, keepdims=True)
            norm[norm == 0] = 1  # avoid division by zero
            normals = normals / norm
        
        pc_graph = self.construct_graph(point_cloud, normals, curvatures, eigen_features)

        return pc_graph, labels, tool_waypoint, initial_waypoint, goal_waypoint, score, task, tool

    # This function downsamples point clouds starting from background and high frequency environment platforms
    def downsample_pointcloud(self, pcl):
        pc = pcl[:, :3]
        labels = pcl[:, 3]

        background_mask = (labels == self.background_label)
        pc_wo_background = ...

    # Can be updated according to the paper
    def construct_graph(self, pc, normals= None, curvatures = None, eigen_features = None):
        pc_tensor = torch.tensor(pc, dtype=torch.float)
        # print(pc_tensor.shape)
        # edge_index = edge_index = radius_graph(pc_tensor, r=cfg['radius'])
        
        edge_index = knn_graph(pc_tensor, k=self.cfg["k-neighbors"])

        features = [pc_tensor]
        if normals is not None:
            features.append(torch.tensor(normals, dtype=torch.float))
        if curvatures is not None:
            features.append(torch.tensor(curvatures, dtype=torch.float))
        if eigen_features is not None: 
            features.append(torch.tensor(eigen_features, dtype=torch.float))

        node_features = torch.cat(features, dim=1)
        # print("AAAA")
        # print(node_features.shape)
        pc_graph = Data(x=node_features, edge_index=edge_index)

        return pc_graph

    def compute_normals_and_curvature(self, pc):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd.orient_normals_towards_camera_location(camera_location=[0.0, 0.0, 10.0])

        # Estimate curvature
        curvatures = []
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        for i in range(len(pc)):
            _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], 20)
            neighbors = np.asarray(pcd.points)[idx]
            cov = np.cov(neighbors.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)
            curvature = eigvals[0] / (eigvals.sum() + 1e-6)
            curvatures.append(curvature)
        
        normals = np.asarray(pcd.normals)
        curvatures = np.expand_dims(np.array(curvatures), axis=1)
        return normals, curvatures
    

    def compute_eigenvalue_features(self, pc, k=20):
        """
        Computes eigenvalue-based local shape descriptors: linearity, planarity, and sphericity.
        Returns a numpy array of shape (N, 3).
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)

        # Build KD-tree for fast neighbor search
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        features = []

        for i in range(len(pc)):
            _, idxs, _ = kdtree.search_knn_vector_3d(pcd.points[i], k)
            neighbors = np.asarray(pcd.points)[idxs, :]

            # Center the neighborhood
            neighbors_centered = neighbors - neighbors.mean(axis=0)

            # Covariance matrix
            cov = np.cov(neighbors_centered.T)

            # Eigenvalues
            eigvals = np.linalg.eigvalsh(cov)  # sorted in ascending order
            eigvals = np.sort(eigvals)[::-1] + 1e-10  # descending, add epsilon to avoid zero division

            λ1, λ2, λ3 = eigvals

            linearity = (λ1 - λ2) / λ1
            planarity = (λ2 - λ3) / λ1
            sphericity = λ3 / λ1

            features.append([linearity, planarity, sphericity])

        return np.array(features)


    def add_random_augmentation(self, point_cloud, waypoint):
        if  np.random.rand() < 0.5:
            point_cloud, waypoint = self.add_translation(point_cloud, waypoint)
        if  np.random.rand() < 0.5:
            point_cloud = self.add_noise(point_cloud)
    
    def add_translation(self, point_cloud, waypoint):
        translation_vector = np.random.uniform(-0.07, 0.07, size=2)
        translation_vector = np.append(translation_vector, 0)

        point_cloud = point_cloud + translation_vector
        waypoint[:2] = waypoint[:2] + translation_vector[:2]

        return point_cloud, waypoint
    
    def add_noise(self, point_cloud, noise_level = 0.01):
        noise = np.random.normal(scale=noise_level, size=point_cloud.shape)
        return point_cloud + noise
    
    