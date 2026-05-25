from torch.utils.data import Dataset
import json
import pandas as pd
import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation as R


# import WayTu_Model.graph_utils as gutils
import Baselines.KETO.model_utils as mutils




class WayTuUnifiedDataset(Dataset):
    def __init__(self, cfg, data_df, device):
        self.cfg = cfg
        self.device = device

        self.data_df = data_df.reset_index(drop=True)
        self.data_df = data_df.nlargest(1000, "score").reset_index(drop=True)

        # -- Just for debug --
        # self.data_df = self.data_df[self.data_df["initial-waypoint"].apply(
        #     lambda x: json.loads(x)[0] > 0
        # )].reset_index(drop=True)

        # def is_z_axis_up(q_scalar_first):
        #     q = json.loads(q_scalar_first)[-4:]  # [w, x, y, z]
        #     q_xyzw = [q[1], q[2], q[3], q[0]]    # -> [x, y, z, w]
        #     Rz = R.from_quat(q_xyzw).as_matrix()[2, 2]
        #     return Rz > 0.5

        # self.data_df = self.data_df[self.data_df["initial-waypoint"].apply(is_z_axis_up)].reset_index(drop=True)
        # ----

        # optional: fine-tune filter (left intact)
        self.label_list = self.cfg["label-list-all"]
        self.env_label_idx = [i for i, name in enumerate(self.label_list) if "platform" in name]
        self.tool_label_idx = [i for i, name in enumerate(self.label_list) if "platform" not in name]
        self.num_points = self.cfg['num-training-points']
        self.data_augmentation = self.cfg.get("data_augmentation", False)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        data_number = row["path"]
        score = torch.tensor(row["score"], dtype=torch.float32)

        tool_waypoint = np.array(json.loads(row["tool-waypoint"]))  # (7,)
        initial_waypoint = torch.tensor(json.loads(row["initial-waypoint"]), dtype=torch.float32)
        goal_waypoint = torch.tensor(json.loads(row["goal-waypoint"]), dtype=torch.float32)

        task_idx = self.label_list.index(row["task"])
        tool_idx = self.label_list.index(row["selected_tool"])

        # choose dataset root: prefer per-row dataset_dir if present (multi-task); else cfg["dataset_dir"]
        base_dir = row["dataset-path"] if "dataset-path" in row else self.cfg["dataset-path"]
        data_path = os.path.join(base_dir, f"data_{data_number}_pc.npy")

        pcl = np.load(data_path)
        pcl = mutils.adaptive_farthest_point_sampling(pcl, self.num_points)

        points = pcl[:, :3]
        labels = pcl[:, 3].astype(int)

        tool_mask = np.isin(labels, self.tool_label_idx)
        env_mask = np.isin(labels, self.env_label_idx)

        tool_points_np = points[tool_mask]
        tool_waypoint_np = tool_waypoint

        # ─── ROTATION AUGMENTATION (only on tool) ────────────────
        if self.data_augmentation:
            tool_points_np, tool_waypoint_np = self.rotation_augmentation_tool_numpy(tool_points_np, tool_waypoint_np)

        # convert to torch and normalize
        tool_points = torch.tensor(tool_points_np, dtype=torch.float32)
        tool_waypoint = torch.tensor(tool_waypoint_np, dtype=torch.float32)
        # tool_points, tool_params = self.normalize_pc(tool_points)

        env_points = torch.tensor(points[env_mask], dtype=torch.float32)
        # env_points, env_params = self.normalize_pc(env_points)

        # normalize waypoint positions
        # tool_waypoint[:3] = (tool_waypoint[:3] - tool_params["center"]) / tool_params["scale"]
        # initial_waypoint[:3] = (initial_waypoint[:3] - env_params["center"]) / env_params["scale"]
        # goal_waypoint[:3] = (goal_waypoint[:3] - env_params["center"]) / env_params["scale"]

        # env one-hot
        env_labels_in_sample = labels[env_mask]
        platform_label = np.bincount(env_labels_in_sample).argmax()
        platform_class_idx = self.env_label_idx.index(platform_label)
        env_onehot = torch.zeros(len(self.env_label_idx), dtype=torch.float32)
        env_onehot[platform_class_idx] = 1.0

        # target_center = gutils.get_target_center(env_points, task_idx)

        return {
            "tool_points": tool_points,
            "env_points": env_points,
            "env_encoding": env_onehot,
            "score": score,
            # "tool_center": tool_params["center"].float(),
            # "env_center": env_params["center"].float(),
            # "tool_scale": tool_params["scale"].clone().detach(),
            # "env_scale": env_params["scale"].clone().detach(),
            # "target_center": target_center.float(),
            "tool_waypoint": tool_waypoint,
            "initial_waypoint": initial_waypoint,
            "goal_waypoint": goal_waypoint,
        }

    # keep your original numpy-based rot augmentation name available
    def rotation_augmentation_tool_numpy(self, tool_points, tool_waypoint):
        return self.rotation_augmentation_tool(tool_points, tool_waypoint)

    def rotation_augmentation_tool(self, tool_points, tool_waypoint):
        """
        tool_points: (N, 3) NumPy array
        tool_waypoint: (7,) NumPy array [pos(3), quat(4)]
        Returns rotated versions
        """
        rand_rot = R.random()
        R_mat = rand_rot.as_matrix()

        tool_points_rot = tool_points @ R_mat.T
        wp_pos = tool_waypoint[:3]
        wp_pos_rot = R_mat @ wp_pos

        wp_quat = tool_waypoint[3:]
        wp_quat_matrix = R.from_quat(wp_quat).as_matrix()
        wp_quat_rot_matrix = R_mat @ wp_quat_matrix
        wp_quat_rot = R.from_matrix(wp_quat_rot_matrix).as_quat()

        waypoint_rot = np.concatenate([wp_pos_rot, wp_quat_rot])
        return tool_points_rot, waypoint_rot

    def normalize_pc(self, pc):
        center = pc.mean(dim=0)
        pc_centered = pc - center
        scale = pc_centered.norm(dim=1).max()
        return pc_centered / scale, {"center": center, "scale": scale}