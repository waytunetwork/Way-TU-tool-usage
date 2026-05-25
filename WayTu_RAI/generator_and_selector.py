import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import WayTu_RAI.model_utils as mutils
import WayTu_RAI.graph_utils as gutils
from WayTu_RAI.feature_extractor_yaw import SmallPointNetEncoderYaw


from WayTu_RAI.waypoint_generator import WaypointGeneratorHead
from WayTu_RAI.tool_selection import ToolSelectionNetwork
from scipy.spatial.transform import Rotation as R

class WayTuUnifiedDataset(Dataset):
    def __init__(self, cfg, data_df, device):
        self.cfg = cfg
        self.device = device
        self.data_df = data_df.reset_index(drop=True)

        self.label_list = cfg["label-list-all"]
        self.env_label_idx = [i for i, name in enumerate(self.label_list) if "platform" in name]
        self.tool_label_idx = [i for i, name in enumerate(self.label_list) if "platform" not in name]
        self.num_points = cfg['num-training-points']
        self.data_augmentation = cfg.get("data_augmentation", False)

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

        data_path = os.path.join(self.cfg["dataset_dir"], f"data_{data_number}_pc.npy")
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

        # convert to torch
        tool_points = torch.tensor(tool_points_np, dtype=torch.float32)
        tool_waypoint = torch.tensor(tool_waypoint_np, dtype=torch.float32)

        # normalize
        tool_points, tool_params = self.normalize_pc(tool_points)

        env_points = torch.tensor(points[env_mask], dtype=torch.float32)
        env_points, env_params = self.normalize_pc(env_points)

        env_labels_in_sample = labels[env_mask]
        platform_label = np.bincount(env_labels_in_sample).argmax()
        platform_class_idx = self.env_label_idx.index(platform_label)
        env_onehot = torch.zeros(len(self.env_label_idx), dtype=torch.float32)
        env_onehot[platform_class_idx] = 1.0

        target_center = gutils.get_target_center(env_points, task_idx)

        return {
            "tool_points": tool_points,
            "env_points": env_points,
            "env_encoding": env_onehot,
            "score": score,
            "tool_center": tool_params["center"].float(),
            "env_center": env_params["center"].float(),
            "tool_scale": tool_params["scale"].clone().detach(),
            "env_scale": env_params["scale"].clone().detach(),
            "target_center": target_center.float(),
            "tool_waypoint": tool_waypoint,
            "initial_waypoint": initial_waypoint,
            "goal_waypoint": goal_waypoint,
        }

    def rotation_augmentation_tool (self, tool_points, tool_waypoint): 
        """
        tool_points: (N, 3) NumPy array
        tool_waypoint: (7,) NumPy array [pos(3), quat(4)]
        Returns rotated versions
        """
        rand_rot = R.random()
        R_mat = rand_rot.as_matrix()

        # Rotate point cloud
        tool_points_rot = tool_points @ R_mat.T

        # Rotate position (first 3)
        wp_pos = tool_waypoint[:3]
        wp_pos_rot = R_mat @ wp_pos

        # Rotate quaternion (last 4)
        wp_quat = tool_waypoint[3:]
        wp_quat_matrix = R.from_quat(wp_quat).as_matrix()
        wp_quat_rot_matrix = R_mat @ wp_quat_matrix
        wp_quat_rot = R.from_matrix(wp_quat_rot_matrix).as_quat()

        # Combine
        waypoint_rot = np.concatenate([wp_pos_rot, wp_quat_rot])
        return tool_points_rot, waypoint_rot

    def normalize_pc(self, pc):
        center = pc.mean(dim=0)
        pc_centered = pc - center
        scale = pc_centered.norm(dim=1).max()
        return pc_centered / scale, {"center": center, "scale": scale}
    


class WayTuUnifiedModel(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Shared encoder (frozen)
        self.encoder = SmallPointNetEncoderYaw(out_dim=cfg["feature-size"]).to(device)
        encoder_ckpt = os.path.join("models", cfg["feature-extractor-path"])
        print(f"Loading frozen feature extractor from {encoder_ckpt}")
        self.encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Sizes
        self.feature_size = cfg["feature-size"]
        self.embedding_size = self.feature_size + 3 + 1 + 1  # emb + center + yaw + scale
        self.waypoint_input_size = self.embedding_size * 2 + 3  # tool + env + target center

        # Heads
        self.generator = WaypointGeneratorHead(cfg, num_waypoints=3, feature_size=self.waypoint_input_size)
        self.selector = ToolSelectionNetwork(tool_feat_dim=self.embedding_size,
                                             task_dim=3,
                                             waypoint_dim=7)

    def get_embedding_vector(self, pc, centers, scales):
        emb, yaw = self.encoder(pc)
        return torch.cat([emb, centers, yaw.unsqueeze(1), scales.unsqueeze(1)], dim=1),yaw

    def forward(self, tool_points, env_points, env_onehot, params):
        """
        Returns:
            positions: (B, 3, 3)
            quaternions: (B, 3, 4)
            score: (B,)
        """
        B = tool_points.size(0)

        tool_emb,yaw = self.get_embedding_vector(tool_points,
                                             centers=params["tool_centers"],
                                             scales=params["tool_scales"])
        env_emb, _ = self.get_embedding_vector(env_points,
                                            centers=params["env_centers"],
                                            scales=params["env_scales"])

        # Generator input
        gen_input = torch.cat([tool_emb, env_emb, params["target_center"]], dim=1)
        positions, quaternions = self.generator(gen_input)

        # Selector input (only uses the first predicted waypoint)
        tool_waypoint = torch.cat([positions[:, 0, :], quaternions[:, 0, :]], dim=1)
        sel_input = torch.cat([tool_emb, env_onehot, tool_waypoint], dim=1)
        score = self.selector(sel_input)

        return positions, quaternions, score, yaw


# def geodesic_loss(prediction, target):
#     dot = (prediction * target).sum(dim=-1).abs()
#     dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
#     return 2.0 * torch.acos(dot)

# def geodesic_loss_v2(prediction, target):
#     dot = (prediction * target).sum(dim=-1)
#     dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
#     return torch.acos(dot) ** 2

def geodesic_loss_v3(prediction, target):
    prediction = prediction / prediction.norm(p=2, dim=-1, keepdim=True)
    target = target / target.norm(p=2, dim=-1, keepdim=True)
    dot = (prediction * target).sum(dim=-1).abs()
    dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
    return 2.0 * torch.acos(dot)


def train_unified_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(os.path.join(cfg["dataset_dir"], "dataset_info.csv"))
    dataset = WayTuUnifiedDataset(cfg, df, device)

    n_val = int(len(dataset) * cfg.get("val_split", 0.2))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    model = WayTuUnifiedModel(cfg, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    best_val_loss = float("inf")
    patience_counter = 0
    log_file = open("logs/unified_model_v4_log.txt", "w")

    for epoch in range(cfg["num_epochs"]):
        model.train()
        total_loss = total_score = total_pos = total_quat = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d} ▶ Train", leave=False)
        for batch in train_bar:
            optimizer.zero_grad()

            tool_points = batch["tool_points"].to(device)
            env_points = batch["env_points"].to(device)
            env_encoding = batch["env_encoding"].to(device)
            gt_score = batch["score"].to(device)
            gt_score = torch.clamp(gt_score, 0.0, 1.0)

            gt_wp = torch.stack([
                batch["tool_waypoint"],
                batch["initial_waypoint"],
                batch["goal_waypoint"]
            ], dim=1).to(device)

            gt_pos = gt_wp[..., :3]
            gt_quat = gt_wp[..., 3:]

            params = {
                "tool_centers": batch["tool_center"].to(device),
                "tool_scales": batch["tool_scale"].to(device),
                "env_centers": batch["env_center"].to(device),
                "env_scales": batch["env_scale"].to(device),
                "target_center": batch["target_center"].to(device),
            }

            pos, quat, score = model(tool_points, env_points, env_encoding, params)

            score_loss = F.mse_loss(score, gt_score)
            pos_loss = F.mse_loss(pos, gt_pos)
            quat_loss = geodesic_loss_v3(quat, gt_quat).mean()

            loss = cfg["lambda_score"] * score_loss + cfg["lambda_pos"] * pos_loss + cfg["lambda_quat"] * quat_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * tool_points.size(0)
            total_score += score_loss.item() * tool_points.size(0)
            total_pos += pos_loss.item() * tool_points.size(0)
            total_quat += quat_loss.item() * tool_points.size(0)

        avg_train_loss = total_loss / len(train_ds)
        avg_score_loss = total_score / len(train_ds)
        avg_pos_loss = total_pos / len(train_ds)
        avg_quat_loss = total_quat / len(train_ds)

        # Validation
        model.eval()
        val_loss = val_score = val_pos = val_quat = 0
        val_total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d} ▶ Val", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                tool_points = batch["tool_points"].to(device)
                env_points = batch["env_points"].to(device)
                env_encoding = batch["env_encoding"].to(device)
                gt_score = batch["score"].to(device)
                gt_score = torch.clamp(gt_score, 0.0, 1.0)

                gt_wp = torch.stack([
                    batch["tool_waypoint"],
                    batch["initial_waypoint"],
                    batch["goal_waypoint"]
                ], dim=1).to(device)

                gt_pos = gt_wp[..., :3]
                gt_quat = gt_wp[..., 3:]

                params = {
                    "tool_centers": batch["tool_center"].to(device),
                    "tool_scales": batch["tool_scale"].to(device),
                    "env_centers": batch["env_center"].to(device),
                    "env_scales": batch["env_scale"].to(device),
                    "target_center": batch["target_center"].to(device),
                }

                pos, quat, score = model(tool_points, env_points, env_encoding, params)

                score_loss = F.mse_loss(score, gt_score)
                pos_loss = F.mse_loss(pos, gt_pos)
                quat_loss = geodesic_loss_v3(quat, gt_quat).mean()

                loss = cfg["lambda_score"] * score_loss + cfg["lambda_pos"] * pos_loss + cfg["lambda_quat"] * quat_loss

                val_loss += loss.item() * tool_points.size(0)
                val_score += score_loss.item() * tool_points.size(0)
                val_pos += pos_loss.item() * tool_points.size(0)
                val_quat += quat_loss.item() * tool_points.size(0)
                val_total += tool_points.size(0)

        avg_val_loss = val_loss / val_total
        avg_val_score = val_score / val_total
        avg_val_pos = val_pos / val_total
        avg_val_quat = val_quat / val_total

        log_msg = (
            f"Epoch {epoch+1}:\n"
            f"  Train → Total: {avg_train_loss:.4f}, Score: {avg_score_loss:.4f}, Pos: {avg_pos_loss:.4f}, Quat: {avg_quat_loss:.4f}\n"
            f"  Val   → Total: {avg_val_loss:.4f}, Score: {avg_val_score:.4f}, Pos: {avg_val_pos:.4f}, Quat: {avg_val_quat:.4f}\n"
        )

        print(log_msg.strip())
        print(log_msg.strip(), file=log_file, flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "waytu_unified_model_v4_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= cfg.get("patience_es", 10):
                print("Early stopping triggered.", file=log_file)
                print("Early stopping triggered.")
                break

    log_file.close()



if __name__ == '__main__':
    parameters = {
        "dataset_dir": "Datasets/minigolf-dataset-top1800-pc",
        "feature-extractor-path":  "small-pointner-encoder-distractor_best.pth",
        "label-list-all": ["lifting-platform", "minigolf-platform", "hammering-platform",
                  "hammer", "spatula", "L-ruler", "screwdriver", "ball", "book", "thin-stick", "ring"],
        "num-training-points": 512,
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "val_split": 0.2,
        "patience_es": 10,
        "feature-size": 128,
        "lambda_score": 0.1,
        "lambda_pos": 0.6,
        "lambda_quat": 0.4
    }
    train_unified_model(parameters)