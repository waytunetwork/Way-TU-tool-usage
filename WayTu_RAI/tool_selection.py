import torch
import numpy as np
import os
import json
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import WayTu_RAI.model_utils as mutils
import WayTu_RAI.graph_utils as gutils
from WayTu_RAI.feature_extractor_yaw import SmallPointNetEncoderYaw


class WayTuSelectionDataset(Dataset):
    def __init__(self, cfg, data_df, device):
        self.cfg = cfg
        self.device = device
        self.data_df = data_df.reset_index(drop=True)
        
        self.label_list = cfg["label-list-all"]

        self.num_points = cfg['num-training-points']

        # Get Tool and Environment related labels
        environmental_mask = np.char.find(self.label_list, "platform") != -1
        self.env_label_idx = np.where(environmental_mask)[0]
        self.tool_label_idx = np.where(~environmental_mask)[0]
        self.data_augmentation = cfg.get("data_augmentation", False)

    def __len__(self):
        return self.data_df.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        data_number = row["path"]

        tool_waypoint = torch.tensor(json.loads(row["tool-waypoint"]), dtype=torch.float32)
        # initial_waypoint = torch.tensor(json.loads(row["initial-waypoint"]), dtype=torch.float32)
        # goal_waypoint = torch.tensor(json.loads(row["goal-waypoint"]), dtype=torch.float32)
        score = torch.tensor(row["score"], dtype=torch.float32)

        task_idx = self.label_list.index(row["task"] + "-platform")
        tool_idx = self.label_list.index(row["selected_tool"])
        task = torch.tensor(task_idx, dtype=torch.long)
        tool = torch.tensor(tool_idx, dtype=torch.long)

        data_path = os.path.join(self.cfg["dataset_dir"], f"data_{data_number}_pc.npy")
        pcl = np.load(data_path)
        # pcl = mutils.adaptive_farthest_point_sampling(pcl, self.num_points)
        pcl = mutils.uniform_object_point_sampling(pcl, self.num_points)
        points = pcl[:, :3]
        labels = pcl[:, 3].astype(int)

        tool_mask = np.isin(labels, self.tool_label_idx)
        env_mask = np.isin(labels, self.env_label_idx)

        env_labels_in_sample = labels[env_mask]
        counts = np.bincount(env_labels_in_sample)
        platform_label = np.argmax(counts)

        platform_class_idx = np.where(self.env_label_idx == platform_label)[0][0]
        env_onehot = np.zeros(len(self.env_label_idx), dtype=np.float32)
        env_onehot[platform_class_idx] = 1.0
        env_onehot = torch.tensor(env_onehot, dtype=torch.float32)
        # print(f"**DEBUG**: {env_onehot}")
        tool_points = points[tool_mask]

        # if tool_points.shape[0] < self.num_points:
        #     raise ValueError(f"Too few tool points in sample {data_number}")
        # if env_points.shape[0] < self.num_points:
        #     raise ValueError(f"Too few env points in sample {data_number}")

        # tool_points = mutils.adaptive_farthest_point_sampling(tool_points, self.num_points)
        tool_points = torch.tensor(tool_points, dtype=torch.float32)
        tool_points, tool_params = self.normalize_pc(tool_points)

        # env_points = mutils.adaptive_farthest_point_sampling(env_points, self.num_points)
        # env_points = torch.tensor(env_points, dtype=torch.float32)
        # norm_env_points, env_params = self.normalize_pc(env_points)

        # target_center = gutils.get_target_center(env_points, task_idx)
        # norm_tool_points = torch.randn((self.num_points, 3), dtype=torch.float32)
        # print(f"norm_tool_points shape: {norm_tool_points.shape}")
        # print(f"pcl: {pcl.shape}")
        if self.data_augmentation:
            theta = torch.rand(1) * 2 * torch.pi  # shape: [1]

            # 2. Create 3D rotation matrix around Z-axis
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            rot_matrix = torch.tensor([
                [cos_t.item(), -sin_t.item(), 0.0],
                [sin_t.item(),  cos_t.item(), 0.0],
                [0.0,           0.0,          1.0]
            ], dtype=torch.float32)

            # 3. Rotate tool point cloud (tool_points: [N, 3])
            tool_points = torch.matmul(tool_points, rot_matrix.T)

            # 4. Rotate position vector (first 3 of tool_waypoint)
            position = tool_waypoint[:3]              # shape: [3]
            rotated_pos = torch.matmul(rot_matrix, position)

            # 5. Rotate quaternion (tool_waypoint[3:] is in wxyz format)
            quat_wxyz = tool_waypoint[3:]             # shape: [4]
            quat_xyzw = torch.stack([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # to xyzw

            # Quaternion multiplication: rot_quat * quat_xyzw
            # rot_quat from Z-rotation
            rot_quat_xyzw = torch.tensor([
                0.0, 0.0, torch.sin(theta / 2),
                torch.cos(theta / 2)
            ], dtype=torch.float32)  # [x, y, z, w] scalar-last

            def quat_multiply(q1, q2):
                # Both inputs are [4] in xyzw format
                x1, y1, z1, w1 = q1
                x2, y2, z2, w2 = q2
                return torch.stack([
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                    w1*w2 - x1*x2 - y1*y2 - z1*z2
                ])

            rotated_quat_xyzw = quat_multiply(rot_quat_xyzw, quat_xyzw)

            # Convert back to wxyz format
            rotated_quat_wxyz = torch.stack([
                rotated_quat_xyzw[3],
                rotated_quat_xyzw[0],
                rotated_quat_xyzw[1],
                rotated_quat_xyzw[2]
            ])

            # 6. Final rotated waypoint
            tool_waypoint = torch.cat([rotated_pos, rotated_quat_wxyz])
        return {
            "tool_points": tool_points,
            "env_encoding" : env_onehot,
            "score": score,
            "tool_center" : tool_params["center"].float(),
            "tool_scale": torch.tensor(tool_params["scale"], dtype=torch.float32),
            "tool_waypoint" : tool_waypoint
        }
    
    def normalize_pc(self, point_cloud):
        M = point_cloud.size(0)
        center = point_cloud.mean(dim=0)                         # (3,)
        pts_centered = point_cloud - center.unsqueeze(0)     

        dists = torch.norm(pts_centered, dim=1)                       # (M,)
        scale = float(dists.max()) if M > 0 else 1.0

        if scale > 1e-8:
            normed = pts_centered / scale
        else:
            normed = pts_centered.clone()
        
        params = {
            "center":   center,      # (3,)
            "scale":    scale,       # float
        }
        return normed, params

class ToolSelectionNetwork(nn.Module):
    def __init__(self, tool_feat_dim, task_dim, waypoint_dim = 7, hidden_dim=256):
        super().__init__()
        input_dim = tool_feat_dim + task_dim + waypoint_dim # concat of tool and task

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # output score
        )

    def forward(self, x):
        # tool_feat: [B, tool_feat_dim]
        # task_onehot: [B, task_dim]
        # x = torch.cat([tool_feat, task_onehot], dim=-1)
        score = self.mlp(x).squeeze(-1)  # [B]
        return score
    
class WayTuSelectionModel(nn.Module):
    def __init__(self, cfg, device):
        super(WayTuSelectionModel, self).__init__()
        self.cfg = cfg 
        self.device = device
        
        self.encoder = SmallPointNetEncoderYaw(out_dim=cfg["feature-size"]).to(device)
        encoder_checkpoint = os.path.join("models", cfg["feature-extractor-path"])
        print(f"Loading feature extractor from {encoder_checkpoint}...")

        self.encoder.load_state_dict(torch.load(encoder_checkpoint, map_location=device)) # lo
        self.encoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.feature_size = cfg["feature-size"]
        self.embedding_size = self.feature_size + 3 + 1 + 1         # feature_size + normalize parameters + yaw + scale
        # self.waypoint_feature_size = self.embedding_size * 2 + 3    # An embedding for tool, an embedding for environment and target center
        self.selection_network = ToolSelectionNetwork(tool_feat_dim=self.embedding_size, task_dim=3).to(device)

    def forward(self, tool_points, env_encoding, waypoint, params):
        B = tool_points.shape[0]
        tool_embedding = self.get_embedding_vector(tool_points, {'centers': params["tool_centers"], 'scales': params["tool_scales"]})
        # env_embedding = self.get_embedding_vector(env_points, {'centers': params["env_centers"], 'scales': params["env_scales"]})

        full_embedding = torch.cat([
            tool_embedding, 
            env_encoding, 
            waypoint
        ], dim=1)

        score = self.selection_network(full_embedding)

        return score
    
    def get_embedding_vector(self, point_cloud, params):
        embeddings, yaws = self.encoder(point_cloud)
        centers = params["centers"]
        scales = params["scales"]

        point_cloud_embedding = torch.cat([
            embeddings,                   # (B, D)
            centers,                      # (B, 3)
            yaws.unsqueeze(1),            # (B, 1)
            scales.unsqueeze(1)           # (B, 1)
        ], dim=1)                         # → (B, D+3+1+1)

        return point_cloud_embedding



def train_waypoint_generator(parameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_df = pd.read_csv(os.path.join(parameters["dataset_dir"],"dataset_info.csv"))
    full_ds = WayTuSelectionDataset(parameters, data_df, device)
    val_split = parameters.get("val_split", 0.2)

    n_val = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=parameters["batch_size"], shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=parameters["batch_size"], shuffle=False, num_workers=4)

    model = WayTuSelectionModel(parameters, device).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=parameters["learning_rate"],
                           weight_decay=parameters["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=parameters["patience_lr"], factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10

    for epoch in range(1, parameters["num_epochs"]+1):
        # — Train —
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:2d} ▶ Train", leave=False)
        train_loss = 0.0
        train_total = 0
        for batch in train_bar:
            optimizer.zero_grad()

            tool_points = batch["tool_points"].to(device)       # (B, N, 3)
            env_onehot = batch["env_encoding"].to(device)
            tool_waypoint = batch["tool_waypoint"].to(device)

            params = {
                "tool_centers": batch["tool_center"].to(device),
                "tool_scales":  batch["tool_scale"].to(device),
            }

            gt_score = batch["score"].to(device)
            gt_score = torch.clamp(gt_score, min=0.0, max=1.0)

            score = model(tool_points, env_onehot, tool_waypoint, params)


            # MSE loss for score
            loss = F.mse_loss(score, gt_score)
            loss.backward()
            optimizer.step()

            
            train_loss += loss.item() * tool_points.size(0)
            train_total += tool_points.size(0)
        
        avg_train_loss = train_loss / train_total
    
        # — Validation —
        model.eval()
        val_loss = 0.0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:2d} ▶ Val  ", leave=False):
                tool_points = batch["tool_points"].to(device)       # (B, N, 3)
                env_onehot = batch["env_encoding"].to(device)
                tool_waypoint = batch["tool_waypoint"].to(device)


                params = {
                    "tool_centers": batch["tool_center"].to(device),
                    "tool_scales":  batch["tool_scale"].to(device),
                }


                gt_score = batch["score"].to(device)
                gt_score = torch.clamp(gt_score, min=0.0, max=1.0)

                score = model(tool_points, env_onehot, tool_waypoint, params)

                loss = F.mse_loss(score, gt_score)

                val_loss += loss.item() * tool_points.size(0)
                val_total += tool_points.size(0)

        avg_val_loss = val_loss / val_total

        print(
            f"Epoch {epoch:2d} "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )
        print(
            f"Epoch {epoch:2d} "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}",
        file= file_out, flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "tool_selection_v1_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

if __name__ == '__main__': 
    file_out = open("logs/tool_selection_v1_log.txt", "w")
    parameters = {
        "dataset_dir" : "Datasets/distractor-objects-waypoints",
        "feature-extractor-path" : "small-pointner-encoder-distractor_best.pth",
        "label-list-all" : ["lifting-platform", "minigolf-platform", "hammering-platform",
                  "hammer", "spatula", "L-ruler", "screwdriver", "ball", "book", "thin-stick", "ring"],
        "num-training-points" : 512,
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "val_split": 0.2,
        "patience_es": 10,
        "patience_lr": 3,
        "feature-size": 128,
        "lambda_pos" : 0.6,
        "lambda_quat" : 0.4,
        "data_augmentation" : True
    }
    train_waypoint_generator(parameters)
    file_out.close()



