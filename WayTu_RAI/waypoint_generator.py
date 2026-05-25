import torch
import numpy as np
import os
import json
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import WayTu_RAI.model_utils as mutils
import WayTu_RAI.graph_utils as gutils
from WayTu_RAI.feature_extractor_yaw import SmallPointNetEncoderYaw

file_out = open("logs/grasp_generator_log.txt", "w")

# ────────────────────────────────────────────────────────────────────────────────
# (1) DATASET DEFINITION
# ────────────────────────────────────────────────────────────────────────────────
class WayTuGeneratorDataset(Dataset):
    def __init__(self, cfg, data_df, device):
        self.cfg = cfg
        self.device = device
        self.data_df = data_df.reset_index(drop=True)
        
        self.label_list = ["lifting-platform", "minigolf-platform", "hammering-platform",
                  "hammer", "spatula", "L-ruler", "screwdriver", "knife", "wrench"]

        self.num_points = cfg['num-training-points']

        # Get Tool and Environment related labels
        environmental_mask = np.char.find(self.label_list, "platform") != -1
        self.env_label_idx = np.where(environmental_mask)[0]
        self.tool_label_idx = np.where(~environmental_mask)[0]

    def __len__(self):
        return self.data_df.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        data_number = row["path"]

        tool_waypoint = torch.tensor(json.loads(row["tool-waypoint"]), dtype=torch.float32)
        initial_waypoint = torch.tensor(json.loads(row["initial-waypoint"]), dtype=torch.float32)
        goal_waypoint = torch.tensor(json.loads(row["goal-waypoint"]), dtype=torch.float32)
        score = torch.tensor(row["score"], dtype=torch.float32)

        task_idx = self.label_list.index(row["task"])
        tool_idx = self.label_list.index(row["selected_tool"])
        task = torch.tensor(task_idx, dtype=torch.long)
        tool = torch.tensor(tool_idx, dtype=torch.long)

        data_path = os.path.join(self.cfg["dataset_dir"], f"data_{data_number}_pc.npy")
        pcl = np.load(data_path)
        pcl = mutils.adaptive_farthest_point_sampling(pcl, self.num_points)
        points = pcl[:, :3]
        labels = pcl[:, 3].astype(int)

        tool_mask = np.isin(labels, self.tool_label_idx)
        env_mask = np.isin(labels, self.env_label_idx)

        tool_points = points[tool_mask]
        env_points = points[env_mask]

        # if tool_points.shape[0] < self.num_points:
        #     raise ValueError(f"Too few tool points in sample {data_number}")
        # if env_points.shape[0] < self.num_points:
        #     raise ValueError(f"Too few env points in sample {data_number}")

        # tool_points = mutils.adaptive_farthest_point_sampling(tool_points, self.num_points)
        tool_points = torch.tensor(tool_points, dtype=torch.float32)
        norm_tool_points, tool_params = self.normalize_pc(tool_points)

        # env_points = mutils.adaptive_farthest_point_sampling(env_points, self.num_points)
        env_points = torch.tensor(env_points, dtype=torch.float32)
        norm_env_points, env_params = self.normalize_pc(env_points)

        target_center = gutils.get_target_center(env_points, task_idx)

        return {
            "tool_points": norm_tool_points,
            "env_points": norm_env_points,
            "tool_waypoint": tool_waypoint,
            "initial_waypoint": initial_waypoint,
            "goal_waypoint": goal_waypoint,
            "score": score,
            "task": task,
            "tool": tool, 
            "tool_center" : tool_params["center"].float(),
            "env_center" : env_params["center"].float(),
            "tool_scale": torch.tensor(tool_params["scale"], dtype=torch.float32),
            "env_scale":  torch.tensor(env_params["scale"],  dtype=torch.float32),
            "target_center" : target_center.float()
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

# ────────────────────────────────────────────────────────────────────────────────
# (2) MODEL DEFINITION
# ────────────────────────────────────────────────────────────────────────────────
class WayTuGeneratorModel(nn.Module):
    def __init__(self, cfg, device):
        super(WayTuGeneratorModel, self).__init__()
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
        self.waypoint_feature_size = self.embedding_size * 2 + 3    # An embedding for tool, an embedding for environment and target center
        self.waypoint_generator = WaypointGeneratorHead(cfg=cfg, num_waypoints=3, feature_size= self.waypoint_feature_size).to(device)

    def forward(self, tool_points, env_points, params):
        B = tool_points.shape[0]
        tool_embedding = self.get_embedding_vector(tool_points, {'centers': params["tool_centers"], 'scales': params["tool_scales"]})
        env_embedding = self.get_embedding_vector(env_points, {'centers': params["env_centers"], 'scales': params["env_scales"]})

        full_embedding = torch.cat([
            tool_embedding, 
            env_embedding, 
            params["target_center"]
        ], dim=1)

        pos_pred, quat_pred = self.waypoint_generator(full_embedding)

        return pos_pred, quat_pred
    
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


class WaypointGeneratorHead(nn.Module):
    def __init__(self, cfg, num_waypoints, feature_size=256):
        super().__init__()
        self.cfg = cfg
        self.num_waypoints = num_waypoints

        self.fc1 = mutils.xavier_initialization(feature_size, 1024)
        self.fc2 = mutils.xavier_initialization(1024, 512)
        self.fc3 = mutils.xavier_initialization(512, 256)
        self.fc4 = mutils.xavier_initialization(256, 128)
        self.fc5 = mutils.xavier_initialization(128, num_waypoints * 7)

        self.norm1 = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(256)
        self.norm4 = nn.LayerNorm(128)

        # Projection layers to match dimensions for residual connections
        self.proj1 = nn.Linear(1024, 512)
        self.proj2 = nn.Linear(512, 256)
        self.proj3 = nn.Linear(256, 128)

    def forward(self, x):
        x = F.silu(self.norm1(self.fc1(x)))
        
        residual = self.proj1(x)
        x = F.silu(self.norm2(self.fc2(x)))
        x += residual

        residual = self.proj2(x)
        x = F.silu(self.norm3(self.fc3(x)))
        x += residual

        residual = self.proj3(x)
        x = F.silu(self.norm4(self.fc4(x)))
        x += residual

        x = self.fc5(x)
        x = x.view(-1, self.num_waypoints, 7)

        positions = x[:, :, :3]
        quaternions = x[:, :, 3:]
        normalized_quaternions = F.normalize(quaternions, p=2, dim=-1)

        return positions, normalized_quaternions




def train_waypoint_generator(parameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_df = pd.read_csv(os.path.join(parameters["dataset_dir"],"dataset_info.csv"))
    full_ds = WayTuGeneratorDataset(parameters, data_df, device)
    val_split = parameters.get("val_split", 0.2)

    n_val = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=parameters["batch_size"], shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=parameters["batch_size"], shuffle=False, num_workers=4)

    model = WayTuGeneratorModel(parameters, device).to(device)
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
        total_pos_loss = 0.0
        total_quat_loss = 0.0
        train_total = 0
        for batch in train_bar:
            optimizer.zero_grad()

            tp = batch["tool_points"].to(device)       # (B, N, 3)
            ep = batch["env_points"].to(device)        # (B, N, 3)

            params = {
                "tool_centers": batch["tool_center"].to(device),
                "tool_scales":  batch["tool_scale"].to(device),
                "env_centers":  batch["env_center"].to(device),
                "env_scales":   batch["env_scale"].to(device),
                "target_center":batch["target_center"].to(device),
            }
            gt_tool = batch["tool_waypoint"].to(device)    # (B,7)
            gt_init = batch["initial_waypoint"].to(device) # (B,7)
            gt_goal = batch["goal_waypoint"].to(device)    # (B,7)
            gt_wp   = torch.stack([gt_tool, gt_init, gt_goal], dim=1)  # (B,3,7)

            gt_pos  = gt_wp[..., :3]   # (B,3,3)
            gt_quat = gt_wp[..., 3:]   # (B,3,4)

            pred_pos, pred_quat = model(tp, ep, params)


            pos_loss  = F.mse_loss(pred_pos,  gt_pos)
            quat_loss = geodesic_loss(pred_quat, gt_quat).mean()
            loss = parameters["lambda_pos"]*pos_loss + parameters["lambda_quat"]*quat_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * tp.size(0)
            total_pos_loss += pos_loss.item() * tp.size(0)
            total_quat_loss += quat_loss.item() * tp.size(0)
            train_total += tp.size(0)
        
        avg_train_loss = train_loss / train_total
        avg_pos_train_loss = total_pos_loss / train_total
        avg_quat_train_loss = total_quat_loss / train_total
        

        # — Validation —
        model.eval()
        val_loss = 0.0
        total_pos_val_loss = 0.0
        total_quat_val_loss = 0.0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:2d} ▶ Val  ", leave=False):
                tp = batch["tool_points"].to(device)
                ep = batch["env_points"].to(device)

                params = {
                    "tool_centers": batch["tool_center"].to(device),
                    "tool_scales":  batch["tool_scale"].to(device),
                    "env_centers":  batch["env_center"].to(device),
                    "env_scales":   batch["env_scale"].to(device),
                    "target_center":batch["target_center"].to(device),
                }

                gt_tool = batch["tool_waypoint"].to(device)
                gt_init = batch["initial_waypoint"].to(device)
                gt_goal = batch["goal_waypoint"].to(device)
                gt_wp   = torch.stack([gt_tool, gt_init, gt_goal], dim=1)

                gt_pos  = gt_wp[..., :3]
                gt_quat = gt_wp[..., 3:]

                pred_pos, pred_quat = model(tp, ep, params)

                pos_loss  = F.mse_loss(pred_pos,  gt_pos)
                quat_loss = geodesic_loss(pred_quat, gt_quat).mean()
                loss = parameters["lambda_pos"]*pos_loss + parameters["lambda_quat"]*quat_loss

                val_loss += loss.item() * tp.size(0)
                total_pos_val_loss += pos_loss.item() * tp.size(0)
                total_quat_val_loss += quat_loss.item() * tp.size(0)
                val_total += tp.size(0)

        avg_val_loss = val_loss / val_total
        avg_pos_val_loss = total_pos_val_loss / val_total
        avg_quat_val_loss = total_quat_val_loss / val_total

        print(
            f"Epoch {epoch:2d} "
            f"Train Loss: {avg_train_loss:.4f} | Pos: {avg_pos_train_loss:.4f}  Quat: {avg_quat_train_loss:.4f} || "
            f"Val Loss: {avg_val_loss:.4f}   | Pos: {avg_pos_val_loss:.4f}  Quat: {avg_quat_val_loss:.4f}"
        )
        print(
            f"Epoch {epoch:2d} "
            f"Train Loss: {avg_train_loss:.4f} | Pos: {avg_pos_train_loss:.4f}  Quat: {avg_quat_train_loss:.4f} || "
            f"Val Loss: {avg_val_loss:.4f}   | Pos: {avg_pos_val_loss:.4f}  Quat: {avg_quat_val_loss:.4f}",
        file= file_out, flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_grasp_generator_v2.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

def geodesic_loss(prediction, target, reduction="mean"):
    dot = (prediction * target).sum(dim=-1).abs()
    dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
    loss = 2.0 * torch.acos(dot)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


if __name__ == '__main__': 
    
    parameters = {
        "dataset_dir" : "Datasets/minigolf-dataset-top1800-pc",
        "feature-extractor-path" : "checkpoints/small_pointnet_encoder_yaw_v3.pth",
        "num-training-points" : 2048,
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
    }
    train_waypoint_generator(parameters)
    file_out.close()