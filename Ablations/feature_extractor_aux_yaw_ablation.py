import argparse
import json
import os
import glob
import random
import re
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from scipy.spatial.transform import Rotation as R

from torch_geometric.nn import radius_graph
from sklearn.metrics.pairwise import cosine_similarity


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# sys.stdout = log_file
# sys.stderr = log_file

# ────────────────────────────────────────────────────────────────────────────────
# (1) DATASET DEFINITION
# ────────────────────────────────────────────────────────────────────────────────

class ObjectPCDataset(Dataset):
    """
    Each __getitem__ returns either:
      (pts_fixed, center, rotation, scale, label)   if normalize=True
      (pts_fixed, label)                             if normalize=False

    - pts_fixed:   torch.Tensor of shape (NUM_POINTS, 3), dtype float32
                   (centered/rotated/scaled & then subsampled/padded)
    - center:      torch.Tensor of shape (3,)         (only if normalize=True)
    - rotation:    torch.Tensor of shape (3,3)       (only if normalize=True)
    - scale:       float                              (only if normalize=True)
    - label:       int
    """
    def __init__(self, parameters):
        super().__init__()
        self.NUM_POINTS = parameters.get("num_points", 512)
        self.root_dir = parameters["dataset-path"]
        self.outlier_removal_bool = parameters["outlier-removal"]
        self.normalize_bool = parameters["normalize"]
        self.radius = parameters.get("radius", 0.05)
        self.min_neighbors = parameters.get("min_neighbors", 1)
        self.get_side = parameters.get("side", False)
        self.regress_yaw = parameters.get("regress-yaw", False)

        self.quat_csv_path = os.path.join(self.root_dir, "object_quaternions.csv")
        self.quat_df = pd.read_csv(self.quat_csv_path)  

        # Create map from (filename, label_index) → quaternion
        self.file_quat_map = {}
        for _, row in self.quat_df.iterrows():
            filename = f"data_{int(row['env_index'])}_pc_obj{int(row['label_index'])}.npy"
            quat = [row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"]]
            self.file_quat_map[(filename, int(row["label_index"]))] = quat

        # Build list of (file_path, label) for all .npy under root_dir/label_*/
        self.samples = []
        label_regex = re.compile(r"label_?(\d+(?:\.\d+)?)")
        for obj_folder in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, obj_folder)
            if not os.path.isdir(folder_path):
                continue
            if not obj_folder.startswith("label_"):
                continue

            num_str = label_regex.match(obj_folder).group(1)        # e.g. "0.0" or "1.0"
            try:
                l = int(float(num_str))  # convert "0.0" → 0, "1.0" → 1
            except ValueError:
                continue

            # print(f"##DEBUG## folder_path: {folder_path}, len: {len(sorted(glob.glob(os.path.join(folder_path, '*.npy'))))}")
            for fp in sorted(glob.glob(os.path.join(folder_path, "*.npy"))):
                self.samples.append((fp, l))

        if len(self.samples) == 0:
            raise RuntimeError(f"No object‐cloud files found under {self.root_dir}.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        raw = np.load(sample_path)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise RuntimeError(f"Expected shape (M,3) in {sample_path}, got {raw.shape}")

        pts = torch.from_numpy(raw.astype(np.float32))  # (M, 3)
        M = pts.size(0)

        # 1) Outlier removal (if enabled)
        if self.outlier_removal_bool and M > 0:
            keep_mask = self.outlier_removal(pts, radius=self.radius, min_neighbors=self.min_neighbors)
            pts_filtered = pts[keep_mask]
            if pts_filtered.shape[0] == 0:
                pts_filtered = pts  # fallback if all points get removed
        else:
            pts_filtered = pts
        
        # Additional step: get side
        # Calculate side:side = sign of x̄
        if self.get_side:
            env_center = pts_filtered.mean(dim=0)   # (x̄, ȳ, z̄)
            side = -1 if env_center[0] < 0.0 else 1

        # 2) Normalization (if enabled)
        if self.normalize_bool and not self.regress_yaw:
            pts_normed, params = self.normalize_point_cloud(pts_filtered)
        elif self.normalize_bool and self.regress_yaw: 
            pts_normed, params = self.normalize_and_yaw(pts_filtered)
        else:
            pts_normed = pts_filtered

        Nf = pts_normed.shape[0]

        # 3) Subsample or pad to exactly NUM_POINTS
        if Nf >= self.NUM_POINTS:
            indices = torch.randperm(Nf)[: self.NUM_POINTS]
            pts_fixed = pts_normed[indices, :]                       # (NUM_POINTS, 3)
        else:
            if Nf > 0:
                pad_count = self.NUM_POINTS - Nf
                pad_idx = torch.randint(0, Nf, (pad_count,))
                pts_fixed = torch.cat([pts_normed, pts_normed[pad_idx, :]], dim=0)  # (NUM_POINTS, 3)
            else:
                # If no points after filtering, fill with zeros
                pts_fixed = torch.zeros((self.NUM_POINTS, 3), dtype=torch.float32)
        # print(self.normalize_bool)
        # print(self.get_side)

        # if label < 0 or label >= 4:
        #     raise ValueError(f"Invalid label {label} for sample {sample_path}")
        # print(pts_fixed.shape)
        if self.normalize_bool and self.regress_yaw:
            sample_filename = os.path.basename(sample_path)
            quat_key = (sample_filename, label)
            if quat_key not in self.file_quat_map:
                raise RuntimeError(f"Quaternion not found for {quat_key}")
            quat = self.file_quat_map[quat_key]
            yaw = self.quaternion_to_yaw(quat)
            yaw_tensor = torch.tensor(yaw, dtype=torch.float32)
            # print("params yaw: ", yaw_tensor)
            return pts_fixed, params["center"], params["rotation"], params["scale"], label, yaw_tensor # params["yaw"]
        if self.normalize_bool and self.get_side:
            return pts_fixed, params["center"], params["rotation"], params["scale"], label, side
        elif self.normalize_bool and (not self.get_side):
            return pts_fixed, params["center"], params["rotation"], params["scale"], label
        elif(not self.normalize_bool) and self.get_side:
            return pts_fixed, label, side 
        else:
            return pts_fixed, label
    
    def quaternion_to_yaw(self, q):
        """Convert quaternion [w, x, y, z] to yaw in radians."""
        r = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x, y, z, w]
        euler = r.as_euler("zyx", degrees=False)
        return euler[0]  # yaw

    def outlier_removal(self, point_cloud: torch.Tensor, radius: float, min_neighbors: int = 1):
        """
        Keep only points that have ≥ min_neighbors neighbors within 'radius'.
        """
        M = point_cloud.size(0)
        if M == 0:
            return torch.zeros((0,), dtype=torch.bool, device=point_cloud.device)

        edge_index = radius_graph(point_cloud, r=radius, loop=False)
        deg = torch.bincount(edge_index[0], minlength=M)
        keep_mask = deg >= min_neighbors
        return keep_mask

    def normalize_and_yaw(self, point_cloud: torch.Tensor):
        """
        Center → normalize and scale but do not change orientation, calculate the yaw
        Returns (normalized_pts, params) where params = {'center', 'rotation', 'scale'}.
        """
        M = point_cloud.size(0)
        center = point_cloud.mean(dim=0)                         # (3,)
        pts_centered = point_cloud - center.unsqueeze(0)         # (M, 3)

        dists = torch.norm(pts_centered, dim=1)                       # (M,)
        scale = float(dists.max()) if M > 0 else 1.0

        C = pts_centered.T @ pts_centered
        U_c, S_c, Vh_c = torch.linalg.svd(C)                  
        rotation = Vh_c.T
        x_axis = rotation[:, 0]              # first column of rotation matrix
        yaw = torch.atan2(x_axis[1], x_axis[0])

        if scale > 1e-8:
            normed = pts_centered / scale
        else:
            normed = pts_centered.clone()

        params = {
            "center":   center,      # (3,)
            "rotation": rotation,    # (3,3)
            "scale":    scale,       # float
            "yaw": yaw
        }
        return normed, params



    def normalize_point_cloud(self, point_cloud: torch.Tensor):
        """
        Center → align via PCA (SVD) → scale to unit radius.
        Returns (normalized_pts, params) where params = {'center', 'rotation', 'scale'}.
        """
        M = point_cloud.size(0)
        center = point_cloud.mean(dim=0)                         # (3,)
        pts_centered = point_cloud - center.unsqueeze(0)         # (M, 3)

        C = pts_centered.T @ pts_centered
        U_c, S_c, Vh_c = torch.linalg.svd(C)                  
        rotation = Vh_c.T
        aligned = pts_centered @ rotation

        dists = torch.norm(aligned, dim=1)                       # (M,)
        scale = float(dists.max()) if M > 0 else 1.0

        if scale > 1e-8:
            normed = aligned / scale
        else:
            normed = aligned.clone()

        params = {
            "center":   center,      # (3,)
            "rotation": rotation,    # (3,3)
            "scale":    scale        # float
        }
        return normed, params


# ────────────────────────────────────────────────────────────────────────────────
# (2) MINI-POINTNET ENCODER
# ────────────────────────────────────────────────────────────────────────────────

class SmallPointNetEncoderYaw(nn.Module):
    """
    Lightweight PointNet encoder (N,3)→(out_dim), with:
      - Conv1d(3→32) → BN → ReLU
      - Conv1d(32→64) → BN → ReLU
      - Global maxpool → (64,)
      - FC(64→out_dim) → BN → ReLU → (out_dim,)
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, bias=False)
        self.bn2   = nn.BatchNorm1d(64)
        self.fc    = nn.Linear(in_features=64, out_features=out_dim, bias=False)
        self.bn_fc = nn.BatchNorm1d(out_dim)

        self.yaw_head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3) → transpose → (B, 3, N)
        x = x.transpose(2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))   # (B, 32, N)
        x = torch.relu(self.bn2(self.conv2(x)))   # (B, 64, N)

        x = torch.max(x, dim=2)[0]                # (B, 64)  global max pool
        feat  = torch.relu(self.bn_fc(self.fc(x)))    # (B, out_dim)
        yaw_pred = self.yaw_head(feat).squeeze(-1)
        return feat, yaw_pred


# ────────────────────────────────────────────────────────────────────────────────
# (3) TRAIN/VALID LOOP
# ────────────────────────────────────────────────────────────────────────────────

def train_with_validation(parameters):
    # Hyperparameters & device
    BATCH_SIZE    = parameters.get("batch_size", 16)
    NUM_EPOCHS    = parameters.get("num_epochs", 20)
    LEARNING_RATE = parameters.get("learning_rate", 1e-3)
    WEIGHT_DECAY  = parameters.get("weight_decay", 1e-4)
    OUT_DIM       = parameters.get("out_dim", 128)
    VAL_SPLIT     = parameters.get("val_split", 0.2)  # fraction for validation
    YAW_LOSS_WEIGHT = parameters.get("yaw-loss-weight", 1.0)
    SEED          = parameters.get("seed", 0)
    NUM_WORKERS   = parameters.get("num-workers", 0)
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)

    # 3.1) Build the full dataset
    full_dataset = ObjectPCDataset(parameters)

    # 3.2) Determine number of classes from dataset.samples
    all_labels = [lbl for (_, lbl) in full_dataset.samples]
    num_classes = len(set(all_labels))
    # print(f"Detected {num_classes} unique object classes.")

    # 3.3) Split into train/validation
    total_len = len(full_dataset)
    val_len   = int(total_len * VAL_SPLIT)
    train_len = total_len - val_len
    split_generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(
        full_dataset,
        [train_len, val_len],
        generator=split_generator,
    )

    split_path = Path(parameters["split-path"])
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": SEED,
                "train": train_ds.indices,
                "validation": val_ds.indices,
            },
            handle,
            indent=2,
        )

    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
    )

    # 4) Instantiate model (encoder + classifier)
    encoder    = SmallPointNetEncoderYaw(out_dim=OUT_DIM).to(DEVICE)
    classifier = nn.Linear(OUT_DIM, num_classes).to(DEVICE)

    class_loss_fn = nn.CrossEntropyLoss()
    # yaw_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    # Learning rate scheduler (reduce LR on plateau)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # Early stopping variables
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0


    # Optional LR scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # 5) Training + Validation Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Training ---
        encoder.train()
        classifier.train()
        train_loss = 0.0
        total_cls_loss = 0.0
        total_yaw_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader):
            # Unpack: pts_fixed, center, rotation, scale, label
            pts_fixed, _, _, _, labels, yaws = batch
            pts = pts_fixed.to(DEVICE)         # (B, N, 3)
            labels = labels.to(DEVICE)         # (B,)
            yaws   = yaws.to(DEVICE)

            optimizer.zero_grad()
            feats, yaw_preds = encoder(pts)         # (B, OUT_DIM)
            logits     = classifier(feats)  # (B, num_classes)

            loss_cls = class_loss_fn(logits, labels)
            # print(f"**DEBUG** yaw preds: {yaw_preds}")
            # print(f"**DEBUG** yaw preds: {yaws}")
            # loss_yaw = yaw_loss_fn(yaw_preds, yaws)
            loss_yaw = ((torch.sin(yaw_preds) - torch.sin(yaws))**2 + (torch.cos(yaw_preds) - torch.cos(yaws))**2).mean()
            loss = loss_cls + YAW_LOSS_WEIGHT * loss_yaw
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * pts.size(0)
            total_cls_loss += loss_cls.item() * pts.size(0)
            total_yaw_loss += loss_yaw.item() * pts.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total * 100.0
        avg_cls_train_loss = total_cls_loss / train_total
        avg_yaw_train_loss = total_yaw_loss / train_total

        # --- Validation ---
        encoder.eval()
        classifier.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_yaw_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                pts_fixed, _, _, _, labels, yaws = batch
                pts = pts_fixed.to(DEVICE)
                labels = labels.to(DEVICE)
                yaws   = yaws.to(DEVICE)

                feats, yaw_preds = encoder(pts)         # (B, OUT_DIM)
                logits     = classifier(feats)
                loss_cls = class_loss_fn(logits, labels)
                # loss_yaw = yaw_loss_fn(yaw_preds, yaws)
                loss_yaw = ((torch.sin(yaw_preds) - torch.sin(yaws))**2 + (torch.cos(yaw_preds) - torch.cos(yaws))**2).mean()
                loss = loss_cls + YAW_LOSS_WEIGHT * loss_yaw

                val_loss += loss.item() * pts.size(0)
                val_cls_loss += loss_cls.item() * pts.size(0)
                val_yaw_loss += loss_yaw.item() * pts.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total * 100.0
        avg_cls_val_loss = val_cls_loss / val_total
        avg_yaw_val_loss = val_yaw_loss / val_total

        # scheduler.step(avg_val_loss)  # if using a LR scheduler
        # Scheduler step
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            best_model_path = os.path.join("models", parameters["save_path"] + "_best.pth")
            torch.save(encoder.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS}  "
            f"Train Loss: {avg_train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
            f"Train Cls Loss: {avg_cls_train_loss:.4f}  Train Yaw Loss: {avg_yaw_train_loss:.4f}  "
            f"Val Cls Loss: {avg_cls_val_loss:.4f}  Val Yaw Loss: {avg_yaw_val_loss:.4f}  "
            f"Val Loss: {avg_val_loss:.4f}  Val Acc: {val_acc:.2f}%", file=file_out
        )
        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS}  "
            f"Train Loss: {avg_train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
            f"Train Cls Loss: {avg_cls_train_loss:.4f}  Train Yaw Loss: {avg_yaw_train_loss:.4f}%  "
            f"Val Cls Loss: {avg_cls_val_loss:.4f}  Val Yaw Loss: {avg_yaw_val_loss:.4f}%  "
            f"Val Loss: {avg_val_loss:.4f}  Val Acc: {val_acc:.2f}%"
        )

    # 6) Save the encoder (and classifier, if you want)
    last_model_path = os.path.join("models", parameters["save_path"] + "_last.pth")
    torch.save(encoder.state_dict(), last_model_path)
    print("Saved encoder to small_pointnet_encoder.pth")

def test_cosine_similarity(parameters):
    dataset = ObjectPCDataset(parameters)

    # 2) Build a dictionary of indices per label
    label_indices = {}
    for idx, (_, lbl) in enumerate(dataset.samples):
        label_indices.setdefault(lbl, []).append(idx)

    # 3) For each label, pick one random index
    selected_indices = []
    selected_labels = []
    for lbl, indices in label_indices.items():
        chosen_idx = np.random.choice(indices)
        selected_indices.append(chosen_idx)
        selected_labels.append(lbl)

    # 4) Create a Subset and DataLoader for those samples
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, num_workers=0)
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SmallPointNetEncoderYaw(out_dim=128).to(DEVICE)
    encoder.load_state_dict(torch.load("small_pointnet_encoder_distractor.pth", map_location=DEVICE))
    encoder.eval()

    # 6) Compute embeddings for each selected sample
    with torch.no_grad():
        for batch in loader:
            pts_fixed, _, _, _, labels = batch
            pts_fixed = pts_fixed.to(DEVICE)                 # (num_labels, N, 3)
            embeddings = encoder(pts_fixed)                   # (num_labels, 128)
            embeddings = embeddings.cpu().numpy()             # (num_labels, 128)
            break

    # 7) Compute pairwise cosine similarities
    cos_matrix = cosine_similarity(embeddings)             # (num_labels, num_labels)

    # 8) Build a DataFrame with human-readable indices
    label_names = [f"label_{lbl}" for lbl in selected_labels]
    df_cosine = pd.DataFrame(cos_matrix, index=label_names, columns=label_names)

    # 9) df_cosine now contains cosine similarities between every pair of labels
    #    You can print it or save to CSV:
    print(df_cosine)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train controlled auxiliary-yaw encoder ablations."
    )
    parser.add_argument(
        "--dataset-path",
        default="Datasets/distractor-object-dataset",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="aux_yaw_encoder_ablation",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    base_parameters = {
        "dataset-path": args.dataset_path,
        "outlier-removal": True,
        "normalize": True,
        "radius": 0.05,
        "min_neighbors": 1,
        "num_points": 512,
        "val_split": 0.2,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "out_dim": 128,
        # Keep this True for identical preprocessing and batch contents. The
        # controlled difference is only the yaw-loss weight below.
        "regress-yaw": True,
        "seed": args.seed,
        "num-workers": args.num_workers,
        "split-path": str(output_dir / "split_indices.json"),
    }

    variants = (
        ("aux_yaw", 1.0),
        ("no_aux_yaw", 0.0),
    )

    for variant_name, yaw_loss_weight in variants:
        parameters = dict(base_parameters)
        parameters["yaw-loss-weight"] = yaw_loss_weight
        parameters["save_path"] = (
            f"small-pointner-encoder-distractor-{variant_name}"
        )

        log_path = output_dir / f"{variant_name}.log"
        with open(log_path, "w", encoding="utf-8") as file_out:
            print(
                f"Training {variant_name} with yaw loss weight "
                f"{yaw_loss_weight}"
            )
            train_with_validation(parameters)

    print("Auxiliary-yaw encoder ablations completed.")
    print(f"Outputs saved under: {output_dir.resolve()}")
