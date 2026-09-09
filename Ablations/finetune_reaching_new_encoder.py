#!/usr/bin/env python3
"""Fine-tune the Way-Tu hammering model on the reaching dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


PROJECT_ROOT = Path("/home/ece/git/WayTu-002")
DATASET_ROOT = PROJECT_ROOT / "Datasets/reaching-task-data-more"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "waytu_unified_model_hammering_v25_best.pth"
DEFAULT_ENCODER_CHECKPOINT = (
    PROJECT_ROOT
    / "reaching_encoder_finetune/last_block/best_encoder.pth"
)
OUTPUT_ROOT = PROJECT_ROOT / "reaching_finetune"

FEATURE_EXTRACTOR = "small-pointner-encoder-distractor_best.pth"
TOOL_LABELS = (3, 4, 5)
ENVIRONMENT_LABEL = 11
HAMMERING_TASK_INDEX = 2
TASK_ENCODING = (0.0, 0.0, 1.0)


def configure_imports() -> None:
    candidates = (PROJECT_ROOT, PROJECT_ROOT / "WayTu_Model")
    for candidate in candidates:
        if (candidate / "generator_and_selector.py").exists():
            sys.path.insert(0, str(candidate))
            return
    raise FileNotFoundError(
        "generator_and_selector.py was not found in the project root or WayTu_RAI."
    )


os.chdir(PROJECT_ROOT)
(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
configure_imports()

import graph_utils as gutils  # noqa: E402
import model_utils as mutils  # noqa: E402
from generator_and_selector import WayTuUnifiedModel, geodesic_loss_v3  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("frozen", "last_block"),
        default="frozen",
        help="Keep the encoder frozen or fine-tune its final projection block.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint used to initialize the complete Way-Tu model.",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=DEFAULT_ENCODER_CHECKPOINT,
        help=(
            "Fine-tuned feature-extractor checkpoint loaded after the complete "
            "Way-Tu checkpoint."
        ),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output subdirectory name. Defaults to the selected mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and checkpoint, then exit before training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_waypoint(value: str) -> torch.Tensor:
    return torch.tensor(json.loads(value), dtype=torch.float32)


class ReachingFineTuneDataset(Dataset):
    def __init__(self, dataset_root: Path, num_points: int = 512):
        self.dataset_root = dataset_root
        self.num_points = num_points
        self.data_frame = pd.read_csv(dataset_root / "dataset_info.csv").reset_index(
            drop=True
        )

        required_columns = {
            "path",
            "task",
            "selected_tool",
            "tool-waypoint",
            "initial-waypoint",
            "goal-waypoint",
            "score",
        }
        missing_columns = required_columns.difference(self.data_frame.columns)
        if missing_columns:
            raise ValueError(f"Missing dataset columns: {sorted(missing_columns)}")

        unexpected_tasks = set(self.data_frame["task"].unique()) - {
            "reaching-platform"
        }
        if unexpected_tasks:
            raise ValueError(f"Unexpected tasks: {sorted(unexpected_tasks)}")

    def __len__(self) -> int:
        return len(self.data_frame)

    @staticmethod
    def normalize_pc(points: torch.Tensor) -> tuple[torch.Tensor, dict]:
        center = points.mean(dim=0)
        centered = points - center.unsqueeze(0)
        scale = centered.norm(dim=1).max()
        if scale <= 1e-8:
            raise ValueError("Point cloud scale is zero.")
        normalized = centered / scale
        return normalized, {"center": center, "scale": scale}

    def __getitem__(self, index: int) -> dict:
        row = self.data_frame.iloc[index]
        sample_id = int(row["path"])
        point_cloud_path = self.dataset_root / f"data_{sample_id}_pc.npy"
        point_cloud = np.load(point_cloud_path)

        sampled = mutils.adaptive_farthest_point_sampling(
            point_cloud, self.num_points
        )
        points = sampled[:, :3]
        labels = sampled[:, 3].astype(int)

        tool_mask = np.isin(labels, TOOL_LABELS)
        environment_mask = labels == ENVIRONMENT_LABEL

        if not tool_mask.any():
            raise ValueError(f"No tool points found in {point_cloud_path}")
        if not environment_mask.any():
            raise ValueError(f"No environment points found in {point_cloud_path}")

        tool_points = torch.tensor(points[tool_mask], dtype=torch.float32)
        environment_points = torch.tensor(
            points[environment_mask], dtype=torch.float32
        )

        tool_points, tool_params = self.normalize_pc(tool_points)
        environment_points, environment_params = self.normalize_pc(
            environment_points
        )

        tool_waypoint = load_waypoint(row["tool-waypoint"])
        initial_waypoint = load_waypoint(row["initial-waypoint"])
        goal_waypoint = load_waypoint(row["goal-waypoint"])

        tool_waypoint[:3] = (
            tool_waypoint[:3] - tool_params["center"]
        ) / tool_params["scale"]
        initial_waypoint[:3] = (
            initial_waypoint[:3] - environment_params["center"]
        ) / environment_params["scale"]
        goal_waypoint[:3] = (
            goal_waypoint[:3] - environment_params["center"]
        ) / environment_params["scale"]

        target_center = gutils.get_target_center(
            environment_points, HAMMERING_TASK_INDEX
        )

        return {
            "tool_points": tool_points,
            "env_points": environment_points,
            "env_encoding": torch.tensor(TASK_ENCODING, dtype=torch.float32),
            "score": torch.tensor(float(row["score"]), dtype=torch.float32),
            "tool_center": tool_params["center"].float(),
            "env_center": environment_params["center"].float(),
            "tool_scale": tool_params["scale"].float(),
            "env_scale": environment_params["scale"].float(),
            "target_center": target_center.float(),
            "tool_waypoint": tool_waypoint,
            "initial_waypoint": initial_waypoint,
            "goal_waypoint": goal_waypoint,
            "sample_id": sample_id,
        }


def model_config() -> dict:
    return {
        "feature-extractor-path": FEATURE_EXTRACTOR,
        "feature-size": 128,
        "label-list-all": [
            "lifting-platform",
            "minigolf-platform",
            "hammering-platform",
            "hammer",
            "spatula",
            "L-ruler",
            "screwdriver",
            "ball",
            "book",
            "thin-stick",
            "ring",
        ],
    }


def unwrap_state_dict(checkpoint: object) -> dict:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary.")

    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break

    state_dict = {}
    for key, value in checkpoint.items():
        clean_key = key.removeprefix("module.")
        state_dict[clean_key] = value
    return state_dict


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    return unwrap_state_dict(checkpoint)


def configure_trainable_parameters(
    model: WayTuUnifiedModel, mode: str
) -> list[dict]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.generator.parameters():
        parameter.requires_grad = True
    for parameter in model.selector.parameters():
        parameter.requires_grad = True

    parameter_groups = [
        {
            "params": list(model.generator.parameters())
            + list(model.selector.parameters()),
            "name": "heads",
        }
    ]

    if mode == "last_block":
        encoder_modules = (model.encoder.fc, model.encoder.bn_fc, model.encoder.yaw_head)
        encoder_parameters = []
        for module in encoder_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
                encoder_parameters.append(parameter)
        parameter_groups.append({"params": encoder_parameters, "name": "encoder"})

    return parameter_groups


def set_training_mode(model: WayTuUnifiedModel, mode: str) -> None:
    model.train()
    if mode == "frozen":
        model.encoder.eval()
        return

    model.encoder.conv1.eval()
    model.encoder.bn1.eval()
    model.encoder.conv2.eval()
    model.encoder.bn2.eval()
    model.encoder.fc.train()
    model.encoder.bn_fc.train()
    model.encoder.yaw_head.train()


def move_batch(batch: dict, device: torch.device) -> tuple:
    tool_points = batch["tool_points"].to(device)
    environment_points = batch["env_points"].to(device)
    environment_encoding = batch["env_encoding"].to(device)
    score = batch["score"].to(device).clamp(0.0, 1.0)

    waypoints = torch.stack(
        [
            batch["tool_waypoint"],
            batch["initial_waypoint"],
            batch["goal_waypoint"],
        ],
        dim=1,
    ).to(device)

    params = {
        "tool_centers": batch["tool_center"].to(device),
        "tool_scales": batch["tool_scale"].to(device),
        "env_centers": batch["env_center"].to(device),
        "env_scales": batch["env_scale"].to(device),
        "target_center": batch["target_center"].to(device),
    }

    return (
        tool_points,
        environment_points,
        environment_encoding,
        score,
        waypoints,
        params,
    )


def compute_losses(
    predicted_position: torch.Tensor,
    predicted_quaternion: torch.Tensor,
    predicted_score: torch.Tensor,
    target_waypoints: torch.Tensor,
    target_score: torch.Tensor,
) -> dict:
    target_position = target_waypoints[..., :3]
    target_quaternion = target_waypoints[..., 3:]

    score_loss = F.mse_loss(predicted_score, target_score)
    position_loss = F.smooth_l1_loss(
        predicted_position, target_position, beta=0.02
    )
    quaternion_loss = geodesic_loss_v3(
        predicted_quaternion, target_quaternion
    ).mean()
    total_loss = 0.1 * score_loss + 0.7 * position_loss + 0.2 * quaternion_loss

    return {
        "total": total_loss,
        "score": score_loss,
        "position": position_loss,
        "quaternion": quaternion_loss,
    }


def train_one_epoch(
    model: WayTuUnifiedModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str,
) -> dict:
    set_training_mode(model, mode)
    totals = {key: 0.0 for key in ("total", "score", "position", "quaternion")}
    sample_count = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        (
            tool_points,
            environment_points,
            environment_encoding,
            target_score,
            target_waypoints,
            params,
        ) = move_batch(batch, device)

        predicted_position, predicted_quaternion, predicted_score = model(
            tool_points, environment_points, environment_encoding, params
        )
        losses = compute_losses(
            predicted_position,
            predicted_quaternion,
            predicted_score,
            target_waypoints,
            target_score,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()

        batch_size = tool_points.size(0)
        sample_count += batch_size
        for key in totals:
            totals[key] += losses[key].item() * batch_size

    return {key: value / sample_count for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: WayTuUnifiedModel, loader: DataLoader, device: torch.device
) -> dict:
    model.eval()
    total_loss = 0.0
    position_squared_error = 0.0
    score_squared_error = 0.0
    quaternion_error = 0.0
    sample_count = 0
    position_value_count = 0
    score_value_count = 0
    quaternion_value_count = 0

    per_waypoint_position = torch.zeros(3, dtype=torch.float64)
    per_waypoint_quaternion = torch.zeros(3, dtype=torch.float64)

    for batch in tqdm(loader, desc="Validate", leave=False):
        (
            tool_points,
            environment_points,
            environment_encoding,
            target_score,
            target_waypoints,
            params,
        ) = move_batch(batch, device)

        predicted_position, predicted_quaternion, predicted_score = model(
            tool_points, environment_points, environment_encoding, params
        )
        losses = compute_losses(
            predicted_position,
            predicted_quaternion,
            predicted_score,
            target_waypoints,
            target_score,
        )

        target_position = target_waypoints[..., :3]
        target_quaternion = target_waypoints[..., 3:]
        position_error = (predicted_position - target_position).pow(2)
        score_error = (predicted_score - target_score).pow(2)
        angle_error = geodesic_loss_v3(
            predicted_quaternion, target_quaternion
        )

        batch_size = tool_points.size(0)
        sample_count += batch_size
        total_loss += losses["total"].item() * batch_size
        position_squared_error += position_error.sum().item()
        score_squared_error += score_error.sum().item()
        quaternion_error += angle_error.sum().item()
        position_value_count += position_error.numel()
        score_value_count += score_error.numel()
        quaternion_value_count += angle_error.numel()

        per_waypoint_position += position_error.mean(dim=(0, 2)).cpu().double() * batch_size
        per_waypoint_quaternion += angle_error.mean(dim=0).cpu().double() * batch_size

    position_mse = position_squared_error / position_value_count
    metrics = {
        "total_loss": total_loss / sample_count,
        "position_mse": position_mse,
        "position_rmse_normalized": float(np.sqrt(position_mse)),
        "quaternion_error_deg": float(
            np.degrees(quaternion_error / quaternion_value_count)
        ),
        "score_mse": score_squared_error / score_value_count,
        "per_waypoint": {},
    }

    names = ("grasp", "initial", "goal")
    for waypoint_index, name in enumerate(names):
        waypoint_mse = float(per_waypoint_position[waypoint_index] / sample_count)
        waypoint_angle = float(
            np.degrees(per_waypoint_quaternion[waypoint_index] / sample_count)
        )
        metrics["per_waypoint"][name] = {
            "position_mse": waypoint_mse,
            "position_rmse_normalized": float(np.sqrt(waypoint_mse)),
            "quaternion_error_deg": waypoint_angle,
        }

    return metrics


def create_loaders(
    dataset: ReachingFineTuneDataset,
    batch_size: int,
    val_fraction: float,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, list[int], list[int]]:
    indices = np.arange(len(dataset))
    tools = dataset.data_frame["selected_tool"].to_numpy()
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=tools,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        shuffle=True,
        **common_loader_args,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        shuffle=False,
        **common_loader_args,
    )
    return (
        train_loader,
        val_loader,
        train_indices.tolist(),
        val_indices.tolist(),
    )


def save_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(DATASET_ROOT)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.encoder_checkpoint.exists():
        raise FileNotFoundError(args.encoder_checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_directory = OUTPUT_ROOT / (args.run_name or args.mode)
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset = ReachingFineTuneDataset(DATASET_ROOT)
    train_loader, val_loader, train_indices, val_indices = create_loaders(
        dataset,
        args.batch_size,
        args.val_fraction,
        args.seed,
        args.num_workers,
    )

    split = {
        "seed": args.seed,
        "train_indices": train_indices,
        "validation_indices": val_indices,
    }
    save_json(output_directory / "split.json", split)

    model = WayTuUnifiedModel(model_config(), device).to(device)
    state_dict = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(state_dict, strict=True)
    encoder_state_dict = load_checkpoint(args.encoder_checkpoint, device)
    model.encoder.load_state_dict(encoder_state_dict, strict=True)

    parameter_groups = configure_trainable_parameters(model, args.mode)
    optimizer_groups = []
    for group in parameter_groups:
        learning_rate = (
            args.encoder_lr if group["name"] == "encoder" else args.head_lr
        )
        optimizer_groups.append(
            {"params": group["params"], "lr": learning_rate}
        )

    optimizer = torch.optim.Adam(
        optimizer_groups,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Encoder checkpoint: {args.encoder_checkpoint}")

    baseline_metrics = evaluate(model, val_loader, device)
    save_json(output_directory / "baseline_metrics.json", baseline_metrics)
    print("Baseline validation metrics:")
    print(json.dumps(baseline_metrics, indent=2))

    if args.dry_run:
        print("Dry run completed successfully.")
        return

    best_validation_loss = float("inf")
    patience_counter = 0
    history = []
    best_checkpoint_path = output_directory / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, args.mode
        )
        validation_metrics = evaluate(model, val_loader, device)
        scheduler.step(validation_metrics["total_loss"])

        record = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total"],
            "train_score_loss": train_metrics["score"],
            "train_position_loss": train_metrics["position"],
            "train_quaternion_loss": train_metrics["quaternion"],
            "validation_total_loss": validation_metrics["total_loss"],
            "validation_position_mse": validation_metrics["position_mse"],
            "validation_quaternion_error_deg": validation_metrics[
                "quaternion_error_deg"
            ],
            "validation_score_mse": validation_metrics["score_mse"],
            "head_learning_rate": optimizer.param_groups[0]["lr"],
        }
        if len(optimizer.param_groups) > 1:
            record["encoder_learning_rate"] = optimizer.param_groups[1]["lr"]
        history.append(record)
        pd.DataFrame(history).to_csv(
            output_directory / "history.csv", index=False
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics['total']:.6f} | "
            f"val={validation_metrics['total_loss']:.6f} | "
            f"pos_mse={validation_metrics['position_mse']:.6f} | "
            f"quat_deg={validation_metrics['quaternion_error_deg']:.3f} | "
            f"score_mse={validation_metrics['score_mse']:.6f}"
        )

        if validation_metrics["total_loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["total_loss"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": args.mode,
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "source_checkpoint": str(args.checkpoint),
                    "source_encoder_checkpoint": str(
                        args.encoder_checkpoint
                    ),
                },
                best_checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    best_state = load_checkpoint(best_checkpoint_path, device)
    model.load_state_dict(best_state, strict=True)
    final_metrics = evaluate(model, val_loader, device)
    summary = {
        "mode": args.mode,
        "source_checkpoint": str(args.checkpoint),
        "source_encoder_checkpoint": str(args.encoder_checkpoint),
        "dataset_size": len(dataset),
        "training_size": len(train_indices),
        "validation_size": len(val_indices),
        "baseline_metrics": baseline_metrics,
        "best_metrics": final_metrics,
        "best_checkpoint": str(best_checkpoint_path),
    }
    save_json(output_directory / "summary.json", summary)

    print("Best validation metrics:")
    print(json.dumps(final_metrics, indent=2))
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
