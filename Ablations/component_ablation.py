#!/usr/bin/env python3
"""Controlled component ablations for the Way-Tu unified model.

This script keeps the dataset, train/validation split, frozen point-cloud
encoder, generator/selector heads, optimizer, and loss weights fixed while
changing one component at a time.

Supported variants
------------------
full
    Unmodified unified Way-Tu model.
no_env
    Removes only the learned environment point-cloud embedding. The explicit
    target center is retained as the minimum task specification.
no_task
    Removes the task one-hot vector supplied to the selector.
no_yaw
    Removes the explicit yaw scalar predicted by the frozen encoder from both
    tool and environment embeddings. This does not ablate the encoder's
    auxiliary yaw-training objective; doing that requires retraining the
    feature extractor itself.
grasp_score_only
    Trains the unified model with grasp_score as the selector target.
task_score_only
    Trains the unified model with task_score as the selector target.

The script intentionally does not include:
  * selector-vs-random selection, which is an inference/execution comparison;
  * KOMO removal, which requires an alternative trajectory generator;
  * removal of auxiliary yaw supervision, which requires encoder retraining.

Place this file beside generator_and_selector_original.py, feature_extractor.py,
waypoint_generator.py, and tool_selection.py. Run it from the repository root.
"""

import argparse
import importlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


LABELS = [
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
]

SUPPORTED_ABLATIONS = (
    "full",
    "no_env",
    "no_task",
    "no_yaw",
    "grasp_score_only",
    "task_score_only",
)

WAYPOINT_NAMES = ("grasp", "initial", "goal")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_ablation_list(value):
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if requested == ["all"]:
        return list(SUPPORTED_ABLATIONS)

    unknown = sorted(set(requested) - set(SUPPORTED_ABLATIONS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown ablation(s): {unknown}. Choose from "
            f"{list(SUPPORTED_ABLATIONS)} or 'all'."
        )
    if not requested:
        raise argparse.ArgumentTypeError("At least one ablation is required.")
    return requested


def dataframe_for_ablation(dataframe, ablation):
    """Select the score supervision while leaving all samples unchanged."""
    dataframe = dataframe.copy()
    if ablation == "no_task" and dataframe["task"].nunique() < 2:
        raise ValueError(
            "The no_task ablation is not meaningful on a single-task dataset: "
            "the task encoding is constant and can be absorbed as a bias. Run "
            "this variant only with the same multi-task dataset used to train "
            "the unified multi-task model."
        )
    if ablation == "grasp_score_only":
        if "grasp_score" not in dataframe.columns:
            raise KeyError("dataset_info.csv has no 'grasp_score' column.")
        dataframe["score"] = dataframe["grasp_score"]
    elif ablation == "task_score_only":
        if "task_score" not in dataframe.columns:
            raise KeyError("dataset_info.csv has no 'task_score' column.")
        dataframe["score"] = dataframe["task_score"]
    return dataframe


def build_model(base_module, cfg, device, ablation):
    """Create a model with exactly one input component removed."""

    class ComponentAblationModel(base_module.WayTuUnifiedModel):
        def __init__(self):
            super().__init__(cfg, device)
            self.ablation = ablation

        def embedding_with_optional_yaw(self, points, centers, scales):
            features, yaw = self.encoder(points)
            if self.ablation == "no_yaw":
                yaw = torch.zeros_like(yaw)
            return torch.cat(
                [features, centers, yaw.unsqueeze(1), scales.unsqueeze(1)],
                dim=1,
            )

        def forward(self, tool_points, env_points, task_onehot, params):
            tool_embedding = self.embedding_with_optional_yaw(
                tool_points,
                params["tool_centers"],
                params["tool_scales"],
            )

            if self.ablation == "no_env":
                # Keep target_center below, but remove all learned environment
                # point-cloud features and environment normalization metadata
                # from the generator input.
                env_embedding = tool_embedding.new_zeros(
                    (tool_embedding.size(0), self.embedding_size)
                )
            else:
                env_embedding = self.embedding_with_optional_yaw(
                    env_points,
                    params["env_centers"],
                    params["env_scales"],
                )

            generator_input = torch.cat(
                [tool_embedding, env_embedding, params["target_center"]], dim=1
            )
            positions, quaternions = self.generator(generator_input)

            grasp_waypoint = torch.cat(
                [positions[:, 0, :], quaternions[:, 0, :]], dim=1
            )
            if self.ablation == "no_task":
                task_onehot = torch.zeros_like(task_onehot)
            selector_input = torch.cat(
                [tool_embedding, task_onehot, grasp_waypoint], dim=1
            )
            score = self.selector(selector_input)
            return positions, quaternions, score

    return ComponentAblationModel().to(device)


def prepare_batch(batch, device):
    tool_points = batch["tool_points"].to(device)
    env_points = batch["env_points"].to(device)
    task_onehot = batch["env_encoding"].to(device)

    params = {
        "tool_centers": batch["tool_center"].to(device),
        "tool_scales": batch["tool_scale"].to(device),
        "env_centers": batch["env_center"].to(device),
        "env_scales": batch["env_scale"].to(device),
        "target_center": batch["target_center"].to(device),
    }

    ground_truth = torch.stack(
        [
            batch["tool_waypoint"],
            batch["initial_waypoint"],
            batch["goal_waypoint"],
        ],
        dim=1,
    ).to(device)

    targets = {
        "positions": ground_truth[..., :3],
        "quaternions": ground_truth[..., 3:],
        "score": batch["score"].to(device).clamp(0.0, 1.0),
    }
    return tool_points, env_points, task_onehot, params, targets


def loss_terms(base_module, positions, quaternions, score, targets):
    return {
        "position": F.mse_loss(positions, targets["positions"]),
        "quaternion": base_module.geodesic_loss_v3(
            quaternions, targets["quaternions"]
        ).mean(),
        "score": F.mse_loss(score, targets["score"]),
    }


@torch.no_grad()
def evaluate(model, loader, device, base_module):
    model.eval()
    model.encoder.eval()

    sample_count = 0
    position_squared_error = torch.zeros(3, dtype=torch.float64)
    quaternion_error_rad = torch.zeros(3, dtype=torch.float64)
    score_squared_error = 0.0

    for batch in loader:
        tool, env, task, params, targets = prepare_batch(batch, device)
        positions, quaternions, score = model(tool, env, task, params)
        batch_size = tool.size(0)

        # Mean XYZ squared error for each waypoint, summed over samples.
        per_waypoint_position = (
            (positions - targets["positions"]).pow(2).mean(dim=-1).sum(dim=0)
        )
        per_waypoint_quaternion = base_module.geodesic_loss_v3(
            quaternions, targets["quaternions"]
        ).sum(dim=0)

        position_squared_error += per_waypoint_position.detach().cpu().double()
        quaternion_error_rad += per_waypoint_quaternion.detach().cpu().double()
        score_squared_error += (
            (score - targets["score"]).pow(2).sum().item()
        )
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Validation loader is empty.")

    position_mse_per_waypoint = position_squared_error / sample_count
    quaternion_rad_per_waypoint = quaternion_error_rad / sample_count

    # These aggregate definitions match a mean over all waypoint coordinates
    # and a mean over all waypoint orientation errors, respectively.
    position_mse = position_mse_per_waypoint.mean().item()
    quaternion_rad = quaternion_rad_per_waypoint.mean().item()

    per_waypoint = {}
    for index, name in enumerate(WAYPOINT_NAMES):
        per_waypoint[name] = {
            "position_mse": position_mse_per_waypoint[index].item(),
            "position_rmse_normalized": math.sqrt(
                position_mse_per_waypoint[index].item()
            ),
            "quaternion_error_deg": (
                quaternion_rad_per_waypoint[index].item() * 180.0 / math.pi
            ),
        }

    return {
        "position_mse": position_mse,
        "position_rmse_normalized": math.sqrt(position_mse),
        "quaternion_error_deg": quaternion_rad * 180.0 / math.pi,
        "score_mse": score_squared_error / sample_count,
        "per_waypoint": per_waypoint,
    }


def validation_objective(metrics, cfg):
    return (
        cfg["lambda_pos"] * metrics["position_mse"]
        + cfg["lambda_quat"]
        * metrics["quaternion_error_deg"]
        * math.pi
        / 180.0
        + cfg["lambda_score"] * metrics["score_mse"]
    )


def train_variant(
    model,
    train_loader,
    val_loader,
    cfg,
    device,
    base_module,
    checkpoint_path,
    ablation,
):
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    best_validation = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        # The encoder is frozen and must remain in evaluation mode even while
        # the trainable heads are in training mode.
        model.encoder.eval()

        running = {"total": 0.0, "position": 0.0, "quaternion": 0.0, "score": 0.0}
        sample_count = 0

        for batch in tqdm(
            train_loader,
            desc=f"{ablation} {epoch:03d}",
            leave=False,
        ):
            tool, env, task, params, targets = prepare_batch(batch, device)
            optimizer.zero_grad()
            positions, quaternions, score = model(tool, env, task, params)
            terms = loss_terms(
                base_module, positions, quaternions, score, targets
            )
            loss = (
                cfg["lambda_pos"] * terms["position"]
                + cfg["lambda_quat"] * terms["quaternion"]
                + cfg["lambda_score"] * terms["score"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = tool.size(0)
            running["total"] += loss.item() * batch_size
            for name in ("position", "quaternion", "score"):
                running[name] += terms[name].item() * batch_size
            sample_count += batch_size

        validation_metrics = evaluate(model, val_loader, device, base_module)
        validation_total = validation_objective(validation_metrics, cfg)
        epoch_record = {
            "epoch": epoch,
            "train_total": running["total"] / sample_count,
            "train_position": running["position"] / sample_count,
            "train_quaternion": running["quaternion"] / sample_count,
            "train_score": running["score"] / sample_count,
            "validation_objective": validation_total,
            "validation_metrics": validation_metrics,
        }
        history.append(epoch_record)

        print(
            f"{ablation} epoch {epoch:03d}: "
            f"train={epoch_record['train_total']:.6f}, "
            f"val={validation_total:.6f}, "
            f"pos_rmse={validation_metrics['position_rmse_normalized']:.6f}, "
            f"quat_deg={validation_metrics['quaternion_error_deg']:.3f}, "
            f"score_mse={validation_metrics['score_mse']:.6f}"
        )

        if validation_total < best_validation:
            best_validation = validation_total
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience_es"]:
                print(f"{ablation}: early stopping.")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    final_metrics = evaluate(model, val_loader, device, base_module)
    return final_metrics, history


def create_split(dataset_size, validation_fraction, seed, split_file=None):
    if split_file is not None:
        with open(split_file, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        train_indices = saved["train"]
        validation_indices = saved.get("validation", saved.get("val"))
        if validation_indices is None:
            raise KeyError(
                "Split file must contain 'validation' (or 'val') indices."
            )
    else:
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(dataset_size, generator=generator).tolist()
        validation_size = max(1, int(dataset_size * validation_fraction))
        validation_indices = permutation[:validation_size]
        train_indices = permutation[validation_size:]

    all_indices = train_indices + validation_indices
    if not all_indices or min(all_indices) < 0 or max(all_indices) >= dataset_size:
        raise ValueError(
            "Split indices do not match the current dataset size "
            f"({dataset_size})."
        )
    if set(train_indices).intersection(validation_indices):
        raise ValueError("Train and validation splits overlap.")
    return train_indices, validation_indices


def build_loaders(dataset, train_indices, validation_indices, cfg, seed, num_workers):
    train_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)
    shuffle_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        generator=shuffle_generator,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train controlled Way-Tu component ablations."
    )
    parser.add_argument(
        "--ablations",
        type=parse_ablation_list,
        default=["no_env"],
        help=(
            "Comma-separated variants, e.g. no_env,no_task, or 'all'. "
            "Default: no_env"
        ),
    )
    parser.add_argument(
        "--dataset-dir", default="Datasets/minigolf-dataset-top1800-pc"
    )
    parser.add_argument(
        "--feature-extractor-path",
        default="small-pointner-encoder-distractor_best.pth",
    )
    parser.add_argument(
        "--base-module", default="generator_and_selector_original"
    )
    parser.add_argument("--output-dir", default="component_ablation_minigolf")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing variant checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs("logs", exist_ok=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_module = importlib.import_module(args.base_module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ablations: {args.ablations}")

    cfg = {
        "dataset_dir": args.dataset_dir,
        "feature-extractor-path": args.feature_extractor_path,
        "label-list-all": LABELS,
        "num-training-points": 512,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "val_split": 0.2,
        "patience_es": args.patience,
        "feature-size": 128,
        "lambda_score": 0.1,
        "lambda_pos": 0.6,
        "lambda_quat": 0.4,
        "fine-tune": False,
        "multi-task": False,
        "data_augmentation": False,
    }

    source_dataframe = pd.read_csv(
        os.path.join(cfg["dataset_dir"], "dataset_info.csv")
    )
    train_indices, validation_indices = create_split(
        len(source_dataframe),
        cfg["val_split"],
        args.seed,
        split_file=args.split_file,
    )

    split_output = output_dir / "split_indices.json"
    with open(split_output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": args.seed,
                "train": train_indices,
                "validation": validation_indices,
            },
            handle,
        )

    all_results = {}
    for ablation in args.ablations:
        print(f"\n=== Training ablation: {ablation} ===")
        variant_dir = output_dir / ablation
        variant_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = variant_dir / "model_best.pth"
        if checkpoint_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint_path}. "
                "Use --overwrite or choose a new --output-dir."
            )

        dataframe = dataframe_for_ablation(source_dataframe, ablation)
        dataset = base_module.WayTuUnifiedDataset(cfg, dataframe, device)
        train_loader, validation_loader = build_loaders(
            dataset,
            train_indices,
            validation_indices,
            cfg,
            args.seed,
            args.num_workers,
        )

        set_seed(args.seed)
        model = build_model(base_module, cfg, device, ablation)
        final_metrics, history = train_variant(
            model,
            train_loader,
            validation_loader,
            cfg,
            device,
            base_module,
            checkpoint_path,
            ablation,
        )

        metadata = {
            "ablation": ablation,
            "dataset_dir": cfg["dataset_dir"],
            "score_target": (
                "grasp_score"
                if ablation == "grasp_score_only"
                else "task_score"
                if ablation == "task_score_only"
                else "score"
            ),
            "target_center_retained": True,
            "seed": args.seed,
            "train_size": len(train_indices),
            "validation_size": len(validation_indices),
            "config": cfg,
            "final_validation_metrics": final_metrics,
        }
        with open(variant_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        with open(variant_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        all_results[ablation] = final_metrics
        with open(
            output_dir / "validation_metrics.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(all_results, handle, indent=2)

        # Release GPU memory before the next variant.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nFinal validation metrics")
    print(json.dumps(all_results, indent=2))
    print(f"Outputs saved under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
