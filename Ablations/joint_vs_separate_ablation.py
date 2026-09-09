#!/usr/bin/env python3
"""Train a controlled joint-vs-separate Way-Tu ablation.

The comparison keeps the dataset, frozen feature extractor, waypoint head,
selector head, split, and hyperparameters fixed.

Joint model:
    score <- tool embedding + task one-hot + predicted grasp waypoint
    generator and selector are optimized with one combined loss.

Separate model:
    score <- tool embedding + task one-hot
    phase 1 trains only the waypoint generator;
    phase 2 freezes the generator and trains only the selector.

Place this file next to generator_and_selector_original.py, feature_extractor.py,
waypoint_generator.py, and tool_selection.py, then run it from that directory.
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_trainable(module, value):
    for parameter in module.parameters():
        parameter.requires_grad = value


class SeparateWayTuModel(nn.Module):
    """Independent waypoint and selection branches with a frozen encoder."""

    def __init__(self, cfg, device, base_module):
        super().__init__()
        self.device = device
        self.encoder = base_module.SmallPointNetEncoderYaw(
            out_dim=cfg["feature-size"]
        ).to(device)

        encoder_path = os.path.join("models", cfg["feature-extractor-path"])
        print(f"Loading frozen feature extractor from {encoder_path}")
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        self.encoder.eval()
        set_trainable(self.encoder, False)

        self.embedding_size = cfg["feature-size"] + 3 + 1 + 1
        waypoint_input_size = self.embedding_size * 2 + 3

        self.generator = base_module.WaypointGeneratorHead(
            cfg, num_waypoints=3, feature_size=waypoint_input_size
        )

        # waypoint_dim=0 is the key ablation: selection is not conditioned on
        # a ground-truth or predicted manipulation waypoint.
        self.selector = base_module.ToolSelectionNetwork(
            tool_feat_dim=self.embedding_size,
            task_dim=3,
            waypoint_dim=0,
        )

    def get_embedding_vector(self, points, centers, scales):
        features, yaw = self.encoder(points)
        return torch.cat(
            [features, centers, yaw.unsqueeze(1), scales.unsqueeze(1)], dim=1
        )

    def encode(self, tool_points, env_points, params):
        tool_embedding = self.get_embedding_vector(
            tool_points, params["tool_centers"], params["tool_scales"]
        )
        env_embedding = self.get_embedding_vector(
            env_points, params["env_centers"], params["env_scales"]
        )
        return tool_embedding, env_embedding

    def predict_waypoints(self, tool_points, env_points, params):
        tool_embedding, env_embedding = self.encode(tool_points, env_points, params)
        generator_input = torch.cat(
            [tool_embedding, env_embedding, params["target_center"]], dim=1
        )
        return self.generator(generator_input)

    def predict_score(self, tool_points, env_onehot, params):
        tool_embedding = self.get_embedding_vector(
            tool_points, params["tool_centers"], params["tool_scales"]
        )
        selector_input = torch.cat([tool_embedding, env_onehot], dim=1)
        return self.selector(selector_input)

    def forward(self, tool_points, env_points, env_onehot, params):
        positions, quaternions = self.predict_waypoints(
            tool_points, env_points, params
        )
        score = self.predict_score(tool_points, env_onehot, params)
        return positions, quaternions, score


def prepare_batch(batch, device):
    tool_points = batch["tool_points"].to(device)
    env_points = batch["env_points"].to(device)
    env_encoding = batch["env_encoding"].to(device)

    params = {
        "tool_centers": batch["tool_center"].to(device),
        "tool_scales": batch["tool_scale"].to(device),
        "env_centers": batch["env_center"].to(device),
        "env_scales": batch["env_scale"].to(device),
        "target_center": batch["target_center"].to(device),
    }

    gt_waypoints = torch.stack(
        [
            batch["tool_waypoint"],
            batch["initial_waypoint"],
            batch["goal_waypoint"],
        ],
        dim=1,
    ).to(device)

    targets = {
        "positions": gt_waypoints[..., :3],
        "quaternions": gt_waypoints[..., 3:],
        "score": batch["score"].to(device).clamp(0.0, 1.0),
    }
    return tool_points, env_points, env_encoding, params, targets


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
    count = 0
    totals = {"position_mse": 0.0, "quaternion_rad": 0.0, "score_mse": 0.0}

    for batch in loader:
        tool, env, task, params, targets = prepare_batch(batch, device)
        positions, quaternions, score = model(tool, env, task, params)
        batch_size = tool.size(0)

        totals["position_mse"] += (
            F.mse_loss(positions, targets["positions"]).item() * batch_size
        )
        totals["quaternion_rad"] += (
            base_module.geodesic_loss_v3(
                quaternions, targets["quaternions"]
            ).mean().item()
            * batch_size
        )
        totals["score_mse"] += (
            F.mse_loss(score, targets["score"]).item() * batch_size
        )
        count += batch_size

    position_mse = totals["position_mse"] / count
    quaternion_rad = totals["quaternion_rad"] / count
    return {
        "position_mse": position_mse,
        "position_rmse_normalized": math.sqrt(position_mse),
        "quaternion_error_deg": quaternion_rad * 180.0 / math.pi,
        "score_mse": totals["score_mse"] / count,
    }


def train_joint(model, train_loader, val_loader, cfg, device, base_module, output):
    set_trainable(model.generator, True)
    set_trainable(model.selector, True)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    best = float("inf")
    patience_counter = 0
    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        model.encoder.eval()
        running = 0.0
        count = 0

        for batch in tqdm(train_loader, desc=f"Joint {epoch:03d}", leave=False):
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
            running += loss.item() * tool.size(0)
            count += tool.size(0)

        metrics = evaluate(model, val_loader, device, base_module)
        val_total = (
            cfg["lambda_pos"] * metrics["position_mse"]
            + cfg["lambda_quat"]
            * metrics["quaternion_error_deg"]
            * math.pi
            / 180.0
            + cfg["lambda_score"] * metrics["score_mse"]
        )
        print(
            f"Joint epoch {epoch:03d}: train={running/count:.6f}, "
            f"val={val_total:.6f}, metrics={metrics}"
        )

        if val_total < best:
            best = val_total
            patience_counter = 0
            torch.save(model.state_dict(), output)
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience_es"]:
                print("Joint early stopping.")
                break

    model.load_state_dict(torch.load(output, map_location=device))
    return evaluate(model, val_loader, device, base_module)


def train_separate(model, train_loader, val_loader, cfg, device, base_module, output_dir):
    generator_output = output_dir / "separate_generator_best.pth"
    final_output = output_dir / "separate_model_best.pth"

    # Phase 1: waypoint prediction only.
    set_trainable(model.generator, True)
    set_trainable(model.selector, False)
    optimizer = torch.optim.Adam(
        model.generator.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    best = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        model.encoder.eval()
        model.selector.eval()

        for batch in tqdm(train_loader, desc=f"Separate generator {epoch:03d}", leave=False):
            tool, env, _, params, targets = prepare_batch(batch, device)
            optimizer.zero_grad()
            positions, quaternions = model.predict_waypoints(tool, env, params)
            pos_loss = F.mse_loss(positions, targets["positions"])
            quat_loss = base_module.geodesic_loss_v3(
                quaternions, targets["quaternions"]
            ).mean()
            loss = cfg["lambda_pos"] * pos_loss + cfg["lambda_quat"] * quat_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            optimizer.step()

        metrics = evaluate(model, val_loader, device, base_module)
        val_waypoint = (
            cfg["lambda_pos"] * metrics["position_mse"]
            + cfg["lambda_quat"]
            * metrics["quaternion_error_deg"]
            * math.pi
            / 180.0
        )
        print(f"Separate generator epoch {epoch:03d}: val={val_waypoint:.6f}")
        if val_waypoint < best:
            best = val_waypoint
            patience_counter = 0
            torch.save(model.generator.state_dict(), generator_output)
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience_es"]:
                print("Separate generator early stopping.")
                break

    model.generator.load_state_dict(torch.load(generator_output, map_location=device))

    # Phase 2: tool selection only; generator remains frozen and is not used by
    # the selector.
    set_trainable(model.generator, False)
    set_trainable(model.selector, True)
    optimizer = torch.optim.Adam(
        model.selector.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    best = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        model.encoder.eval()
        model.generator.eval()

        for batch in tqdm(train_loader, desc=f"Separate selector {epoch:03d}", leave=False):
            tool, _, task, params, targets = prepare_batch(batch, device)
            optimizer.zero_grad()
            score = model.predict_score(tool, task, params)
            loss = F.mse_loss(score, targets["score"])
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, val_loader, device, base_module)
        print(
            f"Separate selector epoch {epoch:03d}: "
            f"score_mse={metrics['score_mse']:.6f}"
        )
        if metrics["score_mse"] < best:
            best = metrics["score_mse"]
            patience_counter = 0
            torch.save(model.state_dict(), final_output)
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience_es"]:
                print("Separate selector early stopping.")
                break

    model.load_state_dict(torch.load(final_output, map_location=device))
    return evaluate(model, val_loader, device, base_module), final_output


def build_loaders(cfg, base_module, device, seed, num_workers):
    dataframe = pd.read_csv(os.path.join(cfg["dataset_dir"], "dataset_info.csv"))
    dataset = base_module.WayTuUnifiedDataset(cfg, dataframe, device)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(dataset), generator=generator).tolist()
    val_size = max(1, int(len(dataset) * cfg["val_split"]))
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, train_indices, val_indices


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--mode", choices=["joint", "separate", "both"], default="both")
    parser.add_argument("--output-dir", default="ablation_minigolf")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
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

    train_loader, val_loader, train_indices, val_indices = build_loaders(
        cfg, base_module, device, args.seed, args.num_workers
    )
    with open(output_dir / "split_indices.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"seed": args.seed, "train": train_indices, "validation": val_indices},
            handle,
        )

    results = {}
    if args.mode in {"joint", "both"}:
        set_seed(args.seed)
        joint_model = base_module.WayTuUnifiedModel(cfg, device).to(device)
        joint_output = output_dir / "joint_model_best.pth"
        results["joint"] = train_joint(
            joint_model,
            train_loader,
            val_loader,
            cfg,
            device,
            base_module,
            joint_output,
        )

    if args.mode in {"separate", "both"}:
        set_seed(args.seed)
        separate_model = SeparateWayTuModel(cfg, device, base_module).to(device)
        results["separate"], separate_output = train_separate(
            separate_model,
            train_loader,
            val_loader,
            cfg,
            device,
            base_module,
            output_dir,
        )

    with open(output_dir / "validation_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nFinal validation metrics")
    print(json.dumps(results, indent=2))
    print(f"Outputs saved under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
