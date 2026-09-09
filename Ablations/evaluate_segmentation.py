#!/usr/bin/env python3
"""Evaluate Way-Tu semantic segmentation on raw XYZ-label point clouds."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_max_pool, knn_graph
from tqdm import tqdm


PROJECT_ROOT = Path("/home/ece/git/WayTu-002")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models/segmentation_distractor_v9-best.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "segmentation_evaluation"

CLASS_NAMES = (
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
)
PLATFORM_IDS = (0, 1, 2)
TOOL_IDS = tuple(range(3, len(CLASS_NAMES)))


def configure_imports() -> None:
    candidates = (PROJECT_ROOT / "WayTu_Model", PROJECT_ROOT)
    for candidate in candidates:
        if (candidate / "model_utils.py").exists():
            sys.path.insert(0, str(candidate))
            return
    raise FileNotFoundError(
        "model_utils.py was not found in WayTu_Model or the project root."
    )


configure_imports()
import model_utils as mutils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        required=True,
        help="Folder containing data_*_pc.npy files. May be repeated.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--feature-knn", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--recompute-cache",
        action="store_true",
        help="Ignore cached point features and recompute them.",
    )
    return parser.parse_args()


class GraphPointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="max")
        self.mlp = Sequential(
            Linear(in_channels + 3 + 1, out_channels),
            nn.ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(
        self, features: torch.Tensor, positions: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        return self.propagate(edge_index, h=features, pos=positions)

    def message(
        self, h_j: torch.Tensor, pos_j: torch.Tensor, pos_i: torch.Tensor
    ) -> torch.Tensor:
        edge_vector = pos_j - pos_i
        edge_length = torch.norm(edge_vector, dim=1, keepdim=True)
        return self.mlp(torch.cat([h_j, edge_vector, edge_length], dim=-1))


class GraphPointNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = GraphPointNetLayer(42, 64)
        self.conv2 = GraphPointNetLayer(64, 256)
        self.conv3 = GraphPointNetLayer(512, 512)
        self.segmentation = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

    def forward(self, graph: Data) -> torch.Tensor:
        positions = graph.x[:, :3]
        encoded_positions = self.pos_encoder(positions)
        features = torch.cat([graph.x, encoded_positions], dim=1)
        batch = graph.batch
        if batch is None:
            batch = torch.zeros(
                features.size(0), dtype=torch.long, device=features.device
            )

        features = F.relu(self.conv1(features, positions, graph.edge_index))
        features = F.relu(self.conv2(features, positions, graph.edge_index))
        global_features = global_max_pool(features, batch)[batch]
        features = torch.cat([features, global_features], dim=1)
        features = F.relu(self.conv3(features, positions, graph.edge_index))
        global_features = global_max_pool(features, batch)[batch]
        combined = torch.cat([features, global_features], dim=1)
        return self.segmentation(combined)


def load_state_dict(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary.")

    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break

    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


def compute_normals_and_curvature(
    points: np.ndarray, knn: int
) -> tuple[np.ndarray, np.ndarray]:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    )
    point_cloud.orient_normals_towards_camera_location(
        camera_location=[0.0, 0.0, 10.0]
    )

    normals = np.asarray(point_cloud.normals)
    tree = o3d.geometry.KDTreeFlann(point_cloud)
    curvatures = []
    for index in range(len(points)):
        _, neighbor_indices, _ = tree.search_knn_vector_3d(
            point_cloud.points[index], knn
        )
        neighbors = points[neighbor_indices]
        covariance = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        curvature = eigenvalues[0] / (eigenvalues.sum() + 1e-6)
        curvatures.append(curvature)
    return normals, np.asarray(curvatures)[:, None]


def compute_eigen_features(points: np.ndarray, knn: int) -> np.ndarray:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(point_cloud)
    features = []

    for index in range(len(points)):
        _, neighbor_indices, _ = tree.search_knn_vector_3d(
            point_cloud.points[index], knn
        )
        neighbors = points[neighbor_indices]
        centered = neighbors - neighbors.mean(axis=0)
        covariance = np.cov(centered.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1] + 1e-10
        first, second, third = eigenvalues
        features.append(
            [
                (first - second) / first,
                (second - third) / first,
                third / first,
            ]
        )
    return np.asarray(features)


def cache_path_for(
    source_path: Path, data_root: Path, output_dir: Path
) -> Path:
    root_name = data_root.name or "dataset"
    return output_dir / "feature_cache" / root_name / f"{source_path.stem}.npz"


def load_or_compute_features(
    source_path: Path,
    data_root: Path,
    output_dir: Path,
    num_points: int,
    feature_knn: int,
    recompute: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = cache_path_for(source_path, data_root, output_dir)
    if cache_path.exists() and not recompute:
        cached = np.load(cache_path)
        return (
            cached["points"],
            cached["labels"],
            cached["normals"],
            cached["curvatures"],
            cached["eigen"],
        )

    raw = np.load(source_path)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError(
            f"Expected an N x 4 XYZ-label array in {source_path}, got {raw.shape}."
        )
    if not np.isfinite(raw).all():
        raise ValueError(f"Non-finite values found in {source_path}.")

    sampled = mutils.adaptive_farthest_point_sampling(raw, num_points)
    points = sampled[:, :3].astype(np.float64)
    labels = sampled[:, 3].astype(np.int64)
    invalid_labels = np.setdiff1d(np.unique(labels), np.arange(len(CLASS_NAMES)))
    if len(invalid_labels):
        raise ValueError(
            f"Labels outside the model range found in {source_path}: "
            f"{invalid_labels.tolist()}"
        )

    normals, curvatures = compute_normals_and_curvature(points, feature_knn)
    eigen = compute_eigen_features(points, feature_knn)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        points=points,
        labels=labels,
        normals=normals,
        curvatures=curvatures,
        eigen=eigen,
    )
    return points, labels, normals, curvatures, eigen


def create_graph(
    points: np.ndarray,
    normals: np.ndarray,
    curvatures: np.ndarray,
    eigen: np.ndarray,
    k_neighbors: int,
    device: torch.device,
) -> Data:
    point_tensor = torch.tensor(points, dtype=torch.float32)
    features = torch.cat(
        [
            point_tensor,
            torch.tensor(normals, dtype=torch.float32),
            torch.tensor(curvatures, dtype=torch.float32),
            torch.tensor(eigen, dtype=torch.float32),
        ],
        dim=1,
    )
    edge_index = knn_graph(point_tensor, k=k_neighbors)
    graph = Data(x=features, edge_index=edge_index)
    graph.batch = torch.zeros(len(points), dtype=torch.long)
    return graph.to(device)


def confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    encoded = targets * num_classes + predictions
    return np.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    support = matrix.sum(axis=1)
    union = true_positive + false_positive + false_negative

    iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan),
        where=union > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.full_like(true_positive, np.nan),
        where=support > 0,
    )
    precision_denominator = true_positive + false_positive
    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.full_like(true_positive, np.nan),
        where=precision_denominator > 0,
    )

    total = matrix.sum()
    overall_accuracy = true_positive.sum() / total if total else np.nan
    present = support > 0
    platform_present = present & np.isin(np.arange(len(CLASS_NAMES)), PLATFORM_IDS)
    tool_present = present & np.isin(np.arange(len(CLASS_NAMES)), TOOL_IDS)

    return {
        "overall_accuracy": float(overall_accuracy),
        "mean_iou": float(np.nanmean(iou[present])),
        "platform_mean_iou": (
            float(np.nanmean(iou[platform_present]))
            if platform_present.any()
            else None
        ),
        "tool_mean_iou": (
            float(np.nanmean(iou[tool_present])) if tool_present.any() else None
        ),
        "evaluated_points": int(total),
        "per_class_iou": iou,
        "per_class_recall": recall,
        "per_class_precision": precision,
        "per_class_support": support,
    }


def infer_task_name(labels: np.ndarray) -> str:
    platform_labels = [label for label in PLATFORM_IDS if np.any(labels == label)]
    if len(platform_labels) != 1:
        return "unknown"
    return CLASS_NAMES[platform_labels[0]].removesuffix("-platform")


def bootstrap_intervals(
    scene_matrices: list[np.ndarray], samples: int, seed: int
) -> dict:
    if not scene_matrices or samples <= 0:
        return {}

    generator = np.random.default_rng(seed)
    accuracies = []
    mean_ious = []
    scene_count = len(scene_matrices)
    for _ in range(samples):
        indices = generator.integers(0, scene_count, size=scene_count)
        matrix = np.sum([scene_matrices[index] for index in indices], axis=0)
        metrics = metrics_from_confusion(matrix)
        accuracies.append(metrics["overall_accuracy"])
        mean_ious.append(metrics["mean_iou"])

    return {
        "overall_accuracy_95_ci": [
            float(np.percentile(accuracies, 2.5)),
            float(np.percentile(accuracies, 97.5)),
        ],
        "mean_iou_95_ci": [
            float(np.percentile(mean_ious, 2.5)),
            float(np.percentile(mean_ious, 97.5)),
        ],
    }


def serializable_summary(
    metrics: dict, scene_count: int, intervals: dict | None = None
) -> dict:
    summary = {
        "scene_count": scene_count,
        "evaluated_points": metrics["evaluated_points"],
        "overall_accuracy": metrics["overall_accuracy"],
        "mean_iou": metrics["mean_iou"],
        "platform_mean_iou": metrics["platform_mean_iou"],
        "tool_mean_iou": metrics["tool_mean_iou"],
    }
    if intervals:
        summary.update(intervals)
    return summary


def save_results(
    output_dir: Path,
    total_matrix: np.ndarray,
    task_matrices: dict[str, list[np.ndarray]],
    scene_records: list[dict],
    checkpoint: Path,
    bootstrap_samples: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_metrics = metrics_from_confusion(total_matrix)
    all_scene_matrices = [record.pop("confusion_matrix") for record in scene_records]
    intervals = bootstrap_intervals(all_scene_matrices, bootstrap_samples, seed)

    summary = {
        "checkpoint": str(checkpoint),
        "sampling": "existing adaptive farthest point sampling pipeline",
        "overall": serializable_summary(
            total_metrics, len(scene_records), intervals
        ),
        "by_task": {},
    }

    for task_name, matrices in sorted(task_matrices.items()):
        task_matrix = np.sum(matrices, axis=0)
        task_metrics = metrics_from_confusion(task_matrix)
        task_intervals = bootstrap_intervals(
            matrices, bootstrap_samples, seed
        )
        summary["by_task"][task_name] = serializable_summary(
            task_metrics, len(matrices), task_intervals
        )

    per_class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        per_class_rows.append(
            {
                "class_id": class_index,
                "class_name": class_name,
                "support": int(total_metrics["per_class_support"][class_index]),
                "iou": total_metrics["per_class_iou"][class_index],
                "precision": total_metrics["per_class_precision"][class_index],
                "recall": total_metrics["per_class_recall"][class_index],
            }
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    pd.DataFrame(per_class_rows).to_csv(
        output_dir / "per_class_metrics.csv", index=False
    )
    pd.DataFrame(scene_records).to_csv(
        output_dir / "per_scene_metrics.csv", index=False
    )
    pd.DataFrame(total_matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    print(json.dumps(summary, indent=2))
    print(f"Results saved to: {output_dir}")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)

    for data_root in args.data_root:
        if not data_root.exists():
            raise FileNotFoundError(data_root)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    samples = []
    for data_root in args.data_root:
        files = sorted(data_root.glob("data_*_pc.npy"))
        if not files:
            raise FileNotFoundError(f"No data_*_pc.npy files found in {data_root}")
        samples.extend((data_root, file_path) for file_path in files)

    model = GraphPointNet(len(CLASS_NAMES)).to(device)
    model.load_state_dict(load_state_dict(args.checkpoint, device), strict=True)
    model.eval()

    total_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    task_matrices = defaultdict(list)
    scene_records = []

    with torch.no_grad():
        for data_root, source_path in tqdm(samples, desc="Evaluate scenes"):
            points, labels, normals, curvatures, eigen = load_or_compute_features(
                source_path=source_path,
                data_root=data_root,
                output_dir=args.output_dir,
                num_points=args.num_points,
                feature_knn=args.feature_knn,
                recompute=args.recompute_cache,
            )
            graph = create_graph(
                points,
                normals,
                curvatures,
                eigen,
                args.k_neighbors,
                device,
            )
            logits = model(graph)
            predictions = logits.argmax(dim=1).cpu().numpy()
            scene_matrix = confusion_matrix(
                labels, predictions, len(CLASS_NAMES)
            )
            scene_metrics = metrics_from_confusion(scene_matrix)
            task_name = infer_task_name(labels)

            total_matrix += scene_matrix
            task_matrices[task_name].append(scene_matrix)
            scene_records.append(
                {
                    "file": str(source_path),
                    "task": task_name,
                    "point_count": len(labels),
                    "overall_accuracy": scene_metrics["overall_accuracy"],
                    "mean_iou": scene_metrics["mean_iou"],
                    "confusion_matrix": scene_matrix,
                }
            )

    save_results(
        output_dir=args.output_dir,
        total_matrix=total_matrix,
        task_matrices=task_matrices,
        scene_records=scene_records,
        checkpoint=args.checkpoint,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
