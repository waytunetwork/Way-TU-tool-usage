import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from sklearn.metrics import confusion_matrix
from torch_geometric.nn import radius_graph
from sklearn.cluster import DBSCAN

def estimate_normals_and_curvature(env_pc, knn=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(env_pc)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_towards_camera_location(camera_location=[0.0, 0.0, 10.0])

    normals = np.asarray(pcd.normals)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = []
    for i in range(len(env_pc)):
        _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], knn)
        neighbors = np.asarray(pcd.points)[idx]
        cov = np.cov(neighbors.T)
        eigvals = np.linalg.eigvalsh(cov)
        curvature = eigvals[0] / (eigvals.sum() + 1e-6)
        curvatures.append(curvature)
    curvatures = np.expand_dims(np.array(curvatures), axis=1)
    return normals, curvatures

def compute_eigen_features(env_pc, knn=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(env_pc)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    features = []
    for i in range(len(env_pc)):
        _, idxs, _ = kdtree.search_knn_vector_3d(pcd.points[i], knn)
        neighbors = np.asarray(pcd.points)[idxs, :]
        neighbors_centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(neighbors_centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1] + 1e-10
        l1, l2, l3 = eigvals
        linearity = (l1 - l2) / l1
        planarity = (l2 - l3) / l1
        sphericity = l3 / l1
        features.append([linearity, planarity, sphericity])
    return np.array(features)

def build_graph_data(env_pc, cfg):
    pc_tensor = torch.tensor(env_pc, dtype=torch.float)
    edge_index = knn_graph(pc_tensor, k=cfg["k-neighbors"])

    normals, curvatures = estimate_normals_and_curvature(env_pc, knn=20)
    eigen_features = compute_eigen_features(env_pc, knn=20)

    normals_tensor = torch.tensor(normals, dtype=torch.float)
    curvatures_tensor = torch.tensor(curvatures, dtype=torch.float)
    eigen_tensor = torch.tensor(eigen_features, dtype=torch.float)

    node_features = torch.cat([pc_tensor, normals_tensor, curvatures_tensor, eigen_tensor], dim=1)
    return Data(x=node_features, edge_index=edge_index), pc_tensor

# Below the codes are related to debugging and visualization 

def draw_labeled_pointcloud(env_pc, env_label, title):
    xyz = np.ascontiguousarray(env_pc)  # x, y, z
    labels = np.ascontiguousarray(env_label) # labels
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(labels / unique_labels.max())[:, :3]  # Normalize labels and get RGB colors
    colors = np.ascontiguousarray(colors)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Showing {title}")
    o3d.visualization.draw_geometries([pcd])

def print_confusion_matrix(gt_labels, pred_labels, class_names):
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(len(class_names))))
    print("Confusion Matrix:")
    print("Rows = Ground Truth, Columns = Predicted")
    header = "     " + " ".join(f"{name[:4]:>5}" for name in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{val:5d}" for val in row)
        print(f"{class_names[i][:4]:>4}: {row_str}")

# The three function below is taken from the feature extractor dataset 
def outlier_removal(point_cloud: torch.Tensor, radius: float, min_neighbors: int = 1):
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

def normalize_point_cloud(point_cloud: torch.Tensor):
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

def arrange_point_cloud_size (point_cloud: torch.Tensor, num_points:int):
    Nf = point_cloud.shape[0]
    if Nf >= num_points:
        indices = torch.randperm(Nf)[: num_points]
        pts_fixed = point_cloud[indices, :]                       # (NUM_POINTS, 3)
    else:
        if Nf > 0:
            pad_count = num_points - Nf
            pad_idx = torch.randint(0, Nf, (pad_count,))
            pts_fixed = torch.cat([point_cloud, point_cloud[pad_idx, :]], dim=0)  # (NUM_POINTS, 3)
        else:
            # If no points after filtering, fill with zeros
            pts_fixed = torch.zeros((num_points, 3), dtype=torch.float32)
    return pts_fixed

# This function is a runtime heuristic for detecting target centers. 
def get_target_center(env_points, task, side = None):
    # lifting: 0, minigolf: 1, hammering: 2
    print("**************Task:", task)
    if task == 0:
        z = env_points[:, 2]
        z_min, z_max = z.min(), z.max()
        z_mid = (z_min + z_max) / 2.0
        upper_points = env_points[z>z_mid]

        x_min, x_max = upper_points[:, 0].min(), upper_points[:, 0].max()
        y_min, y_max = upper_points[:, 1].min(), upper_points[:, 1].max()

        center_xy = torch.stack([(x_min + x_max) / 2.0,
                             (y_min + y_max) / 2.0]) 
        deltas = upper_points[:, :2] - center_xy
        dists = torch.norm(deltas, dim=1)
        idx = torch.argmin(dists)
        return upper_points[idx]
    
         # Create Open3D PointCloud for each
        pcd_lower = o3d.geometry.PointCloud()
        pcd_lower.points = o3d.utility.Vector3dVector(env_points.cpu().numpy())
        pcd_lower.paint_uniform_color([0.7, 0.7, 0.7])  # gray for lower

        # pcd_upper = o3d.geometry.PointCloud()
        # pcd_upper.points = o3d.utility.Vector3dVector(upper_points.cpu().numpy())
        # pcd_upper.paint_uniform_color([1, 0, 0])  # red for upper

        

        # Visualize both together
        o3d.visualization.draw_geometries(
            [pcd_lower, ], # pcd_upper
            window_name="Mid-Height Slice",
            width=800,
            height=600,
            point_show_normal=False
        )

    elif task == 1:
        percentile = 0.02
        z_vals = env_points[:, 2]
        k = max(int(env_points.size(0) * percentile), 1)
        topk_vals, topk_idx = torch.topk(z_vals, k, largest=True)
        ball_points = env_points[topk_idx]

        if ball_points.size(0) > 0:
            return ball_points.mean(dim=0)
        else:
            return env_points.mean(dim=0)
    elif task == 2: 
        quantile: float = 0.20
        knn_normals: int = 20
        cos_thresh: float = 0.9
        cluster_radius: float = 0.02

        N = env_points.shape[0]

        # 1) seed with the bottom quantile by Z
        zs       = env_points[:,2]
        z_thr    = torch.quantile(zs, quantile)
        bottom   = env_points[zs <= z_thr]                   # [M×3]
        M        = bottom.shape[0]

        if M < 3:
            return env_points.mean(dim=0)   # fallback to everything

        # 2) estimate normals via PCA on the bottom slice
        #   a) build the full pairwise distance matrix
        dists = torch.cdist(bottom, bottom)                 # [M×M]
        #   b) for each point, grab its K+1 nearest (including itself)
        idx   = torch.topk(dists, knn_normals+1, largest=False).indices  # [M×(K+1)]
        nbrs  = bottom[idx[:,1:], :]                        # drop self at [:,0] → [M×K×3]

        #   c) compute per-point covariance matrices
        mean_nbrs = nbrs.mean(dim=1, keepdim=True)          # [M×1×3]
        Xc        = nbrs - mean_nbrs                        # [M×K×3]
        covs      = (Xc.transpose(1,2) @ Xc) / (knn_normals-1)  # [M×3×3]

        #   d) PCA → normals = eigenvectors of smallest eigenvalue
        eigv, eigvecs = torch.linalg.eigh(covs)             # both [M×3]
        normals       = eigvecs[:,:,0]                      # [M×3]

        # 3) keep only those bottom‐slice points whose normals are near ±Z
        horiz_mask = normals[:,2].abs() > cos_thresh        # [M]
        horiz_pts  = bottom[horiz_mask]                     # [P×3]
        P          = horiz_pts.shape[0]
        if P == 0:
            return env_points.mean(dim=0)   # nothing horizontal → fallback

        # 4) build a connectivity graph and extract the largest component
        #    on the horizontal seeds
        d2 = torch.cdist(horiz_pts, horiz_pts)               # [P×P]
        adj = d2 < cluster_radius                           # [P×P]

        visited  = torch.zeros(P, dtype=torch.bool, device=env_points.device)
        clusters = []
        for i in range(P):
            if not visited[i]:
                stack, comp = [i], []
                while stack:
                    u = stack.pop()
                    if visited[u]:
                        continue
                    visited[u] = True
                    comp.append(u)
                    neigh = adj[u].nonzero().view(-1)
                    for v in neigh:
                        if not visited[v]:
                            stack.append(v.item())
                clusters.append(comp)

        # pick the largest cluster (the true platform)
        best = max(clusters, key=len)
        platform_pts = horiz_pts[best]

        # 1) platform centroid (still useful for the XY‐nearest step)
        plat_center = platform_pts.mean(dim=0)   # [3]

        # 2) platform top‐surface height
        plat_z_top  = platform_pts[:,2].max()    # scalar

        # 3) optionally also guard against weird global center noise
        env_center  = env_points.mean(dim=0)     # [3]

        # 4) build your “above” mask:
        #    – strictly above the *top* of the platform
        #    – and (if you like) above the cloud’s global center too
        above_mask = (
            (env_points[:,2] > plat_z_top) &
            (env_points[:,2] > env_center[2])
        )
        above_pts = env_points[above_mask]       # [M×3]

        if above_pts.shape[0] == 0:
            target_pt = torch.tensor([plat_center])  # fallback
        else:
            # 5) XY‐nearest to the platform center
            dx = above_pts[:,0] - plat_center[0]
            dy = above_pts[:,1] - plat_center[1]
            d2 = dx**2 + dy**2
            idx = torch.argmin(d2)
            target_pt = above_pts[idx].unsqueeze(0)   # [1×3]
        
        # tp = target_pt.cpu().numpy().reshape(1,3)
        # # 2) build a 1-point cloud
        # pcd_t = o3d.geometry.PointCloud()
        # pcd_t.points = o3d.utility.Vector3dVector(tp)
        # pcd_t.colors = o3d.utility.Vector3dVector(np.tile([1.0,0.8,0.0], (1,1)))  

        # p_all = env_points.cpu().numpy()
        # p_pl  = platform_pts.cpu().numpy()
        # pcd_all = o3d.geometry.PointCloud(); pcd_all.points = o3d.utility.Vector3dVector(p_all)
        # pcd_all.colors = o3d.utility.Vector3dVector(np.tile([0.7,0.7,0.7], (p_all.shape[0],1)))
        # pcd_pl  = o3d.geometry.PointCloud(); pcd_pl.points = o3d.utility.Vector3dVector(p_pl)
        # pcd_pl.colors = o3d.utility.Vector3dVector(np.tile([0.0,1.0,0.0], (p_pl.shape[0],1)))
        # axes    = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # o3d.visualization.draw_geometries([pcd_all, pcd_pl, axes, pcd_t])

        return target_pt[0]
    elif task == 3:
        points_np = env_points.detach().cpu().numpy()

        z_min = points_np[:, 2].min()
        z_range = np.ptp(points_np[:, 2])
        scene_scale = np.max(np.ptp(points_np, axis=0))

        # Remove the platform surface and upper wall
        height_mask = (
            (points_np[:, 2] > z_min + 0.02 * z_range) &
            (points_np[:, 2] < z_min + 0.40 * z_range)
        )
        candidates = points_np[height_mask]

        print("Candidate points:", candidates.shape[0])

        if candidates.shape[0] == 0:
            target_cluster = points_np
            target_center = points_np.mean(axis=0)

        else:
            labels = DBSCAN(
                eps=0.08 * scene_scale,
                min_samples=2
            ).fit_predict(candidates)

            unique_labels, counts = np.unique(
                labels,
                return_counts=True
            )
            print(
                "DBSCAN clusters:",
                dict(zip(unique_labels, counts))
            )

            scene_center_xy = (
                points_np[:, :2].min(axis=0) +
                points_np[:, :2].max(axis=0)
            ) / 2.0

            possible_targets = []

            for label in unique_labels:
                if label == -1:
                    continue

                cluster = candidates[labels == label]

                if cluster.shape[0] < 2:
                    continue

                cluster_center = (
                    cluster.min(axis=0) +
                    cluster.max(axis=0)
                ) / 2.0

                distance = np.linalg.norm(
                    cluster_center[:2] - scene_center_xy
                )

                possible_targets.append(
                    (distance, cluster_center, cluster, label)
                )

            if possible_targets:
                _, target_center, target_cluster, selected_label = min(
                    possible_targets,
                    key=lambda item: item[0]
                )
                print("Selected cluster:", selected_label)
            else:
                print("No valid DBSCAN cluster; using candidate center.")
                target_cluster = candidates
                target_center = (
                    candidates.min(axis=0) +
                    candidates.max(axis=0)
                ) / 2.0

        print("Selected target center:", target_center)

        # Complete environment: gray
        pcd_environment = o3d.geometry.PointCloud()
        pcd_environment.points = o3d.utility.Vector3dVector(points_np)
        pcd_environment.paint_uniform_color([0.7, 0.7, 0.7])

        # Selected cluster: green
        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(target_cluster)
        pcd_target.paint_uniform_color([0.0, 1.0, 0.0])

        # Selected center: red sphere
        center_marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=0.03 * scene_scale
        )
        center_marker.translate(target_center)
        center_marker.paint_uniform_color([1.0, 0.0, 0.0])

        o3d.visualization.draw_geometries(
            [pcd_environment, pcd_target, center_marker],
            window_name="Reaching Target Detection",
            width=800,
            height=600,
            point_show_normal=False
        )

        return torch.tensor(
            target_center,
            dtype=env_points.dtype,
            device=env_points.device
        )
        

        


# ###
# The below functions are for denormalizing the waypoints: 
# ###

def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of rotation matrices (Bx3x3) to quaternions (Bx4) [w,x,y,z].
    This uses the trace method—robust for most rotations.
    """
    B = R.shape[0]
    # Compute trace
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # (B,)
    # w component
    qw = torch.sqrt(torch.clamp(t + 1.0, min=1e-6)) * 0.5
    # Avoid division by zero
    div = 4.0 * qw
    qx = (R[:, 2, 1] - R[:, 1, 2]) / div
    qy = (R[:, 0, 2] - R[:, 2, 0]) / div
    qz = (R[:, 1, 0] - R[:, 0, 1]) / div
    q = torch.stack((qw, qx, qy, qz), dim=1)
    return q

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product of two batches of quaternions q1, q2 (each B×4 [w,x,y,z]):
    returns q = q1 ⊗ q2 (B×4).
    """
    w1, x1, y1, z1 = q1.unbind(1)
    w2, x2, y2, z2 = q2.unbind(1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=1)

def denormalize_predictions(pred_pos_norm: torch.Tensor,
                            pred_quat_norm: torch.Tensor,
                            env_params_list: list,
                            tool_params_list: list) -> (torch.Tensor, torch.Tensor):
    """
    pred_pos_norm:   (B, 3, 3) normalized [grasp, init, goal] positions
    pred_quat_norm:  (B, 3, 4) normalized quaternions
    env_params_list: length-B list of dicts {center:(3,), rotation:(3,3), scale:float}
    tool_params_list: length-B list, same structure

    Returns:
      pos_world:  (B, 3, 3) de-normalized world positions
      quat_world: (B, 3, 4) de-normalized world quaternions
    """
    B, W, _ = pred_pos_norm.shape
    device = pred_pos_norm.device

    # Stack params into tensors
    centers_env = torch.stack([p["center"]   for p in env_params_list], dim=0).to(device)  # (B,3)
    rots_env    = torch.stack([p["rotation"] for p in env_params_list], dim=0).to(device)  # (B,3,3)
    scales_env  = torch.tensor([p["scale"]    for p in env_params_list],
                               device=device).view(B,1)                                 # (B,1)

    centers_tool = torch.stack([p["center"]   for p in tool_params_list], dim=0).to(device)
    rots_tool    = torch.stack([p["rotation"] for p in tool_params_list], dim=0).to(device)
    scales_tool  = torch.tensor([p["scale"]    for p in tool_params_list],
                                device=device).view(B,1)

    # Prepare output tensors
    pos_world  = torch.zeros_like(pred_pos_norm)   # (B,3,3)
    quat_world = torch.zeros_like(pred_quat_norm)  # (B,3,4)

    # Loop over the 3 waypoints
    for w in range(W):
        # Choose either tool or env params for waypoint w=0 vs w>0
        if w == 0:
            centers = centers_tool        # (B,3)
            rots    = rots_tool           # (B,3,3)
            scales  = scales_tool         # (B,1)
        else:
            centers = centers_env
            rots    = rots_env
            scales  = scales_env

        # ---- De-normalize positions ----
        p_norm = pred_pos_norm[:, w, :]              # (B,3)
        p_scaled = p_norm * scales                    # (B,3)
        # rotate back: (B,1,3) @ (B,3,3) → (B,1,3)
        p_rot = torch.bmm(p_scaled.unsqueeze(1), rots.transpose(1,2)).squeeze(1)
        pos_world[:, w, :] = p_rot + centers         # (B,3)

        # ---- De-normalize quaternions ----
        q_norm = pred_quat_norm[:, w, :]             # (B,4)
        q_R    = rotation_matrix_to_quaternion(rots) # (B,4)
        quat_world[:, w, :] = quaternion_multiply(q_R, q_norm)  # (B,4)

    return pos_world, quat_world

