import os
import torch
import torch.nn as nn
from torch_cluster import knn_graph
from torch_geometric.nn import MessagePassing, GCNConv, BatchNorm, Linear, global_max_pool
from torch.nn import Sequential, Linear
import torch.nn.functional as F
import numpy as np
import WayTu_RAI.model_utils as mutils
import WayTu_RAI.graph_utils as gutils

from feature_extractor import SmallPointNetEncoder



class GNN_Segmentation(torch.nn.Module):
    def __init__(self, cfg):
        super(GNN_Segmentation, self).__init__()

        self.cfg = cfg
        hidden_channels = 128

        self.conv1 = GCNConv(7, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels * 5, hidden_channels)

        self.norm1 = BatchNorm(hidden_channels)
        self.norm2 = BatchNorm(hidden_channels)
        self.norm3 = BatchNorm(hidden_channels)
        self.norm4 = BatchNorm(hidden_channels)
        self.norm5 = BatchNorm(hidden_channels)
        self.norm6 = BatchNorm(hidden_channels)
 
        self.lin = Linear(hidden_channels, 8)
 
    def forward(self, data):
        print(data)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        x1 = self.norm1(x1)
 
        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()
        x2 = self.norm2(x2)

        x3 = self.conv3(x2, edge_index)
        x3 = x3.relu()
        x3 = self.norm3(x3)
 
        x4 = self.conv4(x3, edge_index)
        x4 = x4.relu()
        x4 = self.norm4(x4)
         
        x5 = self.conv5(x4, edge_index)
        x5 = x5.relu()
        x5 = self.norm5(x5)
 
        x6 = self.conv6(torch.hstack([x1, x2, x3, x4, x5]), edge_index)
        x6 = x6.relu()
        x6 = self.norm6(x6)
        
        return self.lin(x6) 


# ---- The good segmentation module / original  ---
class GraphPointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') # We are applying max aggregation 

        self.mlp = Sequential(Linear(in_channels + 3 + 1, out_channels),
                              nn.ReLU(),
                              Linear(out_channels, out_channels),)
        # MLP takes care of transforming neighboring node features and 
        # spatial relation between source and destination nodes to a message
        # MLP is responsible for learning how to combine node features and positional information into a meaningful representation for each node.
    
    def forward(self, h, pos, edge_index): 
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]
        edge_vector = pos_j - pos_i
        edge_length = torch.norm(edge_vector, dim=1, keepdim=True)
        input = pos_j - pos_i 

        input = torch.cat([h_j, edge_vector, edge_length], dim=-1)

        return self.mlp(input)

class GraphPointNet(nn.Module):
    def __init__(self, cfg) :
        super().__init__()

        self.cfg = cfg

        self.conv1 = GraphPointNetLayer(42, 64)
        self.conv2 = GraphPointNetLayer(64, 256)
        # Note: The input dim for conv3 is increased (256 + 256)
        self.conv3 = GraphPointNetLayer(256 + 256, 512)

        self.segmentation = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.cfg['label-list-all']))
        )
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def forward(self, graph_batch):
        pos = graph_batch.x[:, :3]
        h_raw = graph_batch.x  # full node feature (pos + normals + eigs + curvature)
        pos_enc = self.pos_encoder(pos)  # [N, 32]

        h = torch.cat([h_raw, pos_enc], dim=1)  # [N, 10+32]

        batch = graph_batch.batch
        if batch is None:
            batch = h.new_zeros(h.size(0), dtype=torch.long)

        h = self.conv1(h=h, pos=pos, edge_index=graph_batch.edge_index)     
        h = F.relu(h)
        h = self.conv2(h=h, pos=pos, edge_index=graph_batch.edge_index)
        h = F.relu(h)

        # Inject global context here
        global_feat = global_max_pool(h, batch)         # [B, 256]
        global_feat_expanded = global_feat[batch]       # [N, 256]
        h = torch.cat([h, global_feat_expanded], dim=1) # [N, 512] → now used as input to conv3

        h = self.conv3(h=h, pos=pos, edge_index=graph_batch.edge_index)
        h = F.relu(h)

        global_feat = global_max_pool(h, batch)         # [B, 512]
        global_feat_expanded = global_feat[batch]       # [N, 512]
        combined = torch.cat([h, global_feat_expanded], dim=1)  # [N, 1024]

        segmentation_output = self.segmentation(combined)
        softmax_output = F.log_softmax(segmentation_output, dim=1)
        return segmentation_output, h


# Used for feature extractor head after the segmentation model. 
class PointNetMLP(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.global_fc = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

        self.output_dim = output_dim

    def forward(self, x):
        # x: [N, input_dim]
        x = self.point_mlp(x)         # [N, 128]
        x = torch.max(x, dim=0)[0]    # [128]
        x = self.global_fc(x)         # [output_dim]
        return x

# This model is a cleaner version with feature extractor. 
class MultiTaskWayTuModel(nn.Module):
    def __init__(self, cfg, device):
        super(MultiTaskWayTuModel, self).__init__()

        self.cfg = cfg

        self.gnn_seg = GraphPointNet(cfg).to(device)
        self.encoder = SmallPointNetEncoder(out_dim=cfg["feature-size"]).to(device)

        if cfg["pre-segmentation"]:
            segmentation_checkpoint = os.path.join("models", cfg["test-model-path"] + "-best.pth")
            print(f"Loading GNN segmentation model from {segmentation_checkpoint}...")
             
            state_dict = torch.load(segmentation_checkpoint, map_location=device, weights_only=True)
            self.gnn_seg.load_state_dict(state_dict)
            self.gnn_seg.eval()

            for param in self.gnn_seg.parameters():
                param.requires_grad = False

        if cfg["pre-feature"]:
            encoder_checkpoint = os.path.join("models", cfg["feature-extractor-path"])
            print(f"Loading feature extractor from {encoder_checkpoint}...")

            self.encoder.load_state_dict(torch.load(encoder_checkpoint, map_location=device)) # lo
            self.encoder.eval()

            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get the labels for platforms and tools
        environmental_mask = np.char.find(cfg["label-list-all"], "platform") != -1
        tool_mask = ~environmental_mask

        self.environmental_label_idx = np.where(environmental_mask)[0] 
        self.tool_related_label_idx = np.where(tool_mask)[0]

        self.feature_size = cfg["feature-size"]
        if cfg["normalize-new"]:
            self.feature_size = self.feature_size + 3 + 9 + 1 
            # feature_size + center of object + rotation matrix of the object + scale 

        waypoint_features_size = self.feature_size*2 + 3
        self.waypoint_generator = WaypointGeneratorHead(cfg=cfg, num_waypoints=3, feature_size= waypoint_features_size).to(device)
        self.score_prediction = ScorePredictionHead(cfg, num_waypoints=3, feature_size= waypoint_features_size).to(device)
    def forward(self, graph_batch, objs=None):
        # — common: run segmentation —
        if self.cfg["pre-segmentation"]:
            with torch.no_grad():
                preds, _ = self.gnn_seg(graph_batch)
        else:
            preds, _ = self.gnn_seg(graph_batch)

        # — training branch —
        if objs is not None:
            batch_embeddings = []
            env_params_list  = []
            tool_params_list = []

            for i in range(graph_batch.num_graphs):
                sample_mask  = (graph_batch.batch == i)
                sample_pred  = preds[sample_mask]
                sample_pos   = graph_batch.x[sample_mask][:, :3]
                predicted_label = sample_pred.argmax(dim=1)

                # —— env (you already signed off on this) ——
                if objs["env"][i] in predicted_label:
                    env_mask    = (predicted_label == objs["env"][i])
                    env_points  = sample_pos[env_mask]
                else:
                    raise RuntimeError("Couldn't find the environmental points")

                keep     = gutils.outlier_removal(env_points,
                                                radius=self.cfg["radius-outlier"],
                                                min_neighbors=self.cfg["min-neighbors"])
                env_pts  = env_points[keep]

                # Calculate side:side = sign of x̄
                env_center = env_pts.mean(dim=0)   # (x̄, ȳ, z̄)
                side = -1 if env_center[0] < 0.0 else 1

                env_norm, env_params = gutils.normalize_point_cloud(env_pts)
                env_pc   = gutils.arrange_point_cloud_size(env_norm,
                                                        self.cfg["feature-size"])
                self.encoder.eval()
                with torch.no_grad():
                    env_feat = self.encoder(env_pc.unsqueeze(0)).squeeze(0)

                # —— tool (single known tool in training) ——
                if objs["tool"][i] in predicted_label:
                    tool_mask   = (predicted_label == objs["tool"][i])
                    tool_points = sample_pos[tool_mask]
                else:
                    raise RuntimeError("Couldn't find the tool points.")

                keep       = gutils.outlier_removal(tool_points,
                                                    radius=self.cfg["radius-outlier"],
                                                    min_neighbors=self.cfg["min-neighbors"])
                tool_pts   = tool_points[keep]
                tool_norm, tool_params = gutils.normalize_point_cloud(tool_pts)
                tool_pc    = gutils.arrange_point_cloud_size(tool_norm,
                                                            self.cfg["feature-size"])
                with torch.no_grad():
                    tool_feat = self.encoder(tool_pc.unsqueeze(0)).squeeze(0)

                target_center = gutils.get_target_center(env_pts, objs["env"][i],side)

                # collect params for denorm
                env_params_list.append(env_params)
                tool_params_list.append(tool_params)

                
                combined_feature = torch.cat([
                    env_feat,                       # (128,)
                    tool_feat,                      # (128,)
                    env_params["center"],               # (3,)
                    env_params["rotation"].reshape(9),     # (9,)
                    torch.tensor([env_params["scale"]], device=env_points.device),        # (1,)
                    tool_params["center"],              # (3,)
                    tool_params["rotation"].reshape(9),    # (9,)
                    torch.tensor([tool_params["scale"]], device=tool_points.device),       # (1,)
                    target_center             # (3,)
                ], dim=0)
                batch_embeddings.append(combined_feature)

            batch_embeddings = torch.stack(batch_embeddings, dim=0)  # (B, 2D+29)
            env_positions, env_quaternion = self.waypoint_generator(batch_embeddings)
            score = self.score_prediction(batch_embeddings, env_positions, env_quaternion)

            pos_world, quat_world = gutils.denormalize_predictions(
                env_positions, env_quaternion,
                env_params_list, tool_params_list
            )

            return {
                "labels":    batch_embeddings,
                "positions": pos_world,
                "quaternions": quat_world,
                "score":     score
            }

        # — testing branch —
        else:
            B = graph_batch.num_graphs
            env_params_list, tool_params_list = [], []
            pred_pos_list, pred_quat_list, pred_score_list = [], [], []
            selected_tool_indices = []

            for i in range(B):
                # slice out this sample
                mask          = (graph_batch.batch == i)
                sample_pred   = preds[mask]
                sample_pos    = graph_batch.x[mask][:, :3]
                predicted_lbl = sample_pred.argmax(dim=1)

                # — env extraction (approved) —
                env_idx  = torch.tensor(self.environmental_label_idx, 
                                        device=sample_pos.device)
                env_mask = torch.isin(predicted_lbl, env_idx)
                raw_env  = sample_pos[env_mask] if env_mask.any() else sample_pos

                keep     = gutils.outlier_removal(raw_env,
                                                radius=self.cfg["radius-outlier"],
                                                min_neighbors=self.cfg["min-neighbors"])
                clean_env = raw_env[keep]
                
                # Calculate side:side = sign of x̄
                env_center = clean_env.mean(dim=0)   # (x̄, ȳ, z̄)
                side = -1 if env_center[0] < 0.0 else 1

                norm_env, env_params = gutils.normalize_point_cloud(clean_env)
                env_pc   = gutils.arrange_point_cloud_size(norm_env,
                                                        self.cfg["feature-size"])
                self.encoder.eval()
                with torch.no_grad():
                    env_feat = self.encoder(env_pc.unsqueeze(0)).squeeze(0)

                # pick one env label for target‐center fallback
                env_labels = torch.unique(predicted_lbl[env_mask]) 
                env_label  = int(env_labels[0]) if env_labels.numel()>0 else None
                target_center = gutils.get_target_center(clean_env, env_label, side)

                env_params_list.append(env_params)

                # — tool loop & scoring —
                tool_idx  = torch.tensor(self.tool_related_label_idx,
                                        device=sample_pos.device)
                tool_mask = torch.isin(predicted_lbl, tool_idx)

                # no tool → zeros
                if not tool_mask.any():
                    pred_pos_list.append(torch.zeros(3, device=sample_pos.device))
                    pred_quat_list.append(torch.zeros(4, device=sample_pos.device))
                    pred_score_list.append(torch.tensor(0., device=sample_pos.device))
                    selected_tool_indices.append(None)
                    tool_params_list.append({
                        "center":   torch.zeros(3, device=sample_pos.device),
                        "rotation": torch.eye(3, device=sample_pos.device),
                        "scale":    1.0
                    })
                    continue

                # pick top-3 most frequent
                tl = predicted_lbl[tool_mask]
                unique_tools, counts = torch.unique(tl, return_counts=True)
                top3 = unique_tools[torch.argsort(counts, descending=True)][:3]

                best_score = None
                for t in top3:
                    t = int(t)
                    raw_t = sample_pos[predicted_lbl == t]

                    keep_t = gutils.outlier_removal(raw_t,
                                                    radius=self.cfg["radius-outlier"],
                                                    min_neighbors=self.cfg["min-neighbors"])
                    clean_t = raw_t[keep_t]
                    norm_t, t_params = gutils.normalize_point_cloud(clean_t)
                    pc_t = gutils.arrange_point_cloud_size(norm_t,
                                                        self.cfg["feature-size"])
                    with torch.no_grad():
                        tool_feat = self.encoder(pc_t.unsqueeze(0)).squeeze(0)

                    combined = torch.cat([
                        env_feat,
                        tool_feat,
                        env_params["center"],
                        env_params["rotation"].reshape(9),
                        torch.tensor([env_params["scale"]], device=pc_t.device),
                        t_params["center"],
                        t_params["rotation"].reshape(9),
                        torch.tensor([t_params["scale"]], device=pc_t.device),
                        target_center
                    ], dim=0).unsqueeze(0)

                    pos_pred, quat_pred = self.waypoint_generator(combined)
                    score_pred = self.score_prediction(combined, pos_pred, quat_pred)

                    if best_score is None or score_pred.item() > best_score.item():
                        best_score       = score_pred
                        best_pos         = pos_pred.squeeze(0)
                        best_quat        = quat_pred.squeeze(0)
                        best_tool_label  = t
                        best_tool_params = t_params

                pred_pos_list.append(best_pos)
                pred_quat_list.append(best_quat)
                pred_score_list.append(best_score)
                selected_tool_indices.append(best_tool_label)
                tool_params_list.append(best_tool_params)

            # denormalize all at once
            pos_world, quat_world = gutils.denormalize_predictions(
                torch.stack(pred_pos_list, dim=0),
                torch.stack(pred_quat_list, dim=0),
                env_params_list,
                tool_params_list
            )
            scores = torch.stack(pred_score_list, dim=0)

            return {
                "labels":          preds,  
                "positions":       pos_world,
                "quaternions":     quat_world,
                "score":           scores,
                "selected_tool":   selected_tool_indices
            }

class WayTuModel(nn.Module):
    def __init__(self, cfg, device, pre_trained_segmentation = False) :
        super().__init__()

        self.cfg = cfg
        self.pre_trained_segmentation = pre_trained_segmentation
        self.feature_size = 1024

        # Find the indices of the labels: 
        environmental_mask = np.char.find(cfg["label-list-all"], "platform") != -1
        tool_mask = ~environmental_mask

        self.environmental_label_idx = np.where(environmental_mask)[0] 
        self.tool_related_label_idx = np.where(tool_mask)[0]

        self.gnn_seg = GraphPointNet(cfg).to(device)
        if pre_trained_segmentation: 
            segmentation_checkpoint = os.path.join("models", cfg["test-model-path"] + "-best.pth")
            print(f"Loading segmentation model from {segmentation_checkpoint}...")
            state_dict = torch.load(segmentation_checkpoint, map_location=device, weights_only=True)
            self.gnn_seg.load_state_dict(state_dict)
            self.gnn_seg.eval() # Set to evaluation mode
            for param in self.gnn_seg.parameters():
                param.requires_grad = False  # Freeze the model
        
        self.pointnet_feature_extractor = PointNetMLP(input_dim=3)
        self.feature_size = 2 * self.pointnet_feature_extractor.output_dim

        # Create waypoint generator head ans the score prediction head. 
        self.waypoint_generator = WaypointGeneratorHead(cfg=cfg, num_waypoints=3, feature_size=self.feature_size).to(device)
        self.score_prediction = ScorePredictionHead(cfg, num_waypoints=3, feature_size= self.feature_size).to(device)

    def forward(self, graph_batch, objs=None):
        if self.pre_trained_segmentation:
            with torch.no_grad():
                preds, h = self.gnn_seg(graph_batch)
        else:
            preds, h = self.gnn_seg(graph_batch)

        batch_preds, batch_h, batch_x = [], [], []
        for i in range(graph_batch.num_graphs):
            mask = (graph_batch.batch == i)
            batch_preds.append(preds[mask])
            batch_h.append(h[mask])
            batch_x.append(graph_batch.x[mask]) # --> for getting initial features of nodes

        batch_preds = torch.stack(batch_preds, dim=0)
        batch_h = torch.stack(batch_h, dim=0)

        env_batch, tool_batch = [], [] # , graph_feature_batch, []
        tool_label_list = []

        for i in range(graph_batch.num_graphs):
            predicted_label = batch_preds[i].argmax(dim=1)
            unique_predicted_labels = torch.unique(predicted_label)
            
            # In training: 
            if objs is not None:
                x_i = batch_x[i]

                if objs["env"][i] in unique_predicted_labels:
                    mask = (predicted_label == objs["env"][i])
                    # environment_features = torch.max(batch_h[i][mask], dim=0)[0]
                    env_points = x_i[mask]
                else:
                    # environment_features = torch.max(batch_h[i], dim=0)[0]
                    env_points = x_i

                environment_features = self.pointnet_feature_extractor(env_points[:, :3])
                env_batch.append(environment_features)

                if objs["tool"][i] in unique_predicted_labels:
                    mask = (predicted_label == objs["tool"][i])
                    # tool_features = torch.max(batch_h[i][mask], dim=0)[0]
                    tool_points = x_i[mask]
                else:
                    # tool_features = torch.max(batch_h[i], dim=0)[0]
                    tool_points = x_i
                
                tool_features = self.pointnet_feature_extractor(tool_points[:, :3])
                # graph_feature_batch.append(torch.max(batch_h[i], dim=0)[0])
                tool_batch.append(tool_features)

            else:
                # mask = torch.isin(predicted_label, torch.tensor(self.environmental_label_idx, device=predicted_label.device))
                x_i = batch_x[i]
                mask = torch.isin(predicted_label, torch.tensor(self.environmental_label_idx, device=predicted_label.device))
                if mask.any():
                    # print(f"#DEBUG# task: {self.cfg['task']} YES")
                    # environment_features = torch.max(batch_h[i][mask], dim=0)[0]
                    # environment_features = torch.mean(batch_h[i][mask], dim=0)
                    env_points = x_i[mask]
                    environment_features = self.pointnet_feature_extractor(env_points[:, :3])  
                else:
                    # print(f"#DEBUG# task: {self.cfg['task']} NO")
                    # environment_features = torch.zeros_like(batch_h[i][0])
                    environment_features = torch.zeros(self.pointnet_feature_extractor.output_dim, device=x_i.device)

                env_batch.append(environment_features)

                tool_mask = torch.isin(predicted_label, torch.tensor(self.tool_related_label_idx, device=predicted_label.device))

                if not tool_mask.any():
                    tool_batch.append([torch.zeros(self.pointnet_feature_extractor.output_dim, device=batch_x.device)] * 3)
                    tool_label_list.append([0, 0, 0])
                    continue

                tool_labels = predicted_label[tool_mask]
                unique_tools, tool_counts = torch.unique(tool_labels, return_counts=True)
                sorted_indices = torch.argsort(tool_counts, descending=True)
                top_tools = unique_tools[sorted_indices[:3]]

                all_tools_batch = []
                all_tools_labels = []

                x_i = batch_x[i]

                for tool in top_tools:
                    one_tool_mask = (predicted_label == tool)
                    tool_points = x_i[one_tool_mask]

                    if tool_points.shape[0] > 0:
                        tool_feat = self.pointnet_feature_extractor(tool_points[:, :3])
                    else:
                        tool_feat = torch.zeros(self.pointnet_feature_extractor.output_dim, device=x_i.device)

                    all_tools_batch.append(tool_feat)
                    all_tools_labels.append(tool.item())

                # Pad to 3 tools if fewer detected
                while len(all_tools_batch) < 3:
                    all_tools_batch.append(torch.zeros(self.pointnet_feature_extractor.output_dim, device=x_i.device))
                    all_tools_labels.append(0)

                tool_batch.append(all_tools_batch)
                tool_label_list.append(all_tools_labels)

        # ...

        if objs is not None:
            # for e in env_batch:
            #     print(f"###DEBUG### env_batch: {e.size()}")
            env_batch = torch.stack(env_batch, dim=0) # env_batch is a list
            tool_batch = torch.stack(tool_batch, dim=0)
            combined_batch = torch.cat((env_batch, tool_batch), dim=1)
            env_positions, env_quaternion = self.waypoint_generator(combined_batch)
            score = self.score_prediction(combined_batch, env_positions, env_quaternion)

            return {
                "labels": batch_preds,
                "positions": env_positions,
                "quaternions": env_quaternion,
                "score": score
            }

        else:
            env_batch = torch.stack(env_batch, dim=0)
            batch_size = len(tool_batch)
            pred_pos_list, pred_quat_list, pred_score_list, selected_tool_indices = [], [], [], []
            
            # === Debug for understanding whether the platform features are distinct ===
            print(f"#DEBUG# environment batch size: {env_batch.size()}")
            env_feature_np = env_batch.squeeze().detach().cpu().numpy()
            task_name = self.cfg["task"]
            sample = 1
            np.save(f"debug_env_feature_{task_name}_{sample}.npy", env_feature_np)
            print(f"Saved: debug_env_feature_{task_name}_{sample}.npy")
            # ===

            for b in range(batch_size):
                env_feat = env_batch[b].unsqueeze(0)
                best_score, best_pos, best_quat, best_tool_idx = None, None, None, -1

                for idx, tool_feat in enumerate(tool_batch[b]):
                    tool_feat = tool_feat.unsqueeze(0)
                    x = torch.cat([env_feat, tool_feat], dim=1)

                    tool_feat_np = tool_feat.squeeze().detach().cpu().numpy()
                    np.save(f"debug_tool_feature_{idx}_{sample}.npy", tool_feat_np)
                    print(f"Saved: debug_tool_feature_{idx}_{sample}.npy")

                    pos, quat = self.waypoint_generator(x)
                    score = self.score_prediction(x, pos, quat)

                    if best_score is None or score.item() > best_score.item():
                        best_score, best_pos, best_quat, best_tool_idx = score, pos, quat, idx

                pred_score_list.append(best_score)
                pred_pos_list.append(best_pos)
                pred_quat_list.append(best_quat)
                selected_tool_indices.append(tool_label_list[b][best_tool_idx])

            env_positions = torch.cat(pred_pos_list, dim=0)
            env_quaternion = torch.cat(pred_quat_list, dim=0)
            score = torch.cat(pred_score_list, dim=0)

            return {
                "labels": batch_preds,
                "positions": env_positions,
                "quaternions": env_quaternion,
                "score": score,
                "selected_tool": selected_tool_indices
            }
        
        # return batch_preds, env_positions, env_quaternion, score

class Waypoint_Generator(nn.Module):
    def __init__(self, cfg, device):
        # super()
        self.cfg = cfg
        self.segmentation_module = GraphPointNet()
        self.segmentation_module.load_state_dict(torch.load(cfg["best-segmentation-model"], map_location=device, weights_only=True))

    def forward(self, graph_batch, infos = None):
        preds, h = self.gnn_seg((graph_batch))
        


class WaypointGeneratorHead(nn.Module):
    def __init__(self, cfg, num_waypoints, feature_size = 256):
        super().__init__()
        self.cfg = cfg

        self.num_waypoints = num_waypoints 

        self.fc5 = mutils.xavier_initialization(feature_size, 1024)
        self.fc6 = mutils.xavier_initialization(1024, 512)
        self.fc7 = mutils.xavier_initialization(512, 128)
        self.fc8 = mutils.xavier_initialization(128, num_waypoints*7)

        self.norm4 = nn.LayerNorm(feature_size)
        self.norm5 = nn.LayerNorm(1024)
        self.norm6 = nn.LayerNorm(512)
        self.norm7 = nn.LayerNorm(128)
    
    def forward(self, x):
        x = self.norm4(x)
        x = F.relu(self.fc5(x))
        x = self.norm5(x)
        x = F.relu(self.fc6(x))
        x = self.norm6(x)
        x = F.relu(self.fc7(x))
        x = self.norm7(x)
        x = self.fc8(x)

        x = x.view(-1, self.num_waypoints, 7) 
        positions = x[:, :, :3]
        quaternions = x[:, :, 3:]

        normalized_quaternions = F.normalize(quaternions, p=2, dim=-1)
        return positions, normalized_quaternions

# Again a direct replica of previous model. This time it gets full feature map of the point cloud
class ScorePredictionHead(nn.Module):
    def __init__(self, cfg, num_waypoints, feature_size = 256):
        super().__init__()

        self.fc5 = mutils.xavier_initialization(feature_size + 7*num_waypoints, 128)
        self.fc6 = mutils.xavier_initialization(128, 1)

        self.bn4 = nn.BatchNorm1d(feature_size+ 7*num_waypoints)
        self.bn5 = nn.BatchNorm1d(128)
    
    def forward(self, x, waypoint_positions, waypoint_quaternions):
        waypoints = torch.cat((waypoint_positions, waypoint_quaternions), dim=-1).view(waypoint_positions.size(0), -1)
        x = torch.cat((x, waypoints), 1)

        x = x.to(torch.float32)
        x.cuda()

        x = self.bn4(x)
        x = F.gelu(self.fc5(x))
        x=self.bn5(x)
        x = self.fc6(x)

        return x
    

class WayTuModel_old(nn.Module):
    def __init__(self, cfg, device) :
        super().__init__()

        self.cfg = cfg

        self.indices = {
            "lifting-platform": 4
        }

        # Find the indices of the labels: 
        environmental_mask = np.char.find(cfg["label-list-all"], "platform") != -1
        tool_mask = ~environmental_mask

        self.environmental_label_idx = np.where(environmental_mask)[0] 
        self.tool_related_label_idx = np.where(tool_mask)[0]



        self.gnn_seg = GraphPointNet(cfg).to(device)
        self.environment_waypoint_generator = WaypointGeneratorHead(cfg, num_waypoints=2).to(device) # initial and goal waypoints
        self.grasp_waypoint_generator = WaypointGeneratorHead(cfg, num_waypoints=1).to(device) # tool waypoint
        # self.score_prediction = ScorePredictionHead(cfg, num_waypoints=3).to(device)

        self.waypoint_generator = WaypointGeneratorHead(cfg=cfg, num_waypoints=3, feature_size=512).to(device)
        self.score_prediction = ScorePredictionHead(cfg, num_waypoints=3, feature_size= 512).to(device)

    def forward(self, graph_batch, objs= None):
        # Get label predictions and node features from graph point net 
        preds, h = self.gnn_seg((graph_batch))
        
        # Split predictions by their batches 
        batch_preds, batch_h = [], []
        for i in range(graph_batch.num_graphs):
            mask = (graph_batch.batch == i)
            batch_preds.append(preds[mask])
            batch_h.append(h[mask])
        
        batch_preds = torch.stack(batch_preds, dim=0)
        batch_h = torch.stack(batch_h, dim=0)


        env_batch, tool_batch, graph_feature_batch = [], [], []
        for i in range(graph_batch.num_graphs):
            # Get predicted labels for the batch 
            predicted_label = batch_preds[i].argmax(dim=1)
            # Get which labels were predicted 
            unique_predicted_labels = torch.unique(predicted_label[i])

           
            if objs is not None: 
                if objs["env"][i] in unique_predicted_labels:
                    mask = (predicted_label == objs["env"][i])
                    environment_features = torch.max(batch_h[i][mask], dim=0)[0]
                else: 
                    environment_features = torch.max(batch_h[i], dim=0)[0]
                # print("unique_predicted labels: ", unique_predicted_labels)
                env_batch.append(environment_features)

                if objs["tool"][i] in unique_predicted_labels:
                    mask = (predicted_label == objs["tool"][i])
                    # print(mask.shape) # without softmax: 3072, 7
                    # print(batch_h[i].shape) # without softmax: 3072, 256
                    tool_features = torch.max(batch_h[i][mask], dim=0)[0]
                    # print("if: ", tool_features.shape)
                else: 
                    tool_features = torch.max(batch_h[i], dim=0)[0]
                    # print("el: ", tool_features.shape)
                
                tool_batch.append(tool_features)

                graph_feature_batch.append(torch.max(batch_h[i], dim=0)[0])


            
            else:
                ... 
               
                mask = np.isin(predicted_label, self.environmental_label_idx)
                environment_features = torch.max(batch_h[i][mask], dim=0)[0]
                env_batch.append(environment_features)

                tool_mask = np.isin(predicted_label, self.tool_related_label_idx)

                tool_labels = torch.max(batch_preds[i][tool_mask], dim=0)[0] # Can be problematic!
                
                unique_tools = np.unique(tool_labels)
                tool_counts = {tool.item(): (predicted_label == tool).sum().item() for tool in unique_tools}
                sorted_tools = sorted(tool_counts, key=tool_counts.get, reverse=True)
                top_tools = sorted_tools[:3]

                all_tools_batch = []
                for tool in top_tools:
                    one_tool_mask = (predicted_label == tool)
                    if one_tool_mask.any():
                        tool_features = torch.max(batch_h[i][one_tool_mask], dim=0)[0]
                    else:
                        tool_features = torch.zeros_like(batch_h[i][0])
                    all_tools_batch.append(tool_features)

                tool_batch.append(all_tools_batch)

        if objs is not None: 
            env_batch = torch.stack(env_batch, dim=0)
            tool_batch = torch.stack(tool_batch, dim = 0)
            combined_batch = torch.cat((env_batch, tool_batch), dim=1)

            # Get one prediction for selected tool and environment features
            # print(combined_batch.shape)
            env_positions, env_quaternion = self.waypoint_generator(combined_batch)
            score = self.score_prediction(combined_batch, env_positions, env_quaternion)
        else: 
            env_batch = torch.stack(env_batch, dim=0)
            possible_positions, possible_quaternions, score_list = [], [], []
            for i in range(tool_batch.shape[1]):
                tool_batch = torch.stack(tool_batch[0][i], dim = 0)
                combined_batch = torch.cat((env_batch, tool_batch), dim=1)

                position_predictions, quaternion_predictions = self.waypoint_generator(combined_batch)
                score_predictions = self.score_prediction(combined_batch, position_predictions, quaternion_predictions)
                possible_positions.append(position_predictions)
                possible_quaternions.append(quaternion_predictions)
                score_list.append(score_predictions)
            
            highest_index = score_list.index(max(score_list))
            env_positions = possible_positions[highest_index]
            env_quaternion = possible_quaternions[highest_index]
            score = score_list[highest_index]   

        
        results = {
                "labels"    : batch_preds, 
                "pos"       : env_positions,
                "qua"       : env_quaternion,
                "score"     : score,
            }
        return results 
