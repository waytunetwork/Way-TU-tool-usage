import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

class MultiTaskWayTuLoss(nn.Module):
    def __init__(self, device):
        super(MultiTaskWayTuLoss, self).__init__()
        self.device = device

        # Fixed weights for the remaining terms:
        self.lambda_pos   = 1.0   # weight for waypoint‐position MSE
        self.lambda_quat  = 1.0   # weight for quaternion geodesic
        self.lambda_score = 1.0   # weight for score MSE
    
    def geodesic_loss(self, prediction, target, reduction="mean"):
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
    
    def forward(self, ground_truths, predictions):
        # 1) Position loss
        pred_pos = predictions["positions"]          # (B, W, 3)
        gt_pos   = ground_truths["positions"]        # (B, W, 3)
        pos_loss = F.mse_loss(pred_pos, gt_pos)

        # 2) Quaternion loss
        pred_quat = predictions["quaternions"]       # (B, W, 4)
        gt_quat   = ground_truths["quaternions"]     # (B, W, 4)
        quat_loss = self.geodesic_loss(pred_quat, gt_quat, reduction="mean")

        # 3) Score loss
        pred_score = predictions["score"].view(-1)    # (B,)
        gt_score   = ground_truths["score"].view(-1)  # (B,)
        score_loss = F.mse_loss(pred_score, gt_score)

        # 4) Weighted sum
        total_loss = (
            self.lambda_pos   * pos_loss +
            self.lambda_quat  * quat_loss +
            self.lambda_score * score_loss
        )

        loss_info = {
            "position":    pos_loss.item(),
            "quaternion":  quat_loss.item(),
            "score":       score_loss.item(),
            "weights": {
                "lambda_pos":   self.lambda_pos,
                "lambda_quat":  self.lambda_quat,
                "lambda_score": self.lambda_score
            }
        }
        return total_loss, loss_info


class WayTuUnifiedLoss(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()

        self.device = device
        self.cfg = cfg

        self.total_epochs = cfg["num-epochs"]

        self.segmentation_loss = nn.CrossEntropyLoss().to(device)

        # Hyperparameters for unified-weighted loss 
        self.segmentation_threshold = 0.3
        self.waypoint_filtering_delay = 40
        self.score_start_delay = 5
        self.filtering_threshold = 0.5
        self.filtering_delay_epochs = 40

        self.segmentation_below_threshold_epoch = None
    
    # Geodesic loss is used for quetarnians in the waypoint prediction.
    def geodesic_loss(self, prediction, target, reduction='mean'):
        dot_product = (prediction * target).sum(dim=-1).abs()
        dot_product = torch.clamp(dot_product, -1.0 + 1e-6, 1.0 - 1e-6)
        loss = 2 * torch.acos(dot_product)  # shape: (batch_size, num_waypoints)

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def forward(self, ground_truths, predictions, epoch, avg_segmentation_loss):
        # === Track segmentation loss threshold crossing ===
        segmentation_trained = (
            self.segmentation_below_threshold_epoch is None and
            avg_segmentation_loss < self.segmentation_threshold
        )
        if segmentation_trained:
            self.segmentation_below_threshold_epoch = epoch

        # === Track waypoint generation filtering === 
        filtering_active = (
            self.segmentation_below_threshold_epoch is not None and
            epoch >= self.segmentation_below_threshold_epoch + self.filtering_delay_epochs
        )

        keep_mask = ground_truths["score"] > self.filtering_threshold

        # === Segmentation Loss ===        
        pred_labels = predictions["labels"].view(-1, len(self.cfg["label-list-all"]))
        gt_labels = ground_truths["labels"].view(-1).long()
        seg_loss = self.segmentation_loss(pred_labels, gt_labels)

        #  === Waypoint Loss === 
        pos_diff = F.mse_loss(predictions["positions"], ground_truths["positions"], reduction='none')
        quat_diff = self.geodesic_loss(predictions["quaternions"], ground_truths["quaternions"], reduction='none')        
    
        if filtering_active:
            if keep_mask.sum() > 0:
                filtered_pos_loss = pos_diff[keep_mask].mean()
                filtered_quat_loss = quat_diff[keep_mask].mean()
            else: 
                filtered_pos_loss = torch.tensor(0.0, device=self.device)
                filtered_quat_loss = torch.tensor(0.0, device=self.device)
        else:
            filtered_pos_loss = pos_diff.mean()
            filtered_quat_loss = quat_diff.mean()
        
        # === Score Loss ===
        score_loss = F.mse_loss(predictions["score"].squeeze(), ground_truths["score"])

        # === Dynamics Weight for waypoint generation and score === 
        if avg_segmentation_loss > self.segmentation_threshold:
            waypoint_weight = 0
        else: 
            waypoint_weight = (self.segmentation_threshold - avg_segmentation_loss) / self.segmentation_threshold
        
        seg_weight = 1.0
        score_weight = waypoint_weight

        total_loss = (
            seg_weight * seg_loss +
            waypoint_weight * (filtered_pos_loss + filtered_quat_loss) +
            score_weight * score_loss
        )

        loss_info = {
            "segmentation": seg_loss.item(),
            "filtered_position": filtered_pos_loss.item(),
            "filtered_quaternion": filtered_quat_loss.item(),
            "score": score_loss.item(),
            "weights": {"seg": seg_weight, "wp": waypoint_weight, "score": score_weight},
            "filtering_active": filtering_active,
            "segmentation_below_threshold_epoch": self.segmentation_below_threshold_epoch,
        }

        return total_loss, loss_info


            

        
            


# This loss class is retired on 17.04.2024
class WeightedLoss(nn.Module):
    def __init__(self, cfg, device) -> None:
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.num_epochs = cfg["num-epochs"]
        self.segmentation_threshold = cfg["segmentation-threshold"]

        self.cross_entropy = nn.CrossEntropyLoss().to(device)

    def quaternion_geodesic_loss(self, prediction, target):
        dot_product = (prediction * target).sum(dim=-1).abs()
        dot_product = torch.clamp(dot_product, -1.0 + 1e-6, 1.0 - 1e-6)
        loss = 2 * torch.acos(dot_product)
        return loss.mean()

    def dynamic_weight(self, epoch):
        # Define weights for segmentation and other losses
        if epoch < self.segmentation_threshold:
            return {"segmentation": 1.0, "others": 0.0}  # Focus on segmentation
        else:
            alpha = (epoch - self.segmentation_threshold) / (self.num_epochs - self.segmentation_threshold)
            return {"segmentation": max(1 - alpha, 0.1), "others": min(alpha, 0.9)}

    def forward(self, ground_truths, predictions, epoch):
        preds = predictions["labels"]
        labels = ground_truths["labels"]
        pred_labels = preds.view(-1, len(self.cfg["label-list-all"]))
        gt_labels = labels.view(-1).long()

        segmentation_loss = self.cross_entropy(pred_labels, gt_labels)
        position_loss = F.mse_loss(ground_truths["pos"], predictions["pos"])
        quaternion_loss = self.quaternion_geodesic_loss(ground_truths["qua"], predictions["qua"])
        score_loss = F.mse_loss(ground_truths["score"].unsqueeze(1), predictions["score"])

        weights = self.dynamic_weight(epoch)

        # Combine losses based on the current weights
        total_loss = (weights["segmentation"] * segmentation_loss +
                      weights["others"] * (position_loss + quaternion_loss + 0.5 * score_loss))

        loss_info = {
            "segmentation_loss": segmentation_loss.item(),
            "position_loss": position_loss.item(),
            "quaternion_loss": quaternion_loss.item(),
            "score_loss": score_loss.item(),
            "weight": weights,
        }

        return total_loss, loss_info


class Old_WeightedLoss(nn.Module):
    def __init__(self, cfg, device) -> None:
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.num_epochs = cfg["num-epochs"]
        self.segmentation_threshold = cfg["segmentation-threshold"]

        # If works this weights can be calculated by the frequency of classes in the dataset
        self.class_weights = torch.tensor([30, 30, 60, 0, 0, 1.43, 1.43], dtype=torch.float)
        self.cross_entropy = nn.CrossEntropyLoss().to(device)
    
    def quaternion_geodesic_loss(self, prediction, target):
        dot_product = (prediction * target).sum(dim=-1).abs()
        dot_product = torch.clamp(dot_product, -1.0 + 1e-6, 1.0 - 1e-6) 
        loss = 2 * torch.acos(dot_product)
        return loss.mean()

    def dynamic_weight(self, epoch):
            return max(1 - (epoch / self.num_epochs), 0.1)

    def forward(self, ground_truths, predictions, epoch):

        # print(ground_truths["labels"].shape)
        # print(predictions["labels"].shape)

        # pred_labels = predictions["labels"].permute(0, 2, 1).contiguous().view(-1, len(self.cfg["labels-list"]))
        # gt_labels = ground_truths["labels"].view(-1).long()
        # 
        preds = predictions["labels"]
        labels = ground_truths["labels"]
        pred_labels = preds.view(-1, len(self.cfg["labels-list"])) # .to(device)
        gt_labels = labels.view(-1).long() # .to(device)
            
            # loss = cross_entropy_loss(pred_labels, gt_labels)
        #  
        # print(pred_labels.shape)
        # print(gt_labels.shape)
        # segmentation_loss = self.cross_entropy(ground_truths["labels"].to(torch.long), predictions["labels"].to(torch.float32))
        segmentation_loss = self.cross_entropy(pred_labels, gt_labels)
        position_loss = F.mse_loss(ground_truths["pos"], predictions["pos"])
        quaternion_loss = self.quaternion_geodesic_loss(ground_truths["qua"], predictions["qua"])
        score_loss = F.mse_loss(ground_truths["score"].unsqueeze(1), predictions["score"])

        loss_lambda = self.dynamic_weight(epoch)

        total_loss = (loss_lambda * segmentation_loss +
                      (1 - loss_lambda) * position_loss + 
                      (1 - loss_lambda) * quaternion_loss + 
                      (1 - loss_lambda) * 0.5 * score_loss)
        
        loss_info = {
            "segmentation_loss" : segmentation_loss.item(),
            "position_loss"     : position_loss.item(),
            "quaternion_loss"   : quaternion_loss.item(),
            "score_loss"        : score_loss.item(),
            "weight"            : loss_lambda
        }

        return total_loss, loss_info 



# class GradScaler(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, scale_factor):
#         ctx.scale_factor = scale_factor
#         return input
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         scale_factor = ctx.scale_factor
         
#         return grad_output * scale_factor, None

# class WayTu_Loss(torch.Module):
#     def __init__(self, cfg):
#         super(WayTu_Loss, self).__init__()

#         self.cfg = cfg