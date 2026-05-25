import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

from torch.utils.data import random_split, DataLoader
from dataset import ToolBotDataset
from model import ToolNet


ROOT_DIR        = "toolbot-minigolf" 
SAVE_DIR        = ...
NUM_EPOCHS      = 50
BATCH_SIZE      = 8
BASE_LR         = 1e-4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

ALPHA           = 0.5
BETA            = 0.5


def heatmap_loss(pred: torch.Tensor, 
                 gt: torch.Tensor,
                 pos_weight:float = 0.5):
    
    # Elementwise Huber loss
    loss_map = F.smooth_l1_loss(pred, gt, reduction='none')
    pos_mask = (gt > 0.5)
    neg_mask = ~pos_mask

    # Average over positives (if any)
    if pos_mask.any():
        pos_loss = loss_map[pos_mask].mean()
    else:
        pos_loss = torch.tensor(0.0, device=loss_map.device)
    # Average over negatives (if any)
    if neg_mask.any():
        neg_loss = loss_map[neg_mask].mean()
    else:
        neg_loss = torch.tensor(0.0, device=loss_map.device)

    return pos_weight * pos_loss + (1 - pos_weight) * neg_loss


def train_ToolNet():
    print("FFF")
    dataset = ToolBotDataset(ROOT_DIR)
    n_val   = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print("EEE")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    print("DDD")
    # 2) Model, optimizer
    model     = ToolNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    print("CCC")
    # warm‑up for first 10% of epochs
    warmup_epochs = max(1, int(0.1 * NUM_EPOCHS))
    
    print("BBB")
    for epoch in range(1, NUM_EPOCHS + 1):
        print("AAA")
        # 3) adjust LR
        lr = BASE_LR * min(epoch, warmup_epochs) / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # 4) training pass
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            rgb   = batch['rgb'].to(DEVICE)
            depth = batch['depth'].to(DEVICE)
            gt_g  = batch['heatmap_grasp'].to(DEVICE)
            gt_m  = batch['heatmap_manip'].to(DEVICE)

            optimizer.zero_grad()
            pred_g, pred_m = model(rgb, depth)

            loss_g = heatmap_loss(pred_g, gt_g, BETA)
            loss_m = heatmap_loss(pred_m, gt_m, BETA)
            loss   = loss_g + ALPHA * loss_m

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * rgb.size(0)

        train_loss /= n_train

        # 5) validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                rgb   = batch['rgb'].to(DEVICE)
                depth = batch['depth'].to(DEVICE)
                gt_g  = batch['heatmap_grasp'].to(DEVICE)
                gt_m  = batch['heatmap_manip'].to(DEVICE)

                pred_g, pred_m = model(rgb, depth)

                loss_g = heatmap_loss(pred_g, gt_g, BETA)
                loss_m = heatmap_loss(pred_m, gt_m, BETA)
                loss   = loss_g + ALPHA * loss_m

                val_loss += loss.item() * rgb.size(0)

        val_loss /= n_val

        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
              f"lr={lr:.2e} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

    return model

if __name__ == '__main__':
    train_ToolNet()
