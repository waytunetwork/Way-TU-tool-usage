import torch
import WayTu_RAI.model_utils as mutils
import os 

from WayTu_RAI.models.WayTu_Dataset import WayTuDataset
from WayTu_RAI.models.WayTU_Model import WayTuModel, GraphPointNet, GNN_Segmentation
from WayTu_RAI.models.WayTu_Loss import WeightedLoss
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from sklearn.metrics import confusion_matrix
import numpy as np
import torch.multiprocessing as mp
import torch.nn as nn
import time

from torch.utils.tensorboard import SummaryWriter





# 18.03: The function is updated to train a good segmentation model. 
def train_segmentation_model(cfg):
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    log_file = open(os.path.join("logs", cfg["log-file"]), "w")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load Datasets
    train_data, val_data = mutils.train_validation_split(cfg)
    train_dataset, val_dataset = WayTuDataset(cfg, train_data,device), WayTuDataset(cfg, val_data, device)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch-size'], shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['batch-size'], shuffle=False, num_workers=2, pin_memory=True)

    # Initialize the model
    model = GraphPointNet(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # class_weights = torch.tensor([0.3, 0, 0, 3.0, 3.0, 3.0, 0, 0, 0], dtype=torch.float).to(device)  
    cross_entropy_loss = nn.CrossEntropyLoss().to(device)

    # Variables to save the models
    last_model_path, best_model_path = cfg["model-path"] + '-last.pth', cfg["model-path"] + '-best.pth'
    best_model = None
    best_val_loss = np.inf

    patience = 5
    early_stop_counter = 0

    # For multiprocessing 
    mp.set_start_method('spawn') 

    for e in range(cfg['num-epochs']):
        print(f"Epoch {e+1}/{cfg['num-epochs']}")

        # Training Phase 
        model.train()
        train_losses = []

        for batch in train_dataloader:
            graph_batch, labels, _, _, _, _, task, tool = batch
            graph_batch, labels = graph_batch.to(device), labels.to(device)

            preds, _ = model((graph_batch))

            # print(f"Preds shape before separation: {preds.shape}")
            batch_preds = []
            for i in range(graph_batch.num_graphs):
                mask = (graph_batch.batch == i)
                batch_preds.append(preds[mask])
            
            batch_preds = torch.stack(batch_preds, dim=0)
            # print(f"Preds shape after separation: {batch_preds.shape}")

            pred_labels = preds.view(-1, len(cfg["label-list-all"])).to(device)
            gt_labels = labels.view(-1).long().to(device)
            
            loss = cross_entropy_loss(pred_labels, gt_labels)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_losses.append(loss.item())

        train_loss_epoch = np.mean(train_losses)
        print(f"Training segmentation loss: {train_loss_epoch:.4f}")
        log_file.write(f"Epoch {e}: Training segmentation loss: {train_loss_epoch:.4f}\n")

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_dataloader:
                graph_batch, labels, _, _, _, _, task, tool = batch
                graph_batch, labels = graph_batch.to(device), labels.to(device)

                preds, _ = model((graph_batch))

                batch_preds = []
                for i in range(graph_batch.num_graphs):
                    mask = (graph_batch.batch == i)
                    batch_preds.append(preds[mask])
                
                batch_preds = torch.stack(batch_preds, dim=0)

                pred_labels = preds.view(-1, len(cfg["label-list-all"])).to(device)
                gt_labels = labels.view(-1).long().to(device)

                val_loss = cross_entropy_loss(pred_labels, gt_labels)
                val_losses.append(val_loss.item())
        
        val_loss_epoch = np.mean(val_losses)
        print(f"Validation segmentation loss: {val_loss_epoch:.4f}")
        log_file.write(f"Epoch {e}: Validation segmentation loss: {val_loss_epoch:.4f}\n")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), os.path.join("models", best_model_path))
            print(f" Best model updated (Val Loss: {best_val_loss:.4f})")
            early_stop_counter = 0
        else: 
            early_stop_counter += 1

        torch.save(model.state_dict(), os.path.join("models", last_model_path))
        if early_stop_counter >= patience:
            print(f"Early stopping triggered! No improvement for {patience} epochs.")
            break

        
    print(" Training complete. Last model saved.")
    log_file.close()

def train_waypoint_generation(cfg):
    log_file = open(os.path.join("logs", cfg["log-file"]), "w")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load Datasets
    train_data, val_data = mutils.train_validation_split(cfg)
    train_dataset, val_dataset = WayTuDataset(cfg, train_data,device), WayTuDataset(cfg, val_data, device)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch-size'], shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['batch-size'], shuffle=False, num_workers=2, pin_memory=True)
    



def train_model(cfg):
    log_file = open(os.path.join("logs", cfg["log-file"]), "w")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data, val_data = mutils.train_validation_split(cfg)

    train_dataset, val_dataset = WayTuDataset(cfg, train_data,device), WayTuDataset(cfg, val_data, device)

    train_dataloader = DataLoader(train_dataset, batch_size = cfg['batch-size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size = cfg['batch-size'], shuffle=True, num_workers=4)

    # Model, optimizer etc initializations .... 
    model = WayTuModel(cfg, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    weighted_los = WeightedLoss(cfg, device)

    best_model = None
    best_val_loss = np.inf

    mp.set_start_method('spawn') 
    for e in range(cfg['num-epochs']):
        print(f"Epoch: {e}")
        model.train()
        batch_information = {
            "segmentation_loss" : [], "segmentation_loss_val"   : [],
            "position_loss"     : [], "position_loss_val"       : [],
            "quaternion_loss"   : [], "quaternion_loss_val"     : [],
            "score_loss"        : [], "score_loss_val"          : [],
            "weight"            : [], 
        }

        for batch in train_dataloader:
            graph_batch, labels, tool_waypoint, initial_waypoint, goal_waypoint, score, task, tool = batch
            
            gt_pos = torch.stack((tool_waypoint[:,:3], initial_waypoint[:,:3], goal_waypoint[:,:3]),1)
            gt_qua = torch.stack((tool_waypoint[:,3:], initial_waypoint[:,3:], goal_waypoint[:,3:]),1)


            graph_batch.to(device)
            labels.to(device)
            tool_waypoint.to(device)
            initial_waypoint.to(device)
            goal_waypoint.to(device)
            score.to(device)
            task.to(device)
            tool.to(device)


            ground_truths = {
                    "labels"    : labels.to(device), 
                    "pos"       : gt_pos.to(device),
                    "qua"       : gt_qua.to(device),
                    "score"     : score.to(device)
                    }
            # print(graph_batch.to(device).x.device)
            predictions = model(graph_batch.to(device), {"env": task, "tool": tool})
            # print(predictions.keys()) 
            predictions['labels'].to(device)
            predictions['pos'].to(device)
            predictions['qua'].to(device)
            predictions['score'].to(device)
  
            loss, loss_info = weighted_los(ground_truths, predictions, e)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # loss_epoch += loss.item()

            for key in loss_info.keys():
                lval = loss_info[key]
                batch_information[key].append(lval)
            

            # conf_matrix = compute_confusion_matrix(preds=results["labels"], labels=torch.flatten(labels), num_classes= len(cfg['labels-list']))
            # print(conf_matrix)
        epoch_segmentation_loss = np.array(batch_information["segmentation_loss"]).sum()/len(batch_information["segmentation_loss"])
        epoch_position_loss = np.array(batch_information["position_loss"]).sum()/len(batch_information["position_loss"])
        epoch_quaternion_loss = np.array(batch_information["quaternion_loss"]).sum()/len(batch_information["quaternion_loss"])
        epoch_score_loss = np.array(batch_information["score_loss"]).sum()/len(batch_information["score_loss"])
        # epoch_weight = np.array(batch_information["weight"]).sum()/len(batch_information["weight"])
        log_file.write(f"Training segmentation loss: {epoch_segmentation_loss} ")
        log_file.write(f"position loss: {epoch_position_loss} ")
        log_file.write(f"quaternion loss: {epoch_quaternion_loss} ")
        log_file.write(f"score loss: {epoch_score_loss} ")
        # log_file.write(f"weight: {epoch_weight} \n")
        
        print(f"Training segmentation loss: {epoch_segmentation_loss} position loss: {epoch_position_loss} quaternion loss: {epoch_quaternion_loss} score loss: {epoch_score_loss}")

        model.eval()
        for batch in val_dataloader:
            graph_batch, labels, tool_waypoint, initial_waypoint, goal_waypoint, score, task, tool = batch

            gt_pos = torch.stack((tool_waypoint[:,:3], initial_waypoint[:,:3], goal_waypoint[:,:3]),1)
            gt_qua = torch.stack((tool_waypoint[:,3:], initial_waypoint[:,3:], goal_waypoint[:,3:]),1)


            ground_truths = {
                        "labels"    : labels.to(device), 
                        "pos"       : gt_pos.to(device),
                        "qua"       : gt_qua.to(device),
                        "score"     : score.to(device)
                        }

            #     predictions = model(graph_batch.to(device), {"env": task, "tool": tool})
            predictions = model(graph_batch.to(device), {"env": task, "tool": tool})
                # print(predictions.keys()) 
            predictions['labels'].to(device)
            predictions['pos'].to(device)
            predictions['qua'].to(device)
            predictions['score'].to(device)
            
            loss, loss_info = weighted_los(ground_truths, predictions, e)
            
            #     loss, loss_info = weighted_los(ground_truths, predictions, e)
            for key in loss_info.keys():
                if key == "weight":
                    continue
                val_key = key + "_val"
                lval = loss_info[key]
                batch_information[val_key].append(lval)
    
        val_segmentation_loss = np.array(batch_information["segmentation_loss_val"]).sum()/len(batch_information["segmentation_loss_val"])
        val_position_loss = np.array(batch_information["position_loss_val"]).sum()/len(batch_information["position_loss_val"])
        val_quaternion_loss = np.array(batch_information["quaternion_loss_val"]).sum()/len(batch_information["quaternion_loss_val"])
        val_score_loss = np.array(batch_information["score_loss_val"]).sum()/len(batch_information["score_loss_val"])
        log_file.write(f"Validation segmentation loss: {val_segmentation_loss} ")
        log_file.write(f"position loss: {val_position_loss} ")
        log_file.write(f"quaternion loss: {val_quaternion_loss} ")
        log_file.write(f"score loss: {val_score_loss} ")

        # val_epoch_loss = val_segmentation_loss + val_position_loss + val_quaternion_loss + val_score_loss
        print(f"Validation segmentation loss: {val_segmentation_loss} position loss: {val_position_loss} quaternion loss: {val_quaternion_loss} score loss: {val_score_loss} \n")
        

        # if val_epoch_loss <= best_val_loss:
        #     best_val_loss = val_epoch_loss
        #     best_model = model.state_dict()

    
        
        last_model_path, best_model_path = cfg["model-path"] + '-last', cfg["model-path"] + '_best'
        torch.save(model.state_dict(), os.path.join("models", last_model_path))
    # torch.save(best_model, os.path.join("models", best_model_path))
    log_file.close()
        

    

def compute_confusion_matrix(preds, labels, num_classes):
    print(preds.shape)
    print(labels.shape)
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    conf_matrix = confusion_matrix(labels, preds, labels=np.arange(num_classes))
    
    return conf_matrix

            

             



