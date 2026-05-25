import os
from tqdm import tqdm
import torch 
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split

from Baselines.KETO.model import KetoAdaptDecoder, KetoAdaptDiscriminator, KetoAdaptEncoder
from Baselines.KETO.dataset import WayTuUnifiedDataset


# df = pd.read_csv(os.path.join(cfg["dataset_dir"], "dataset_info.csv"))
# dataset = WayTuUnifiedDataset(cfg, df, device)

# n_val = int(len(dataset) * cfg.get("val_split", 0.2))
# train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

# train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
# val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)


class Keto_to_WayTu: 
    def __init__(self, keto_cfg):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.batch_size = keto_cfg.get('batch-size',64)
        
        self.task = keto_cfg['task']
        # self.num_points = cfg['num-points']

        self.num_funt_vect = keto_cfg['num-funct-vect']
        self.dataset_path = keto_cfg.get('dataset-path', None)

        self.cfg = keto_cfg

        self.dist_thres=0.4

        self.cfg = keto_cfg
        self.num_epoch = keto_cfg.get('num-epoch', 20)
        self.optimizer_type = keto_cfg.get('optimizer-type','Adam')

        self.encoder_path = self.cfg.get('encoder-model',  'keto_encoder.pth')
        self.decoder_path = self.cfg.get('decoder-model',  'keto_decoder.pth')
        self.disc_path    = self.cfg.get('disc-model',     'keto_discriminator.pth')
        self.best_ckpt_dir= "models"
        os.makedirs(self.best_ckpt_dir, exist_ok=True)

    def train_keto_vae(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- dataset & loaders ---
        df = pd.read_csv(os.path.join(self.cfg["dataset-path"], "dataset_info.csv"))
        dataset = WayTuUnifiedDataset(self.cfg, df, device)

        val_split = float(self.cfg.get("val_split", 0.2))
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # --- models ---
        encoder = KetoAdaptEncoder().to(device)
        decoder = KetoAdaptDecoder().to(device)
        discriminator = KetoAdaptDiscriminator().to(device)

        params = list(encoder.parameters()) + list(decoder.parameters()) + list(discriminator.parameters())
        if self.optimizer_type == 'Adam':
            optimizer = optim.Adam(params, lr=float(self.cfg.get("lr", 1e-3)))
        elif self.optimizer_type == 'SGDM':
            optimizer = optim.SGD(params, lr=float(self.cfg.get("lr", 1e-3)), momentum=0.9)
        else:
            raise NotImplementedError(f"optimizer {self.optimizer_type} not supported")

        best_val = np.inf

        for epoch in range(1, self.num_epoch + 1):
            train_loss, train_acc = self._run_epoch(
                loader=train_loader, device=device,
                encoder=encoder, decoder=decoder, discriminator=discriminator,
                optimizer=optimizer, train=True
            )
            with torch.no_grad():
                val_loss, val_acc = self._run_epoch(
                    loader=val_loader, device=device,
                    encoder=encoder, decoder=decoder, discriminator=discriminator,
                    optimizer=None, train=False
                )

            print(f"[Epoch {epoch:03d}] "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

            # save best by val_loss
            if val_loss < best_val:
                best_val = val_loss
                torch.save(encoder.state_dict(), os.path.join(self.best_ckpt_dir, 'best_encoder.pth'))
                torch.save(decoder.state_dict(), os.path.join(self.best_ckpt_dir, 'best_decoder.pth'))
                torch.save(discriminator.state_dict(), os.path.join(self.best_ckpt_dir, 'best_discriminator.pth'))

        # final save (last epoch)
        torch.save(encoder.state_dict(), self.encoder_path)
        torch.save(decoder.state_dict(), self.decoder_path)
        torch.save(discriminator.state_dict(), self.disc_path)
    
    @staticmethod
    def _run_epoch(loader, device, encoder, decoder, discriminator, optimizer, train: bool):
        """
        One full pass over a loader. If train=True, backprop + step.
        Expects batches shaped like your old KETO dataset:
          point_cloud: (B, N, 3)
          waypoint:    (B, 21) with triplets [pos(3), quat(4)] * 3
          score, task_id (unused here)
        """
        if train:
            encoder.train(); decoder.train(); discriminator.train()
        else:
            encoder.eval();  decoder.eval();  discriminator.eval()

        total_loss = 0.0
        total_acc  = 0.0
        total_n    = 0

        for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):
            # unpack depending on your dataset return signature
            # Old KETO-style: (point_cloud, waypoint, score, task_id, obj_pos?) 
            # if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            #     point_cloud = batch[0].to(device).float()     # (B, N, 3)
            #     waypoint    = batch[1].to(device).float()     # (B, 21) positions+quats
            # elif isinstance(batch, dict):
            #     # If you routed new dict-style dataset here, adapt as needed:
            #     # we need point_cloud (tool-only) and a 21-d waypoint vector.
            #     raise ValueError("This training loop expects the old tuple dataset (pc, waypoint, ...).")
            # else:
            #     raise ValueError("Unexpected batch format.")

            # # slice waypoints: [0:3]=grasp pos, [7:10]=initial pos, [14:17]=goal pos
            # grasp_point   = waypoint[:, 0:3]
            # initial_point = waypoint[:, 7:10]
            # goal_point    = waypoint[:, 14:17]
            tool_pts = batch["tool_points"].to(device).float()   # (B, Nt, 3)
            env_pts  = batch["env_points"].to(device).float()    # (B, Ne, 3)
            # print(tool_pts.shape)
            # print(env_pts.shape)
            # concat along the point dimension → (B, Nt+Ne, 3)
            point_cloud = torch.cat([tool_pts, env_pts], dim=1)
            # print(point_cloud.shape)

            # Compute center and scale per sample
            center = point_cloud.mean(dim=1, keepdim=True)        # (B, 1, 3)
            pc_centered = point_cloud - center                    # (B, Nt+Ne, 3)
            scale = pc_centered.norm(dim=2).max(dim=1, keepdim=True)[0]  # (B, 1)

            # Normalize point cloud
            point_cloud = pc_centered / scale.unsqueeze(-1)       # (B, Nt+Ne, 3)
            # print(point_cloud.shape)
            # waypoints: take positions only (B,3) for each keypoint
            grasp_point   = batch["tool_waypoint"][:,   0:3].to(device).float()   # (B,3)
            initial_point = batch["initial_waypoint"][:,0:3].to(device).float()   # (B,3)
            goal_point    = batch["goal_waypoint"][:,   0:3].to(device).float()   # (B,3)

            # print("grasp_point: ", grasp_point.shape)
            # print(center.shape)
            # print(scale.shape)
            grasp_point   = ((grasp_point   - center.squeeze(1)) / scale).unsqueeze(1)
            initial_point = ((initial_point - center.squeeze(1)) / scale).unsqueeze(1)
            goal_point    = ((goal_point    - center.squeeze(1)) / scale).unsqueeze(1)



            keypoints = [grasp_point, initial_point, goal_point]
            keypoints_labels = torch.zeros((grasp_point.shape[0], 1), device=device)  # label=0 as in your code

            if train:
                optimizer.zero_grad()
            
            # --- forward VAE-style ---
            latent_var = encoder(point_cloud, keypoints)     # z = [mu, sigma]
            z_mu, z_sigma = torch.chunk(latent_var, 2, dim=1)
            z = z_mu + z_sigma * torch.randn_like(z_sigma)

            pred_grasp, pred_init, pred_goal = decoder(point_cloud, latent_var)

            # recon losses
            loss_vae_grasp   = F.l1_loss(pred_grasp, grasp_point.squeeze(1))
            loss_vae_initial = F.l1_loss(pred_init,  initial_point.squeeze(1))
            loss_vae_goal    = F.l1_loss(pred_goal,  goal_point.squeeze(1))

            # KL-like term (your original)
            loss_vae_mmd = torch.mean(z_mu**2 + z_sigma**2 - torch.log(1e-8 + z_sigma**2) - 1)

            loss_vae = loss_vae_grasp + loss_vae_initial + loss_vae_goal + loss_vae_mmd

            # discriminator
            discr_logit = discriminator(point_cloud, keypoints)
            loss_discr  = F.binary_cross_entropy_with_logits(discr_logit, keypoints_labels)

            total = loss_vae + loss_discr

            if train:
                total.backward()
                optimizer.step()

            # metrics
            with torch.no_grad():
                preds = (torch.sigmoid(discr_logit) < 0.5).float()
                acc   = (preds == keypoints_labels).float().mean()

            bs = point_cloud.size(0)
            total_loss += float(total.item()) * bs
            total_acc  += float(acc.item()) * bs
            total_n    += bs

        # average over samples
        avg_loss = total_loss / max(1, total_n)
        avg_acc  = total_acc  / max(1, total_n)
        return avg_loss, avg_acc
    
    def train_keto_discriminator(self, load_from: str | None = None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- dataset & loaders ---
        df = pd.read_csv(os.path.join(self.cfg["dataset-path"], "dataset_info.csv"))
        dataset = WayTuUnifiedDataset(self.cfg, df, device)

        val_split = float(self.cfg.get("val_split", 0.2))
        n_val   = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)

        # --- model ---
        discriminator = KetoAdaptDiscriminator().to(device)
        if load_from:
            discriminator.load_state_dict(torch.load(load_from, map_location=device))

        # --- opt & loss ---
        params = list(discriminator.parameters())
        if self.optimizer_type == 'Adam':
            optimizer = optim.Adam(params, lr=float(self.cfg.get("lr", 1e-3)))
        elif self.optimizer_type == 'SGDM':
            optimizer = optim.SGD(params, lr=float(self.cfg.get("lr", 1e-3)), momentum=0.9)
        else:
            raise NotImplementedError(f"optimizer {self.optimizer_type} not supported")

        criterion = nn.MSELoss(reduction='mean')

        best_val = float('inf')

        for epoch in range(1, self.cfg["num-epochs-disc"] + 1):
            train_loss = self._disc_epoch(discriminator, train_loader, device, optimizer, criterion, train=True)
            with torch.no_grad():
                val_loss = self._disc_epoch(discriminator, val_loader, device, optimizer=None, criterion=criterion, train=False)

            print(f"[Disc Epoch {epoch:03d}] train_mse={train_loss:.4f} | val_mse={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(discriminator.state_dict(), os.path.join(self.best_ckpt_dir, 'best_discriminator.pth'))

        # save last
        torch.save(discriminator.state_dict(), self.disc_path)

    @staticmethod
    def _disc_epoch(model, loader, device, optimizer, criterion, train: bool):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_n = 0

        for batch in tqdm(loader, desc="Disc Train" if train else "Disc Val", leave=False):
            # Expect dict from WayTuUnifiedDataset; adapt if you use the old tuple dataset.
            # Inputs:
            #   - "tool_points": (B, Nt, 3)  (used like your point_cloud)
            #   - waypoints: "tool_waypoint", "initial_waypoint", "goal_waypoint" (each (B,7))
            #   - "score": scalar in [0,1] (or your scaled score)
            tool_pts = batch["tool_points"].to(device).float()   # (B, Nt, 3)
            env_pts  = batch["env_points"].to(device).float()    # (B, Ne, 3)

            # concat along the point dimension → (B, Nt+Ne, 3)
            point_cloud = torch.cat([tool_pts, env_pts], dim=1)

            # Compute center and scale per sample
            center = point_cloud.mean(dim=1, keepdim=True)        # (B, 1, 3)
            pc_centered = point_cloud - center                    # (B, Nt+Ne, 3)
            scale = pc_centered.norm(dim=2).max(dim=1, keepdim=True)[0]  # (B, 1)

            # Normalize point cloud
            point_cloud = pc_centered / scale.unsqueeze(-1)       # (B, Nt+Ne, 3)

            # waypoints: take positions only (B,3) for each keypoint
            grasp_point   = batch["tool_waypoint"][:,   0:3].to(device).float()   # (B,3)
            initial_point = batch["initial_waypoint"][:,0:3].to(device).float()   # (B,3)
            goal_point    = batch["goal_waypoint"][:,   0:3].to(device).float()   # (B,3)
            y    = batch["score"].to(device).float().unsqueeze(1)

            grasp_point   = ((grasp_point   - center.squeeze(1)) / scale).unsqueeze(1)
            initial_point = ((initial_point - center.squeeze(1)) / scale).unsqueeze(1)
            goal_point    = ((goal_point    - center.squeeze(1)) / scale).unsqueeze(1)
            keypoints = [grasp_point, initial_point, goal_point]

            if train:
                optimizer.zero_grad()

            pred = model(point_cloud, keypoints)    # (B,1) regression
            loss = criterion(pred, y)

            if train:
                loss.backward()
                optimizer.step()

            bs = point_cloud.size(0)
            total_loss += float(loss.item()) * bs
            total_n    += bs

        return total_loss / max(1, total_n)
    
    def test_keto(self,
                  tool_points: np.ndarray | torch.Tensor,
                  env_points:  np.ndarray | torch.Tensor | None = None,
                  num_samples: int = 128,
                  truncated_normal: bool = False):
        """
        Generate waypoint candidates with the decoder and select the best
        by the discriminator.

        Args:
            tool_points: (Nt,3) tool point cloud in world coords.
            env_points:  (Ne,3) optional env cloud (will be concatenated).
            num_samples: number of decoder samples to draw.
            truncated_normal: if True, decoder uses truncated sampling.

        Returns:
            best_kps: np.ndarray of shape (3,3) [grasp, initial, goal]
            best_logit: float (discriminator logit; lower is better)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Prepare point cloud (B, N, 3) ----
        def _to_torch(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x.astype(np.float32))
            return x.float()

        pc = _to_torch(tool_points)
        if env_points is not None:
            env = _to_torch(env_points)
            pc  = torch.cat([pc, env], dim=0)  # (N,3)



        center = pc.mean(dim=0)
        pc = pc - center
        scale = pc.norm(dim=1).max()

        # convert for later denormalization
        center_np = center.cpu().numpy()
        scale_np = scale.item()     

        assert pc.ndim == 2 and pc.size(1) == 3, f"pc shape must be (N,3), got {tuple(pc.shape)}"
        pc = pc.unsqueeze(0).to(device)           # (1,N,3)
        pc = pc.repeat(num_samples, 1, 1)         # (B,N,3) with B=num_samples

        # ---- Load models ----
        decoder = KetoAdaptDecoder().to(device).eval()
        discriminator = KetoAdaptDiscriminator().to(device).eval()

        # allow both absolute and already-loaded paths in cfg
        dec_path = self.cfg.get('decoder-model', self.decoder_path)
        disc_path = self.cfg.get('disc-model', self.disc_path)
        decoder.load_state_dict(torch.load(dec_path, map_location=device))
        discriminator.load_state_dict(torch.load(disc_path, map_location=device))

        # ---- Build latent_var = [mu, sigma] with mu=0, sigma=1 ----
        # Your encoder outputs 4 dims → split into (2,2) as (mu, sigma).
        # We mimic prior sampling by giving the decoder mu=0, sigma=1;
        # it will internally do z = mu + sigma * randn_like(sigma).
        B = num_samples
        latent_mu    = torch.zeros(B, 2, device=device)
        latent_sigma = torch.ones( B, 2, device=device)
        latent_var   = torch.cat([latent_mu, latent_sigma], dim=1)  # (B,4)

        # ---- Decode keypoints ----
        # decoder returns [grasp, initial, goal], each (B,3)
        kps_list = decoder(pc, latent_var, truncated_normal=truncated_normal)
        grasp_pred, init_pred, goal_pred = kps_list  # each (B,3)

        # ---- Score candidates with discriminator ----
        # Discriminator expects ks as a list of (B,3)
        ks = [grasp_pred, init_pred, goal_pred]   # list of length 3, each (B,3)
        logits = discriminator(pc, ks)            # (B,1)
        logits = logits.view(-1)                  # (B,)

        # We trained BCEWithLogits with target=0.0 → lower logit = better
        best_idx   = int(torch.argmin(logits).item())
        best_logit = float(logits[best_idx].item())

        # ---- Extract & return the best triplet ----
        best_grasp = grasp_pred[best_idx].detach().cpu().numpy()  # (3,)
        best_init  = init_pred [best_idx].detach().cpu().numpy()  # (3,)
        best_goal  = goal_pred [best_idx].detach().cpu().numpy()  # (3,)

        best_grasp = best_grasp * scale_np + center_np
        best_init  = best_init  * scale_np + center_np
        best_goal  = best_goal  * scale_np + center_np

        best_kps = np.stack([best_grasp, best_init, best_goal], axis=0)  # (3,3)
        return best_kps, best_logit
    
    # def test_keto(self, point_cloud):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     latent_var = torch.cat([torch.zeros(self.cfg['num-samples'], 2).to(device), torch.ones(self.cfg['num-samples'], 2).to(device)], dim=1).to(device)
    #     self.decoder = KetoAdaptDecoder().to(device)

    #     self.decoder.load_state_dict(torch.load(self.cfg['decoder-model']))

    #     keypoints_vae = self.decoder(point_cloud, latent_var)

    #     # print("42: ", point_cloud.shape) # torch.Size([1, 2000, 3])
    #     # print("42: ", keypoints_vae[0].shape) # torch.Size([1, 3])
    #     keypoints_cat = torch.cat([k.unsqueeze(1) for k in keypoints_vae], dim=1)
    #     # print("42: ", keypoints_cat.shape) # torch.Size([1, 3, 3])
        
    #     keypoints_cat = keypoints_cat.unsqueeze(2)  # Shape: [1, 3, 1, 3]
    #     point_cloud_exp = point_cloud.unsqueeze(1)  # Shape: [1, 1, 2000, 3]
    #     dist_mat = torch.norm(point_cloud_exp - keypoints_cat, dim=3)

    #     dist = torch.max(torch.min(dist_mat, dim=2).values, dim=1).values
    #     dist_min = torch.min(dist)

    #     mask = (dist < self.dist_thres) | (dist == dist_min)

    #     keypoints_vae = [k[mask] for k in keypoints_vae]

    #     point_cloud_discr = point_cloud[mask]

    #     self.discriminator = KetoAdaptDiscriminator().to(device)
    #     self.discriminator.load_state_dict(torch.load(self.cfg['disc-model']))
    #     score = self.discriminator(point_cloud, keypoints_vae)

    #     index = torch.argmax(score.view(-1))
    #     top_score = score[index]
    #     top_keypoints = [k[index] for k in keypoints_vae]

    #     return top_keypoints, top_score
    

    # def train_keto_discriminator_old(self, model_path = None): 
    #     # data_paths = hlp.get_combined_dataset(self.dataset_path)
    #     dataset = WayTuDataset(data_paths=data_paths, task=self.task)
    #     dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle=True, num_workers=4)

    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #     self.discriminator = KetoAdaptDiscriminator()
    #     self.discriminator.to(device)
    #     if model_path:
    #         self.discriminator.load_state_dict(torch.load(self.cfg['disc-model']))
        
    #     parameters = list(self.discriminator.parameters())

    #     if self.optimizer_type == 'Adam':
    #         self.optimizer = optim.Adam(parameters, lr=0.001)
    #     elif self.optimizer_type == 'SGDM':
    #         self.optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9)
    #     else:
    #         raise NotImplementedError
        
    #     self.criterion = nn.MSELoss()

    #     for epoch in range(self.num_epoch):
    #         loss = self.disc_one_epoch(dataloader, device)
    #         print(f"Epoch: {epoch} Total Loss: {loss}")
        
    #     torch.save(self.discriminator.state_dict(), self.cfg['disc-model'])
    
    # def disc_one_epoch(self, dataloader, device):
    #     epoch_loss = 0

    #     for i, data in enumerate(dataloader):
    #         point_cloud, waypoint, score,task_id = data

    #         score = score.unsqueeze(1).float().to(device)

    #         grasp_point = waypoint[:,0:3].float().to(device)
    #         initial_point = waypoint[:,7:10].float().to(device)
    #         goal_point = waypoint[:,14:17].float().to(device)

    #         keypoints = [grasp_point, initial_point, goal_point]

    #         point_cloud = point_cloud.to(device)

    #         self.optimizer.zero_grad()
    #         output = self.discriminator(point_cloud, keypoints)
    #         loss = self.criterion(output, score)

    #         loss.backward()
    #         self.optimizer.step()

    #         epoch_loss += loss.item()

    #     return epoch_loss


    
    # def train_keto_vae_old(self):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     df = pd.read_csv(os.path.join(self.cfg["dataset_dir"], "dataset_info.csv"))
    #     dataset = WayTuUnifiedDataset(self.cfg, df, device)

    #     n_val = int(len(dataset) * self.cfg.get("val_split", 0.2))
    #     train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    #     train_loader = DataLoader(train_ds, batch_size=self.cfg["batch_size"], shuffle=True, num_workers=4)
    #     val_loader = DataLoader(val_ds, batch_size=self.cfg["batch_size"], shuffle=False, num_workers=4)


    #     # self.encoder = KetoKeypointEncoder().to(device)
    #     self.encoder = KetoAdaptEncoder().to(device)
    #     # self.decoder = KetoKeypointDecoder().to(device)
    #     self.decoder = KetoAdaptDecoder().to(device)
    #     # self.discriminator = KetoKeypointDiscriminator().to(device)
    #     self.discriminator = KetoAdaptDiscriminator().to(device)
    #     parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.discriminator.parameters())
        
    #     if self.optimizer_type == 'Adam':
    #         self.optimizer = optim.Adam(parameters, lr=0.001)
    #     elif self.optimizer_type == 'SGDM':
    #         self.optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9)
    #     else:
    #         raise NotImplementedError

    #     for epoch in range(self.num_epoch):
    #         # self.one_epoch(dataloader,device)
    #         loss, accuracy = self.one_adaptation_epoch(dataloader, device)
    #         print(f"Epoch: {epoch} Total Loss: {loss} Accuracy: {accuracy}")
        
    #     torch.save(self.encoder.state_dict(), self.cfg['encoder-model'])
    #     torch.save(self.decoder.state_dict(), self.cfg['decoder-model'])
    #     torch.save(self.discriminator.state_dict(), self.cfg['disc-model'])


    # def one_adaptation_epoch(self,dataloader, device):
    #     epoch_loss = 0 
    #     epoch_accuracy = 0
    #     for i, data in enumerate(dataloader):
    #         point_cloud, waypoint, score,task_id = data

    #         grasp_point = waypoint[:,0:3].float().to(device)
    #         initial_point = waypoint[:,7:10].float().to(device)
    #         goal_point = waypoint[:,14:17].float().to(device)

    #         point_cloud = point_cloud.to(device)
            
    #         #Updated for going through 3 keypoints
    #         keypoints = [grasp_point, initial_point, goal_point]
    #         keypoints_labels = torch.zeros((grasp_point.shape[0], 1), device=self.device)
            
    #         self.optimizer.zero_grad()
    #         # print(next(self.encoder.parameters()).is_cuda)
    #         latent_var = self.encoder(point_cloud, keypoints) # z=miu, sigma 

            

    #         z_mean, z_std = torch.chunk(latent_var, 2, dim=1)
    #         z = z_mean + z_std * torch.randn_like(z_std)
    #         z_std = torch.mean(torch.std(z, dim=1))
    #         z_mean = torch.mean(z_mean)

    #         keypoint_vae = self.decoder(point_cloud, latent_var)
    #         grasp_point_vae, initial_vae_point, goal_vae_point = keypoint_vae


    #         # print("keypoint_vae: ", grasp_point_vae.shape)
    #         # print("keypoint: ", grasp_point.shape)


    #         loss_vae_grasp = F.l1_loss(grasp_point_vae, grasp_point)
    #         loss_vae_initial = F.l1_loss(initial_vae_point, initial_point)
    #         loss_vae_goal = F.l1_loss(goal_vae_point, goal_point)

    #         point_cloud_mean = torch.mean(point_cloud, dim=1, keepdim=True)

    #         std_gt_grasp = torch.mean(torch.std(grasp_point - point_cloud_mean, dim=0))
    #         std_gt_initial = torch.mean(torch.std(initial_point - point_cloud_mean, dim=0))
    #         std_g_goal = torch.mean(torch.std(goal_point - point_cloud_mean, dim=0))

    #         miu, sigma = torch.chunk(latent_var, 2, dim=1)
    #         loss_vae_mmd = torch.mean(miu**2 + sigma**2 - torch.log(1e-8 + sigma**2) - 1)

    #         discr_logit = self.discriminator(point_cloud, keypoints)

    #         # raise Exception

    #         loss_discr = F.binary_cross_entropy_with_logits(discr_logit, keypoints_labels)
            
    #         loss_vae = loss_vae_grasp + loss_vae_initial + loss_vae_goal + loss_vae_mmd
    #         total_loss = loss_vae + loss_discr

    #         total_loss.backward()
    #         self.optimizer.step()
            
    #         pred_labels = (torch.sigmoid(discr_logit) < 0.5 ).float()
    #         pred_equal_gt = (pred_labels == keypoints_labels).float()
    #         acc_discr = torch.mean(pred_equal_gt)

    #         prec_discr = torch.sum(pred_equal_gt * pred_labels) / (torch.sum(pred_labels)+ 1e-6)
    #         epoch_loss += total_loss.item()
    #         epoch_accuracy += acc_discr
    #     return epoch_loss, epoch_accuracy 
            

    # def one_epoch(self,dataloader, device):
    #     for i, data in enumerate(dataloader):
    #         point_cloud, waypoint, score,task_id = data

    #         grasp_point = waypoint[:,0:3].float().to(device)
    #         print(f"grasp_point shape: {grasp_point.shape}")
    #         function_point = waypoint[:,7:10].float().to(device)
    #         funct_vect = waypoint[:,14:17].float().to(device)

    #         point_cloud = point_cloud.to(device)

    #         keypoints = [grasp_point, function_point]
    #         keypoints_labels = torch.zeros((self.batch_size, 1), device=self.device)
            
    #         self.optimizer.zero_grad()
    #         print(next(self.encoder.parameters()).is_cuda)
    #         latent_var = self.encoder(point_cloud, keypoints, funct_vect)

    #         z_mean, z_std = torch.chunk(latent_var, 2, dim=1)
    #         z = z_mean + z_std * torch.randn_like(z_std)
    #         z_std = torch.mean(torch.std(z, dim=1))
    #         z_mean = torch.mean(z_mean)

    #         keypoint_vae, funct_vect_vae = self.decoder(point_cloud, latent_var, self.num_funt_vect)
    #         grasp_point_vae, funct_point_vae = keypoint_vae
    #         print("keypoint_vae: ", grasp_point_vae.shape)
    #         print("keypoint: ", grasp_point.shape)


    #         loss_vae_grasp = F.l1_loss(grasp_point_vae, grasp_point)
    #         loss_vae_funct = F.l1_loss(funct_point_vae, function_point)
    #         loss_vae_func_vect = F.l1_loss(funct_vect_vae, funct_vect)

    #         point_cloud_mean = torch.mean(point_cloud, dim=1, keepdim=True)

    #         std_gt_grasp = torch.mean(torch.std(grasp_point - point_cloud_mean, dim=0))
    #         std_gt_funct = torch.mean(torch.std(function_point - point_cloud_mean, dim=0))
    #         std_gt_funct_vect = torch.mean(torch.std(funct_vect, dim=0))

    #         miu, sigma = torch.chunk(latent_var, 2, dim=1)
    #         loss_vae_mmd = torch.mean(miu**2 + sigma**2 - torch.log(1e-8 + sigma**2) - 1)

    #         discr_logit = self.discriminator(point_cloud, keypoints, funct_vect)

    #         loss_discr = F.binary_cross_entropy_with_logits(discr_logit, keypoints_labels)
            
    #         loss_vae = loss_vae_grasp + loss_vae_funct + loss_vae_func_vect + loss_vae_mmd
    #         total_loss = loss_vae + loss_discr

    #         total_loss.backward()
    #         self.optimizer.step()
            
    #         pred_labels = (torch.sigmoid(discr_logit) < 0.5 ).float()
    #         pred_equal_gt = (pred_labels == keypoints_labels).float()
    #         acc_discr = torch.mean(pred_equal_gt)

    #         prec_discr = torch.sum(pred_equal_gt * pred_labels) / (torch.sum(pred_labels)+ 1e-6)

    #         print(f"Total Loss: {total_loss.item()} Accuracy: {acc_discr}")
                     

if __name__ == '__main__':
    parameters = {
        "num-funct-vect" : 1,
        "num-epoch" : 60,
        "num-epochs-disc" : 20,
        "batch-size" : 64,
        "num-training-points": 512,

        "optimizer-type" : 'Adam', # [Adam, SGDM] 
        "encoder-model"     : 'models/minigolf-encoder-n2-last.pt',
        "decoder-model"     : 'models/minigolf-decoder-n2-last.pt',
        "disc-model"        : 'models/minigolf-disc-n2-last.pt',
        "dataset-path": "Datasets/minigolf-dataset-top1800-pc",
        "val_split": 0.2,
        "task" : "minigolf",
        "label-list-all": ["lifting-platform", "minigolf-platform", "hammering-platform",
                  "hammer", "spatula", "L-ruler", "screwdriver", "ball", "book", "thin-stick", "ring"],
    }
    keto = Keto_to_WayTu(keto_cfg= parameters)
    keto.train_keto_vae()

    keto.train_keto_discriminator(parameters['disc-model'])

# This is the configuration file for KETO adaptation
# num-funct-vect    : 1
# num-epoch         : 60
# batch-size        : 128
# optimizer-type    : 'Adam' # [Adam, SGDM] 

# task              : 'reach' 
# dataset-path      : '/home/ece/git/Way-Tu/reach-dataset'
# normalize-pc      : True

# encoder-model     : '/home/ece/git/Way-Tu/Baselines/reach-encoder-last.pt'
# decoder-model     : '/home/ece/git/Way-Tu/Baselines/reach-decoder-last.pt'
# disc-model        : '/home/ece/git/Way-Tu/Baselines/reach-disc-last.pt'

# num-samples       : 1

# # In RAI experiments
# num-trials        : 1
# from Baselines.keto_waytu_adaptation import Keto_to_WayTu
# import yaml
# import os

# with open(os.path.join('Baselines','keto_config.yaml'), 'r') as f:
#     cfg = yaml.safe_load(f)

