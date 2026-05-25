import torch
import torch.nn as nn
import torch.nn.functional as F


class KetoAdaptEncoder(nn.Module):
    def __init__(self):
        super(KetoAdaptEncoder, self).__init__()
        
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=(1, 1))
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=(1, 1))
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=(1, 1))
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(1, 1))
        self.conv5_1 = nn.Conv2d(32, 32, kernel_size=(1, 1))
        self.conv5_2 = nn.Conv2d(32, 256, kernel_size=(1, 1))

        # Updated for dimentionality
        self.fc1 = nn.Linear(1024, 256) # nn.Linear(256, 256) 
        self.fc2 = nn.Linear(256, 256) # nn.Linear(256, 256)

        self.fc_out = nn.Linear(256, 4)

        self.v_fc1 = nn.Linear(3, 32)
        self.v_fc2 = nn.Linear(32, 64)
        self.v_fc3 = nn.Linear(64, 64)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(256)
    
    def conv_layer(self, x, layer, bn, linear=False):
        x = layer(x)
        if not linear:
            x = F.relu(x)
            x = bn(x)
        return x
    
    def fc_layer(self, x, layer, linear=False):
        # print("x layer: ", x.shape)
        x = layer(x)
        if not linear:
            x = F.relu(x)
        return x
    
    def forward(self, x, ks, v=None):
        # Now x= point_cloud ks=3 keypoints
        x = x.float()

        mean_x = torch.mean(x, dim=1, keepdim=True)
        x = x - mean_x


        mean_r = torch.mean(torch.norm(x, dim=2, keepdim=True), dim=1, keepdim=True)
        x = x / (mean_r + 1e-6)

        ks = [k.unsqueeze(2) for k in ks] # ks = keypoints = [grasp_point, function_point]
        num_keypoints = [k.size(2) for k in ks]
         
        ks = [k.reshape(mean_x.shape) for k in ks]

        ks = [(k - mean_x) / (mean_r + 1e-6) for k in ks]

        _, num_points, _ = x.size()

        x = torch.cat([x] + ks, dim=1)

        x = x.unsqueeze(1)
        
        x = self.conv_layer(x, self.conv1_1, self.bn1)
        x = self.conv_layer(x, self.conv1_2, self.bn1)
        x = self.conv_layer(x, self.conv2_1, self.bn2)
        x = self.conv_layer(x, self.conv2_2, self.bn2)
        x = self.conv_layer(x, self.conv5_1, self.bn2)
        x = self.conv_layer(x, self.conv5_2, self.bn3)

        x_ks = torch.split(x, [num_points] + num_keypoints, dim=2)
        x, ks = x_ks[0], x_ks[1:]


        x = torch.amax(x, dim=[2, 3], keepdim=False)

        ks = [torch.amax(k, dim=[2, 3], keepdim=False) for k in ks]

        x = torch.cat([x] + ks, dim=1)


        x = self.fc_layer(x, self.fc1)

        # if v is not None:
        #     v = v / (1e-6 + torch.norm(v, dim=-1, keepdim=True))
        #     # print("v.shape: ", v.shape)
        #     v = self.fc_layer(v, self.v_fc1)
        #     v = self.fc_layer(v, self.v_fc2)
        #     v = self.fc_layer(v, self.v_fc3)
        #     x = torch.cat([x, v], dim=1)

        x = self.fc_layer(x, self.fc2)
        z = self.fc_layer(x, self.fc_out, linear=True)

        miu, sigma = torch.chunk(z, 2, dim=1)

        sigma_sp = F.softplus(sigma)
        z = torch.cat([miu, sigma_sp], dim=1)

        return z

class KetoAdaptDecoder(nn.Module):
    def __init__(self):
        super(KetoAdaptDecoder, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=(1, 1))
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=(1, 1))

        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=(1, 1))
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(1, 1))


        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=(1, 1))
        self.conv3_2 =  nn.Conv2d(64, 256, kernel_size=(1, 1))

        self.conv6_1 = nn.Conv2d(256, 256, kernel_size=(1, 1))
        self.conv6_2 = nn.Conv2d(1, 3, kernel_size=(1, 1))

        #Additional layers for goal keyypoint
        self.conv62_1 = nn.Conv2d(256, 256, kernel_size=(1, 1))
        self.conv62_2 = nn.Conv2d(1, 3, kernel_size=(1, 1))

        self.conv7_1 = nn.Conv2d(256, 256, kernel_size=(1, 1))
        self.conv7_2 = nn.Conv2d(1, 3, kernel_size=(1, 1))

        self.conv8_1 = nn.Conv2d(256, 256, kernel_size=(1, 1))
        self.conv8_2 = nn.Conv2d(1, 3, kernel_size=(1, 1))

        self.grasping_linear = nn.Linear(7,1)
        self.initial_linear = nn.Linear(7,1)
        self.goal_liner = nn.Linear(7,1)

    
    def conv_layer(self, x, layer, linear=False):
        x = layer(x)
        if not linear:
            x = F.relu(x)
        return x
    
    def concat_xz(self, x, z):
        z_n = z.expand(-1, x.size(1), x.size(2), -1) # Definitely not sure 

        x_z_con = torch.cat([x, z_n], dim=3)

        return x_z_con


    def down_sample(self, x, p, stride, thres):
        # Max pooling to downsample the points
        new_p = F.max_pool2d(p, kernel_size=(1, 1), stride=(stride, 1))
        x = F.max_pool2d(x, kernel_size=(1, 1), stride=(stride, 1)) 
        p = p.unsqueeze(4)
        
        d = torch.unsqueeze(new_p.permute(0,1, 3, 2 ), 1)
        d = d - p
        d = torch.norm(d, dim=(2, 3))
        
        _, h, w = d.shape
        
        d_mean = torch.mean(d, dim=1, keepdim=True)
        mask = (d < (d_mean * thres)).float()

        mask = mask.permute(0, 2, 1)

        mask_sum = torch.sum(mask, dim=1, keepdim=True) + 1e-6
        x_size_1 = x.size(1)
        x = x.permute(0, 1, 3, 2).reshape(x.size(0), x.size(1) * x.size(3), x.size(2))

        x = torch.matmul(x, mask) / mask_sum

        x = x.reshape(x.shape[0], x_size_1, int(x.shape[1]/x_size_1))

        x = x.unsqueeze(2) # x.permute(0, 2, 1)
        
        return x, new_p
    
    def forward(self, x, z, nv=0, truncated_normal=False):
        x = x.float()
        miu, sigma = torch.chunk(z, 2, dim=1)

        if truncated_normal:
            z = miu + sigma * torch.fmod(torch.randn_like(sigma), 2)
        else:
            z = miu + sigma * torch.randn_like(sigma)
        
        # In here probably we get a mu and sigma value for every sample in batch)
        z = z.view(-1, 1, 1, z.size(1))
        
        mean_x = x.mean(dim=1, keepdim=True)
        x = x - mean_x
        
        mean_r = torch.mean(torch.norm(x, dim=2, keepdim=True), dim=1, keepdim=True)
        x = x / (mean_r + 1e-6)
        x = x.unsqueeze(1)
        p = x
        

        x = self.conv_layer(x, self.conv1_1)
        x = self.conv_layer(x, self.conv1_2)
        x = self.concat_xz(x, z)
        
        x = self.conv_layer(x, self.conv2_1)
        x = self.conv_layer(x, self.conv2_2)
        x = self.concat_xz(x, z)
        
        x = self.conv_layer(x, self.conv3_1)
        x = self.conv_layer(x, self.conv3_2)
        x, p = self.down_sample(x, p ,64, 0.7) # -> did not understand or nworked 

        x_f = self.conv_layer(x, self.conv6_1)
        x_f = x_f.max(dim=1, keepdim=True)[0]

        initial_keypoints = self.conv_layer(x_f, self.conv6_2, linear=True)

        mean_r_expanded = mean_r.unsqueeze(-1) 
        mean_x_expanded = mean_r.unsqueeze(-1) 

        
        initial_keypoints = initial_keypoints * (mean_r_expanded+ 1e-6) 
        initial_keypoints = initial_keypoints+ mean_x_expanded


        # for keypoints there should not be 2000 points
        x_g = self.conv_layer(x, self.conv7_1)

        x_g = x_g.max(dim=1, keepdim=True)[0]

        grasping_keypoints = self.conv_layer(x_g, self.conv7_2, linear=True)  * (mean_r_expanded + 1e-6) + mean_x_expanded

        x_e = self.conv_layer(x, self.conv62_1)
        x_e = x_e.max(dim=1, keepdim=True)[0]
        goal_keypoints = self.conv_layer(x_e, self.conv6_2, linear=True) * (mean_r_expanded + 1e-6) + mean_x_expanded
        
        # if nv:
        #     # print(nv)
        #     x_v = self.conv_layer(x, self.conv8_1)
        #     x_v = x_v.max(dim=1, keepdim=True)[0]
        #     # print("x_v:", x_v.shape)
        #     x_v = self.conv_layer(x_v, self.conv8_2, linear=True).squeeze(2)
        #     # print("x_v:", x_v.shape)
        #     # x_v = x_v.squeeze(1)
        #     # funct_vect = x_v.view(-1, nv, 3)
        #     funct_vect = x_v
        #     # print("300: ", funct_vect.shape)
        # else:
        #     funct_vect = None
        

        # Additional dimensional correction
        grasping_keypoints = self.grasping_linear(grasping_keypoints).squeeze(-1)
        initial_keypoints = self.initial_linear(initial_keypoints).squeeze(-1)
        goal_keypoints = self.goal_liner(goal_keypoints).squeeze(-1)

        keypoints = [grasping_keypoints, initial_keypoints,goal_keypoints]
        keypoints = [k.squeeze(2) for k in keypoints]
        return keypoints

class KetoAdaptDiscriminator(nn.Module):
    def __init__(self):
        super(KetoAdaptDiscriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=(1, 1))
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=(1, 1))
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=(1, 1))
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(1, 1))
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=(1, 1))
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=(1, 1))
        self.conv4_1 = nn.Conv2d(64, 512, kernel_size=(1, 1))
        self.fc_v1 = nn.Linear(3, 32)  # Adjust input size based on the actual input
        self.fc_v2 = nn.Linear(32, 64)
        self.fc_v3 = nn.Linear(64, 128)
        self.fc1 = nn.Linear(1280, 256)  # Adjust input size based on the actual input
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)

        self.fc_correct = nn.Linear(1536,1280)

    def conv_layer(self, x, layer, linear=False):
        x = layer(x)
        if not linear:
            x = F.relu(x)
        return x
    
    def rot_mat(self, rx, ry, rz):
        zeros = torch.zeros_like(rx)
        ones = torch.ones_like(rx)
        
        # Rotation matrix around x-axis
        Rx = torch.cat([ones, zeros, zeros,
                        zeros, torch.cos(rx), -torch.sin(rx),
                        zeros, torch.sin(rx), torch.cos(rx)], dim=1)
        Rx = Rx.view(-1, 3, 3)
        
        # Rotation matrix around y-axis
        Ry = torch.cat([torch.cos(ry), zeros, torch.sin(ry),
                        zeros, ones, zeros,
                        -torch.sin(ry), zeros, torch.cos(ry)], dim=1)
        Ry = Ry.view(-1, 3, 3)
        
        # Rotation matrix around z-axis
        Rz = torch.cat([torch.cos(rz), -torch.sin(rz), zeros,
                        torch.sin(rz), torch.cos(rz), zeros,
                        zeros, zeros, ones], dim=1)
        Rz = Rz.view(-1, 3, 3)
        
        # Combined rotation matrix
        R = torch.bmm(Rx, torch.bmm(Ry, Rz))
        return R

    def align(self, x, p):
        c, rx, ry, rz = torch.split(p, [3, 1, 1, 1], dim=1)
        R = self.rot_mat(-rx, -ry, -rz).float()
        c = c.unsqueeze(1)
        x = (x - c).permute(0, 2, 1).float()
        x = torch.bmm(R, x)
        y = x.permute(0, 2, 1)

        return y
    
    
    def forward(self, x, ks, v=None):
        g_kp, i_kp, e_kp = [k.squeeze(1) for k in ks]
        
        vx, vy, vz = torch.split(e_kp - g_kp, 1, dim=1)
        rz = torch.atan2(vy, vx)
        rx = torch.zeros_like(rz)
        pose_g = torch.cat([g_kp, rx, rx, rz], dim=1)
        pose_i = torch.cat([i_kp, rx, rx, rz], dim=1)
        pose_e = torch.cat([i_kp, rx, rx, rz], dim=1)

        
        x_g = self.align(x.squeeze(2), pose_g).unsqueeze(2)
        x_i = self.align(x.squeeze(2), pose_i).unsqueeze(2)
        x_e = self.align(x.squeeze(2), pose_e).unsqueeze(2)

        
        x = torch.cat([x_g, x_i, x_e], dim=1)
        x = x.permute(0,3,1,2) 
        

        x = self.conv_layer(x, self.conv1_1)
        x = self.conv_layer(x, self.conv1_2)
        x = self.conv_layer(x, self.conv2_1)
        x = self.conv_layer(x, self.conv2_2)
        x = self.conv_layer(x, self.conv3_1)
        x = self.conv_layer(x, self.conv3_2)
        x = self.conv_layer(x, self.conv4_1)
       
        N = x.size(2) // 3
        # (optional safety) if not divisible, clamp to the smallest equal chunks
        sizes = [N, N, x.size(2) - 2*N]
        x_g, x_i, x_e = torch.split(x, sizes, dim=2)
        
        x_g = F.adaptive_max_pool2d(x_g, (1, 1)).view(x_g.size(0), -1)
        x_i = F.adaptive_max_pool2d(x_i, (1, 1)).view(x_i.size(0), -1)
        x_e = F.adaptive_max_pool2d(x_e, (1, 1)).view(x_e.size(0), -1)

        
        x = torch.cat([x_g, x_i, x_e], dim=1)
        # print("410: ",x.shape)
        
        # if v is not None:
        #     print("427-v: ", v.shape)
        #     print("427-pose_v: ", pose_v.shape)
        #     x_v = self.align(v, pose_v)
        #     print("423: ", x_v.shape)
        #     x_v = F.relu(self.fc_v1(x_v))
        #     x_v = F.relu(self.fc_v2(x_v))
        #     x_v = F.relu(self.fc_v3(x_v))
        #     # print("426: ", x_v.shape)
        #     # print(x_v.size(0))
        #     x_v = x_v.reshape(x_v.size(0), x_v.size(1) * x_v.size(2))
        #     print("426: ", x_v.shape)
        #     print("426: ", x.shape)
        #     x = torch.cat([x, x_v], dim=1)
        x = self.fc_correct(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        p = self.fc_out(x)
        
        return p