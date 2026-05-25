import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class JPU(nn.Module):
    def __init__(self, in_channels, width, dilations):
        super(JPU, self).__init__() 
        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=3, padding=d, dilation=d,bias=False),
                           nn.BatchNorm2d(width),
                           nn.ReLU(inplace=True)) for d in dilations]
        )
        self.project = nn.Sequential(
            nn.Conv2d(width * len(dilations), width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        x_cat = torch.cat(feats, dim=1)
        return self.project(x_cat)

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained, in_channels, jpu_size):
        super().__init__()

        self.dilations = (1,2,4,8)

        self.initialize_dResNet(pretrained, in_channels)

        self.reduce_c3 = nn.Conv2d(512, jpu_size, kernel_size=1, bias=False)
        self.reduce_c4 = nn.Conv2d(1024, jpu_size, kernel_size=1, bias=False)
        self.reduce_c5 = nn.Conv2d(2048, jpu_size, kernel_size=1, bias=False)

        self.jpu = JPU(in_channels=jpu_size*3, width=jpu_size, dilations=self.dilations)
    def initialize_dResNet(self, pretrained, in_channels):
        resnet = models.resnet50(pretrained = pretrained)
        layers = list(resnet.children())

        dstem = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels= 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels= 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels= 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        )

        self.dstem = dstem
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        
    def forward(self, x):
        c1 = self.dstem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        f3 = self.reduce_c3(c3)
        f4 = F.interpolate(self.reduce_c4(c4), scale_factor=2, mode='bilinear', align_corners=False)
        f5 = F.interpolate(self.reduce_c5(c5), scale_factor=4, mode='bilinear', align_corners=False)

        jpu_input = torch.cat([f3, f4, f5], dim=1)
        return self.jpu(jpu_input)

class EncodingLayer(nn.Module):
    def __init__(self, channels, E):
        super().__init__()

        self.C = channels
        self.E = E          # Size of learnable codebook

        self.codewords = nn.Parameter(torch.randn(self.E, self.C))
        self.scale = nn.Parameter(torch.ones(self.E))
        self.batch_norm = nn.BatchNorm1d(self.C * self.E)
    
    # def forward(self, x):
    #     B, C, H, W = x.shape
    #     N = H * W 
    #     X = x.view(B, C, N).permute(0, 2, 1)
    #     residuals = x.unsqueeze(2) - self.codewords.unsqueeze(0)
    #     dist2 = (residuals **2).sum(-1)
    #     weights = F.softmax(-self.scale.unsqueeze(0).unsqueeze(0) * dist2, dim=-1)
    #     agg = (weights.unsqueeze(-1) * residuals).sum(dim=1)

    #     agg = agg.view(B, -1)
    #     agg = self.batch_norm(agg)
    #     agg = F.relu(agg, inplace=True)

        # return agg 
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N          = H * W

        # 1) flatten spatial → (B, N, C)
        X = x.view(B, C, N).permute(0, 2, 1)

        # 2) broadcast codewords → (1, 1, E, C)
        cw = self.codewords.unsqueeze(0).unsqueeze(0)

        # 3) residuals → (B, N, E, C)
        residuals = X.unsqueeze(2) - cw

        # 4) squared dist → (B, N, E)
        dist2 = (residuals ** 2).sum(-1)

        # 5) soft‐assignment weights → (B, N, E)
        scale   = self.scale.unsqueeze(0).unsqueeze(0)  
        weights = F.softmax(-scale * dist2, dim=-1)

        # 6) aggregate residuals → (B, E, C)
        agg = (weights.unsqueeze(-1) * residuals).sum(dim=1)

        # 7) BN + ReLU on flattened (E*C) vector
        agg = agg.view(B, -1)  
        agg = self.batch_norm(agg)
        return F.relu(agg, inplace=True)

class ActionBranch(nn.Module):
    def __init__(self, feature_size, num_orient):
        super().__init__()
        self.E = 32
        self.upsample_scale = 8
        self.num_orient = num_orient
        self.enc_layer = EncodingLayer(channels=feature_size, E = self.E)
        self.dropout = nn.Dropout(0.1)
        self.spatial_projection = nn.Linear(self.E*feature_size, feature_size)
        self.conv1x1 = nn.Conv2d(feature_size, num_orient, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        ctx = self.enc_layer(x)
        ctx = self.dropout(ctx)

        ctx_sp = self.spatial_projection(ctx)
        ctx_sp = ctx_sp.view(B, C, 1, 1)
        ctx_sp = ctx_sp.expand(-1, -1, H, W)

        heatmap = torch.sigmoid(self.conv1x1(ctx_sp))
        upsampled = F.interpolate(heatmap, scale_factor=self.upsample_scale, mode="bilinear", align_corners=False)

        return upsampled

class ToolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.jpu_width = 2048
        self.bridge_size = 512
        self.rgb_feature_extractor = FeatureExtractor(pretrained= False, in_channels=3, jpu_size=self.jpu_width)
        self.ddd_feature_extractor = FeatureExtractor(pretrained= False, in_channels=3, jpu_size=self.jpu_width)
    
        self.bridge = nn.Sequential(
            nn.Conv2d(in_channels= self.jpu_width*2, out_channels=self.bridge_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.bridge_size),
            nn.ReLU(inplace=True)
        )

        self.grasp_branch = ActionBranch(feature_size=self.bridge_size, num_orient=16)
        self.manip_branch = ActionBranch(feature_size=self.bridge_size, num_orient=32)


    def forward(self, rgb, depth):
        ddd = depth.repeat(1,3,1,1)
        rgb_features = self.rgb_feature_extractor(rgb)
        ddd_features = self.ddd_feature_extractor(ddd)

        features = torch.cat([rgb_features, ddd_features], dim = 1)
        bridged = self.bridge(features)

        grasp_heatmap = self.grasp_branch(bridged)
        manip_heatmap = self.manip_branch(bridged)

        return grasp_heatmap, manip_heatmap



tool_net = ToolNet()
