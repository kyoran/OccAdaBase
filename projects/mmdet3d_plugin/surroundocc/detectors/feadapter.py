import torch
import torch.nn as nn
import torch.nn.functional as F

class D2Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(D2Conv3D, self).__init__()
        self.d2conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        
    def forward(self, x):
        return self.d2conv3d(x)

class FEAdapter(nn.Module):
    def __init__(self):
        super(FEAdapter, self).__init__()
        self.down_proj_0 = nn.Conv3d(6, 64, kernel_size=1)  # 进一步降低通道数
        self.down_proj_1 = nn.Conv3d(6, 64, kernel_size=1)
        self.down_proj_2 = nn.Conv3d(6, 64, kernel_size=1)
        
        self.d2conv3d_0 = D2Conv3D(64, 64)  # 进一步降低通道数
        self.d2conv3d_1 = D2Conv3D(64, 64)
        self.d2conv3d_2 = D2Conv3D(64, 64)
        
        self.up_proj_0 = nn.Conv3d(64, 17, kernel_size=1)
        self.up_proj_1 = nn.Conv3d(64, 17, kernel_size=1)
        self.up_proj_2 = nn.Conv3d(64, 17, kernel_size=1)
        self.up_proj_3 = nn.Conv3d(64, 17, kernel_size=1)

    def forward(self, mlvl_feats):
        occ_displacement = []

        # 第一层
        x = self.down_proj_0(mlvl_feats[0])
        x = self.d2conv3d_0(x)
        x = self.up_proj_0(x)
        x = F.interpolate(x, size=(25, 25, 2), mode='trilinear', align_corners=False)
        occ_displacement.append(x)

        # 第二层
        x = self.down_proj_1(mlvl_feats[1])
        x = self.d2conv3d_1(x)
        x = self.up_proj_1(x)
        x = F.interpolate(x, size=(50, 50, 4), mode='trilinear', align_corners=False)
        occ_displacement.append(x)

        # 第三层
        x = self.down_proj_2(mlvl_feats[2])
        x = self.d2conv3d_2(x)
        x = self.up_proj_2(x)
        x = F.interpolate(x, size=(100, 100, 8), mode='trilinear', align_corners=False)
        occ_displacement.append(x)

        # 第四层
        x = F.interpolate(mlvl_feats[2], size=(200, 200, 16), mode='trilinear', align_corners=False)
        x = self.down_proj_2(x)
        x = self.d2conv3d_2(x)
        x = self.up_proj_3(x)
        occ_displacement.append(x)

        return occ_displacement