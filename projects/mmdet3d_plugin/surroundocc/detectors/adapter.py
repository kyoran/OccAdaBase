import torch
import torch.nn as nn
import torch.nn.functional as F

class OccupancyAdapter(nn.Module):
    def __init__(self):
        super(OccupancyAdapter, self).__init__()
        self.adapt_conv1 = nn.Conv3d(6, 17, kernel_size=1, stride=1, padding=0) # 凑一下数
        self.adapt_conv2 = nn.Conv3d(6, 17, kernel_size=1, stride=1, padding=0)
        self.adapt_conv3 = nn.Conv3d(6, 17, kernel_size=1, stride=1, padding=0)
        self.output_conv1 = nn.Conv3d(17, 17, kernel_size=3, stride=1, padding=1)
        self.output_conv2 = nn.Conv3d(17, 17, kernel_size=3, stride=1, padding=1)
        self.output_conv3 = nn.Conv3d(17, 17, kernel_size=3, stride=1, padding=1)
        self.output_conv4 = nn.Conv3d(17, 17, kernel_size=3, stride=1, padding=1)

    def forward(self, mlvl_feats):
        x1 = self.adapt_conv1(mlvl_feats[0])
        x2 = self.adapt_conv2(mlvl_feats[1])
        x3 = self.adapt_conv3(mlvl_feats[2])
        x1 = F.interpolate(x1, size=(25, 25, 2), mode='trilinear', align_corners=True)
        x1 = self.output_conv1(x1)
        x2 = F.interpolate(x2, size=(50, 50, 4), mode='trilinear', align_corners=True)
        x2 = self.output_conv2(x2)
        x3 = F.interpolate(x3, size=(100, 100, 8), mode='trilinear', align_corners=True)
        x3 = self.output_conv3(x3)
        x4 = F.interpolate(x3, size=(200, 200, 16), mode='trilinear', align_corners=True) # 这一层先直接从上一层采样吧
        x4 = self.output_conv4(x4)

        return [x1, x2, x3, x4]
