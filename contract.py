import torch # 命令行是逐行立即执行的
import numpy as np
model_pre = torch.load('./ckpts/r101_dcn_fcos3d_pretrain.pth')
model_epoch1 = torch.load('./work_dirs/surroundocc/epoch_1.pth')
model_epoch2 = torch.load('./work_dirs/surroundocc/epoch_2.pth')
s1 = model_pre['state_dict']
s2 = model_epoch1['state_dict']
s3 = model_epoch2['state_dict']
# for k2, k3 in zip(s2, s3):
#     print(s2[k2] == s3[k3])
#print(s2["pts_bbox_head.final_conv3D.0.weight"] == s3["pts_bbox_head.final_conv3D.0.weight"])#全是false
# print(s3["pts_bbox_head.final_conv3D.0.weight"])
# print(s2["pts_bbox_head.transfer_conv.2.0.weight"] == s3["pts_bbox_head.transfer_conv.2.0.weight"])#全是true
# print(s2["img_backbone.layer1.0.conv2.weight"] == s3["img_backbone.layer1.0.conv2.weight"])#全是true
for key in s2.keys():
    flag = (s2[key]==s3[key])
    if(torch.all(flag)==False):#torch.all，如果tensor全为true,则返回true
        print(key)