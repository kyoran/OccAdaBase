import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv(nn.Module):

    def __init__(self):
        super(UpSampleConv, self).__init__() 
        self.fc = nn.Linear(32, 32 * 40 * 40 * 4)
        self.deconv1 = nn.ConvTranspose3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            padding=(0, 0, 1)
        )
        self.deconv2 = nn.ConvTranspose3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1, 5, 5),
            stride=(1, 2, 2),
            padding=(0, 2, 2)
        )   
        self.deconv3 = nn.ConvTranspose3d(
            in_channels=32,
            out_channels=17,
            kernel_size=(3, 5, 5),
            stride=(2, 2, 2),
            padding=(1, 2, 2),
            output_padding=(1, 0, 0)  # 为了匹配目标深度16
        )

    def forward(self, x):
        # offset: c=17, l=200, w=200, h=16
        # input: (32, )
        # fc: (32, 40*40*4)
        # view: (32, 40, 40, 4)
        # dconv3d: (32, 80, 80, 10)
        # [17, 200, 200, 16]
        x = self.fc(x)
        x = x.view(-1, 32, 40, 40, 4)
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = F.relu(x)
        x = F.interpolate(x, size=(200, 200, 16), mode='trilinear', align_corners=False)
        return x


# 
class FeatureFusionModule(nn.Module):

    def __init__(self):
        super(FeatureFusionModule, self).__init__()
        # [6, 512, 116, 200]->[6, 512, 1160, 1280]
        self.upsample_to_match = nn.Upsample(size=(1160, 1280), mode='bilinear', align_corners=False)
        # 减少通道数到 16
        self.conv1x1_512_to_16 = nn.Conv2d(512, 16, kernel_size=1)
        # 融合后的处理
        self.conv_fusion = nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1)
        self.upsample_final = nn.Upsample(size=(200, 200), mode='bilinear', align_corners=False)

    def forward(self, tensor1, tensor2, tensor3):
        # 调整 tensor3 到与 tensor1 和 tensor2 相同的空间尺寸
        tensor3_resized = self.upsample_to_match(tensor3)
        tensor3_reduced = self.conv1x1_512_to_16(tensor3_resized)
        tensor1 = tensor1.permute(0, 3, 1, 2)
        tensor2 = tensor2.permute(0, 3, 1, 2)
        # tensor3_reduced = tensor3_reduced.permute(0, 3, 1, 2)
        fused = torch.cat([tensor1, tensor2, tensor3_reduced], dim=1)  # Concatenate along the channel dimension
        fused = self.conv_fusion(fused)
        output = self.upsample_final(fused)
        output = output.permute(0, 2, 3, 1) 
        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        # 这些是q,k,v需要的线性层
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False,)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False,)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False,)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # 将输入数据拆分成self.heads个头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        # 乘以查询矩阵并计算得分
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # 使用softmax函数得到注意力概率分数
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


class IFS_Encoder(nn.Module):

    def __init__(self, n):
        super(IFS_Encoder, self).__init__()
        # 假设 n 是已知的，根据 n 调整通道数
        self.conv1 = nn.Conv2d(2 * n, 4 * n, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(4 * n)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * n, 16 * n, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(16 * n)
        self.conv3 = nn.Conv2d(16 * n, 32 * n, kernel_size=3, padding=1, stride=2)  # 尺寸减半
        self.bn3 = nn.BatchNorm2d(32 * n)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((116, 200))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        return x


class OccAdapter(nn.Module):

    def __init__(self):
        super(OccAdapter, self).__init__()
        self.attention_for_cur_and_prev = nn.MultiheadAttention(32, num_heads=1)
        self.attention_for_cur_and_ifs = nn.MultiheadAttention(32, num_heads=1)
        
        # self.conv1x1_for_q_ifs = nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0) # ifs_encoder 里面处理过了
        self.conv1x1_for_k_ifs = nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0)
        self.conv1x1_for_v_ifs = nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0)
        
        self.conv1x1_for_q = nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0)
        self.conv1x1_for_k = nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0)
        self.conv1x1_for_v = nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0)
        # self.fc_for_q = nn.Linear(32, 32)
        # self.fc_for_k = nn.Linear(32, 32)
        # self.fc_for_v = nn.Linear(32, 32)
        # self.bn = nn.BatchNorm1d(32)  # 注意调整维度以匹配全连接层的输出
        
        self.ifs_encoder = IFS_Encoder(1)  # 1就是n [6, 2*n, ..., ...]
        # self.fc_for_qkv_output1 = nn.Linear(725 * 192 * 1024, 1024)
        # self.fc_for_qkv_output2 = nn.Linear(725 * 192 * 1024, 1024)
        self.deconv = UpSampleConv()
        self.deconv_ifs = UpSampleConv()
        self.fusion = FeatureFusionModule()
        
        self.conv3d_1x1 = nn.Conv3d(34, 17, kernel_size=1, stride=1, padding=0)
        
    def preprocess_ifs(self, IFs):
        # [{}, {}, ...]
        # {CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT}
        all_result = []
        for i in range(0, len(IFs)):
            this_time_dict = IFs[i]

            def process_camera_tensors(tensors):
                tensors = [x.permute(0, 3, 1, 2) for x in tensors]
                return torch.cat(tensors, dim=1)

            # 处理每个摄像头的数据
            camera_tensors = []
            camera_keys = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            for key in camera_keys:
                processed_tensors = process_camera_tensors(this_time_dict[key])
                camera_tensors.append(processed_tensors)
            final_tensor = torch.stack(camera_tensors, dim=1) 
            all_result.append(final_tensor)
        res = torch.cat(all_result, dim=2)  # [1, 6, 2, 180, 320]
        return res

    def preprocess_feats(self, mlvl_feats_prev):
        prev_feats = []  # 需要先选出分辨率最大的
        for i in range(0, len(mlvl_feats_prev)):
            cur_feats_list = mlvl_feats_prev[i]
            assert(len(cur_feats_list) == 3)
            prev_feats.append(cur_feats_list[0])
        stacked_tensor = torch.cat(prev_feats, dim=2)  # 第三通道堆叠可能通道数太大了，先放一下，现在列表里只有一个元素
        return stacked_tensor
    
    def forward(self, mlvl_feats, mlvl_feats_prev, IFs):
        # mlvl_feats, mlvl_feats_prev, IFs 可以作为 Wv, Wq, Wk
        #mlvl_feats当前帧resnet特征，IFS帧插
        img_now = mlvl_feats[0]  # 挑选分辨率最大的
        # torch.Size([1, 6, 512, 116, 200]) ([1, 6, 2, 180, 320])
        # ([1, 6, 2, 180, 320])
        # ([1, 6, 2, 180, 320])
        ifs = self.preprocess_ifs(IFs).float()
        prev_feats = self.preprocess_feats(mlvl_feats_prev)
        now_feats = img_now
        ifs_remove_batch = ifs.squeeze(0)
        prev_feats_remove_batch = prev_feats.squeeze(0)  # [6, 512, 116, 200]
        now_feats_remove_batch = now_feats.squeeze(0)  # [6, 512, 116, 200]
        # build q,k,v
        q1 = self.conv1x1_for_q(prev_feats_remove_batch)
        k1 = self.conv1x1_for_k(now_feats_remove_batch)
        v1 = self.conv1x1_for_v(now_feats_remove_batch)
        
        # cut the img
        
        def cut_img(input_tensor):  # [32, 116, 200]
            # 定义patch的大小，例如 4x4
            patch_size = 6
            # 使用unfold将图像特征分割成patches
            patches = input_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
            # patches的形状为 (C, num_patches_w, num_patches_h, patch_size, patch_size)
            # 重新排列patches，调整为形状 (num_patches, C, patch_size, patch_size)  # [1450, 32, 4, 4]
            patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 32, patch_size, patch_size)
            # 对token维度 (C) 进行汇总，例如取平均值
            # 这里我们将token维度降到新的维度C_new
            summarized_patches = patches.mean(axis=(2, 3))
            # mean, avg_pool, max_pool, 1x1conv
            return summarized_patches
        
        def qkv_process(q_or_k_or_v): 
            all_cam = []
            for i in range(6):
                one = cut_img(q_or_k_or_v[i])
                all_cam.append(one.clone())
            all_cam = torch.stack(all_cam).contiguous().view(-1, 32)
            return all_cam
        
        q1 = qkv_process(q1)
        k1 = qkv_process(k1)
        v1 = qkv_process(v1)
        
        # q1 = self.fc_for_q(prev_feats.view(-1, 512))
        # k1 = self.fc_for_k(now_feats.view(-1, 512))
        # v1 = self.fc_for_v(now_feats.view(-1, 512))
        # add bn
        # q1 = self.bn(q1)
        # k1 = self.bn(k1)
        # v1 = self.bn(v1)
        # (L, N, E) L是序列长度，N是批次大小，E是特征维度
        # q1 = q1.view(23200, 6, 64)
        # k1 = k1.view(23200, 6, 64)
        # v1 = v1.view(23200, 6, 64)
        q1 = q1.view(-1, 1, 32)  # 这里我不知道怎么设置好了，因为第一个通道太大根本跑不起来，第二个通道最少必须48
        k1 = k1.view(-1, 1, 32)
        v1 = v1.view(-1, 1, 32)
        # 
        output_qkv1, _ = self.attention_for_cur_and_prev(q1, k1, v1) 
        
        # 处理ifs
        #  6, 2, 180, 320 -> 6, 512, 116, 200
        ifs_encoded = self.ifs_encoder(ifs_remove_batch)  # 
        # 1x1卷积
        # ifs_encoded = self.conv1x1_for_q_ifs(ifs_encoded)  # [6, 32, 116, 200]
        k_ifs = self.conv1x1_for_k_ifs(now_feats_remove_batch)
        v_ifs = self.conv1x1_for_v_ifs(now_feats_remove_batch)
        
        q_ifs = qkv_process(ifs_encoded)
        k_ifs = qkv_process(k_ifs)
        v_ifs = qkv_process(v_ifs)
        
        q_ifs = q_ifs.view(-1, 1, 32)
        k_ifs = k_ifs.view(-1, 1, 32)
        v_ifs = v_ifs.view(-1, 1, 32)
        
        output_qkv2, _ = self.attention_for_cur_and_ifs(q_ifs, k_ifs, v_ifs)  # torch.Size([8700, 1, 32])
        
        output_qkv1 = output_qkv1.mean(axis=0).squeeze()  # (32, )
        output_qkv2 = output_qkv2.mean(axis=0).squeeze()  # (32, )
        
        # 反卷积调整大小
        qkv1 = self.deconv(output_qkv1)  # (17, 200, 200, 16)
        qkv2 = self.deconv_ifs(output_qkv2)  # (17, 200, 200, 16)
        final_qkv = torch.cat((qkv1, qkv2), dim=1)  # (34, 200, 200, 16)
        # output = self.fusion(qkv1, qkv2, now_feats_remove_batch)
        offset = self.conv3d_1x1(final_qkv)  # (17, 200, 200, 16) # aspp
        # https://zhuanlan.zhihu.com/p/524519178
        return offset