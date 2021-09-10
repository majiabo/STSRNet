from torch import nn
from torch.nn import functional as F
import torch
from lib.model import PixelShuffle
from lib import common


class TransformerV5(nn.Module):
    def __init__(self, in_c, topk=5, locate=None):
        super(TransformerV5, self).__init__()
        self.locate = locate
        self.locations = {}
        self.topk = topk
        self.feature = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1)
        )
        self.texture_conv1 = nn.Conv2d(in_c*(topk-1), in_c, kernel_size=1, padding=0)
        self.texture_conv2 = nn.Conv2d(in_c*3, in_c, kernel_size=1, padding=0)

    def select(self, v_unfold, dim, index):
        views = [v_unfold.shape[0]] + [1 if i!=dim else -1 for i in range(1, len(v_unfold.shape))]
        expanse = list(v_unfold.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(v_unfold, dim, index)

    def top_k(self, r_maxes, r_indexes, fh, fw, qx, qy):
        if r_maxes.shape[0] > 1:
            raise RuntimeError('Only support the situation of batch_size == 1')
        l = qy*fw + qx
        raw_index = r_indexes[0, :, l]
        scores = r_maxes[0, :, l]
        ys = raw_index/fw
        xs = raw_index%fw
        coors = [(x.item(), y.item()) for x, y in zip(xs, ys)]
        scores = [s.item() for s in scores]
        return {'coor': coors, 'score': scores}

    def forward(self, x):
        feature = self.feature(x)
        q = x.clone()
        k = x.clone()

        # [N, c*3*3, H*W]

        q_unfold = F.unfold(nn.ReflectionPad2d(1)(q),  kernel_size=(3, 3), padding=0)
        # [N, H*W, c*3*3]
        k_unfold = F.unfold(nn.ReflectionPad2d(1)(k), kernel_size=(3, 3), padding=0).permute((0, 2, 1))

        q_unfold = F.normalize(q_unfold, dim=1)
        k_unfold = F.normalize(k_unfold, dim=2)

        # [N,H*W, H*W]
        R_qk = torch.bmm(k_unfold, q_unfold)
        # R_max, R_index = torch.max(R_qk, dim=1)
        R_maxs, R_indexs = torch.topk(R_qk, self.topk, dim=1)

        if self.locate is not None:
            _, _, h_, w_ = x.shape
            self.locations = self.top_k(R_maxs, R_indexs, h_, w_, *self.locate)

        Ts = []
        x_unfold = F.unfold(nn.ReflectionPad2d(1)(x.clone()), (3, 3), padding=0)
        # print(x_unfold.shape)
        for i in range(1, self.topk):
            # print(R_maxs.shape)
            R_max, R_index = R_maxs[:, i, :], R_indexs[:, i, :]
            # print(R_index.shape)
            T_unfold = self.select(x_unfold, 2, R_index)

            T = F.fold(T_unfold, output_size=x.size()[-2:], kernel_size=(3, 3), padding=1)/(3.*3)
            S = R_max.view(R_max.shape[0], 1, x.shape[2], x.shape[3])
            Ts.append(T*S)
        texture = self.texture_conv1(torch.cat(Ts, dim=1))
        y = self.texture_conv2(torch.cat([feature, x, texture], dim=1))
        return y


class STSRNet(nn.Module):
    def __init__(self, sr=True, out_num=1, blocks=16, in_c=3, query_c=32, conv=common.default_conv,
                 locate=None):
        super(STSRNet, self).__init__()
        self.sub_mean = common.MeanShift(1.0)
        self.add_mean = common.MeanShiftPlus(1.0, repeat=out_num, sign=1)
        self.head = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=3, padding=1))
        backbone = [common.ResBlock(conv, 64, 3, act=nn.ReLU(True), res_scale=1.0)
              for _ in range(blocks-1)]
        # backbone.append(TransformerV5(32, topk=5, locate=locate))
        backbone.append(TransformerV5(64, topk=5, locate=locate))
        self.backbone = nn.Sequential(*backbone)
        if sr:
            self.tail = nn.Sequential(
                PixelShuffle(64, 2),
                nn.Conv2d(64, out_num*in_c, kernel_size=3, padding=1))
        else:
            self.tail = nn.Conv2d(64, out_num*in_c, kernel_size=3, padding=1)

    def get_locate(self):
        pass

    def load_mode(self, path, tail=True):
        weight = torch.load(path)
        if not tail:
            keys = [i for i in weight.keys() if 'tail' in i]
            for k in keys:
                weight.pop(k)
            self.load_state_dict(weight, strict=False)
        else:
            self.load_state_dict(weight)

    def forward(self, x):
        x = self.sub_mean(x)
        head = self.head(x)
        body = self.backbone(head)
        tail = self.tail(head+body)
        # tail = self.add_mean(tail)
        return tail


class EmbeddedBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EmbeddedBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 64, 3,padding=1)
        self.prelu1 = nn.PReLU(64)
        self.conv2 = nn.Conv2d(64, 64, 3,padding=1)
        self.prelu2 = nn.PReLU(64)
        self.conv3 = nn.Conv2d(64, 64, 3,padding=1)
        self.prelu3 = nn.PReLU(64)
        self.conv4 = nn.Conv2d(64, out_c, 3,padding=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        prelu1 = self.prelu1(conv1)

        conv2 = self.conv2(prelu1)
        prelu2 = self.prelu2(conv2)
        add2 = prelu1 + prelu2

        conv3 = self.conv3(add2)
        prelu3 = self.prelu3(conv3)
        add3 = add2 + prelu3

        out = self.conv4(add3)
        return out


class TransformerV2(nn.Module):
    def __init__(self, in_c, query_c = 32, topk=5, locate=None):
        super(TransformerV2, self).__init__()
        self.locate = locate
        self.locations = {}
        self.topk = topk
        self.feature = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.ReLU()
        )
        self.q_conv1 = nn.Conv2d(in_c, query_c, kernel_size=3, padding=1)
        self.k_conv1 = nn.Conv2d(in_c, query_c, kernel_size=3, padding=1)

    def select(self, v_unfold, dim, index):
        views = [v_unfold.shape[0]] + [1 if i!=dim else -1 for i in range(1, len(v_unfold.shape))]
        expanse = list(v_unfold.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(v_unfold, dim, index)

    def top_k(self, r_maxes, r_indexes, fh, fw, qx, qy):
        if r_maxes.shape[0] > 1:
            raise RuntimeError('Only support the situation of batch_size == 1')
        l = qy*fw + qx
        raw_index = r_indexes[0, :, l]
        scores = r_maxes[0, :, l]
        ys = raw_index/fw
        xs = raw_index%fw
        coors = [(x.item(), y.item()) for x, y in zip(xs, ys)]
        scores = [s.item() for s in scores]
        return {'coor': coors, 'score': scores}

    def forward(self, x):
        feature = self.feature(x)
        q = self.q_conv1(x)
        k = self.k_conv1(x)

        # [N, c*3*3, H*W]
        q_unfold = F.unfold(q,  kernel_size=(3, 3), padding=1)
        # [N, H*W, c*3*3]
        k_unfold = F.unfold(k, kernel_size=(3, 3), padding=1).permute((0, 2, 1))

        q_unfold = F.normalize(q_unfold, dim=1)
        k_unfold = F.normalize(k_unfold, dim=2)

        # [N,H*W, H*W]
        R_qk = torch.bmm(k_unfold, q_unfold)
        # R_max, R_index = torch.max(R_qk, dim=1)
        R_maxs, R_indexs = torch.topk(R_qk, self.topk, dim=1)

        if self.locate is not None:
            # temp, remeber to delete following three lines
            # R_qk = torch.bmm(q_unfold.permute((0, 2, 1)), q_unfold)
            # R_max, R_index = torch.max(R_qk, dim=1)
            # R_maxs, R_indexs = torch.topk(R_qk, self.topk, dim=1)
            # -------------------end----------------------
            _, _, h_, w_ = x.shape
            self.locations = self.top_k(R_maxs, R_indexs, h_, w_, *self.locate)
        # temp, remeber ttto delete following three lines
        # R_qk = torch.bmm(k_unfold, q_unfold)
        # R_max, R_index = torch.max(R_qk, dim=1)
        # R_maxs, R_indexs = torch.topk(R_qk, self.topk, dim=1)
        # -----------------------end ------

        Ts = []
        x_unfold = F.unfold(x, (3, 3), padding=1)
        # print(x_unfold.shape)
        for i in range(1, self.topk):
            # print(R_maxs.shape)
            R_max, R_index = R_maxs[:, i, :], R_indexs[:, i, :]
            # print(R_index.shape)
            T_unfold = self.select(x_unfold, 2, R_index)

            T = F.fold(T_unfold, output_size=x.size()[-2:], kernel_size=(3, 3), padding=1)/(3.*3)
            S = R_max.view(R_max.shape[0], 1, x.shape[2], x.shape[3])
            Ts.append(T*S)
        y = feature + x + sum(Ts)/len(Ts)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, in_c, query_c = 32):
        super(TransformerBlock, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.ReLU()
        )
        self.v_conv1 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1)
        self.v_conv2 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1)

        self.q_conv1 = nn.Conv2d(in_c, query_c, kernel_size=3, padding=1)
        self.k_conv1 = nn.Conv2d(in_c, query_c, kernel_size=3, padding=1)

    def select(self, v_unfold, dim, index):
        views = [v_unfold.shape[0]] + [1 if i!=dim else -1 for i in range(1, len(v_unfold.shape))]
        expanse = list(v_unfold.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(v_unfold, dim, index)

    def forward(self, x):
        v = self.v_conv1(x)
        v = self.v_conv2(v+x)

        q = self.q_conv1(x)
        k = self.k_conv1(x)

        q_unfold = F.unfold(q,  kernel_size=(3, 3), padding=1)
        k_unfold = F.unfold(k, kernel_size=(3, 3), padding=1).permute((0, 2, 1))

        q_unfold = F.normalize(q_unfold, dim=1)
        k_unfold = F.normalize(k_unfold, dim=2)

        R_qk = torch.bmm(k_unfold, q_unfold)
        R_max, R_index = torch.max(R_qk, dim=1)

        v_unfold = F.unfold(v, kernel_size=(3, 3), padding=1)
        T_unfold = self.select(v_unfold, 2, R_index)

        T = F.fold(T_unfold, output_size=v.size()[-2:], kernel_size=(3, 3), padding=1)/(3.*3)
        S = R_max.view(R_max.shape[0], 1, v.shape[2], v.shape[3])
        feature = self.feature(x)
        return feature+x+T*S


class STTSRNet(nn.Module):
    """
    Attention embedding network
    """
    def __init__(self, sr=False, mask_num=5, in_c=3):
        super(STTSRNet, self).__init__()
        self.SR = sr
        self.mask_num = mask_num
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_c, 64, 3, stride=1, padding=1),
            nn.PReLU()
        )
        self.down1 = TransformerBlock(64)
        self.down2 = TransformerBlock(64)
        self.down3 = TransformerBlock(64)
        self.bottom = nn.Conv2d(64, 64, 3, padding=1)
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            TransformerBlock(64)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            TransformerBlock(64)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            TransformerBlock(64)
        )

        self.out = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_c*mask_num, 3, padding=1)
        )
        if sr:
            self.up_scale = PixelShuffle(64, 2)

    def forward(self, x):
        in_conv = self.in_conv(x)
        down1 = self.down1(in_conv)
        pool1 = F.avg_pool2d(down1, 2, stride=2, padding=0)
        down2 = self.down2(pool1)
        pool2 = F.avg_pool2d(down2, 2, stride=2, padding=0)
        down3 = self.down3(pool2)
        pool3 = F.avg_pool2d(down3, 2, stride=2, padding=0)
        bottom = self.bottom(pool3)

        fusion1 = torch.cat([pool3, bottom], dim=1)
        up1 = self.up1(fusion1)
        inter1 = F.interpolate(up1, scale_factor=2)

        fusion2 = torch.cat([pool2, inter1], dim=1)
        up2 = self.up2(fusion2)
        inter2 = F.interpolate(up2, scale_factor=2)

        fusion3 = torch.cat([pool1, inter2], dim=1)
        up3 = self.up3(fusion3)
        inter3 = F.interpolate(up3, scale_factor=2)

        if self.SR:
            inter3 = self.up_scale(inter3)
        img = self.out(inter3)
        return img


if __name__ == '__main__':
    import os
    import time
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # model = AENet(sr=True)
    # fusion_module = FusionModule(160, 5, 32)
    # t = TransformerBlock(5)
    # x = torch.randn((12, 160, 64, 64))
    x = torch.randn((1, 3, 128, 128)).cuda()
    # model = STTSRNet(True, 5)
    start = time.time()
    # model = STTSRNet1(sr=True, out_num=1).cuda()
    # model = STTRNet2(True, out_num=5).cuda()
    # model = TransformerV2(3, 16, locate=(60, 60)).cuda()
    model = TransformerV5(3, 5).cuda()
    end = time.time()
    y = model(x)
    print(model.locations)
    # y = t(y)
    print(y.shape)
    print('Time:', end-start)
    # img = model(x)
