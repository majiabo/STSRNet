from torch import nn
from torch.nn import functional as F
import torch
from lib.model import PixelShuffle
from SRMethods import common


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


class TransformerV5T(nn.Module):
    """
    modified for T3, T4, T5
    """
    def __init__(self, in_c, topk=None, locate=None):
        """

        :param in_c:
        :param topk:  list , e.g. [2], [3], [2, 3]
        :param locate:
        """
        super(TransformerV5T, self).__init__()
        self.locate = locate
        self.locations = {}
        self.topk = topk
        self.feature = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1)
        )
        self.texture_conv1 = nn.Conv2d(in_c*(len(topk)), in_c, kernel_size=1, padding=0)
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
        R_maxs, R_indexs = torch.topk(R_qk, max(self.topk), dim=1)
        topk = [i-1 for i in self.topk]
        R_maxs, R_indexs = R_maxs[:, topk, ], R_indexs[:, topk]
        if self.locate is not None:
            _, _, h_, w_ = x.shape
            self.locations = self.top_k(R_maxs, R_indexs, h_, w_, *self.locate)

        Ts = []
        x_unfold = F.unfold(nn.ReflectionPad2d(1)(x.clone()), (3, 3), padding=0)
        # print(x_unfold.shape)
        for i in range(0, len(self.topk)):
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


class STTRNetAblationStudy(nn.Module):
    def __init__(self, sr=True, out_num=1, blocks=16, in_c=3, top_k=5, conv=common.default_conv,
                 locate=None):

        super(STTRNetAblationStudy, self).__init__()
        self.sub_mean = common.MeanShift(1.0)
        self.add_mean = common.MeanShiftPlus(1.0, repeat=out_num, sign=1)
        self.head = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=3, padding=1))
        backbone = [common.ResBlock(conv, 64, 3, act=nn.ReLU(True), res_scale=1.0)
              for _ in range(blocks-1)]
        backbone.append(TransformerV5(64, topk=top_k, locate=locate))
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
            print(weight.keys())
            keys = [i for i in weight.keys() if 'tail' in i or 'backbone.15' in i]
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


class STTRNetAblationStudyT(nn.Module):
    """
    For T3, T4, T5
    """
    def __init__(self, sr=True, out_num=1, blocks=16, in_c=3, top_k=5, conv=common.default_conv,
                 locate=None):

        super(STTRNetAblationStudyT, self).__init__()
        self.sub_mean = common.MeanShift(1.0)
        self.add_mean = common.MeanShiftPlus(1.0, repeat=out_num, sign=1)
        self.head = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=3, padding=1))
        backbone = [common.ResBlock(conv, 64, 3, act=nn.ReLU(True), res_scale=1.0)
              for _ in range(blocks-1)]
        backbone.append(TransformerV5T(64, topk=top_k, locate=locate))
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
            print(weight.keys())
            keys = [i for i in weight.keys() if 'tail' in i or 'backbone.15' in i]
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
    model = TransformerV5T(3, [4]).cuda()
    end = time.time()
    y = model(x)
    print(model.locations)
    # y = t(y)
    print(y.shape)
    print('Time:', end-start)
    # img = model(x)
