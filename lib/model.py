from torch import nn
from torch.nn import functional as F
import torch


class EmbeddedBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EmbeddedBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 64, 3,padding=1)
        self.prelu1 = nn.PReLU(64)
        self.conv2 = nn.Conv2d(64, 64, 3,padding=1)
        self.prelu2 = nn.PReLU(64)
        self.conv3 = nn.Conv2d(64, 64, 3,padding=1)
        self.prelu3 = nn.PReLU(64)
        self.conv4 = nn.Conv2d(64, 64, 3,padding=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        prelu1 = self.prelu1(conv1)
        add1 = x + prelu1

        conv2 = self.conv2(add1)
        prelu2 = self.prelu2(conv2)
        add2 = add1 + prelu2

        conv3 = self.conv3(add2)
        prelu3 = self.prelu3(conv3)
        add3 = add2 + prelu3

        out = self.conv4(add3)
        return out


class PixelShuffle(nn.Module):
    def __init__(self,in_c, factor=2):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(in_c, in_c*factor**2, 3, padding=1)
        self.out = nn.PixelShuffle(factor)

    def forward(self, x):
        conv = self.conv(x)
        out = self.out(conv)
        return out


class Discrimator(nn.Module):
    def __init__(self, in_c):
        super(Discrimator, self).__init__()
        self.block1 = nn.Sequential(
                nn.Conv2d(in_c, 64, 7, padding=3),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.MaxPool2d(2, stride=2)
                )
        self.block2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
                )
        self.block3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
                )
        self.block4 = nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 1, 3, padding=0),
                )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return torch.sigmoid(x)


class SingleDiscriminator(nn.Module):
    def __init__(self, in_c):
        super(SingleDiscriminator, self).__init__()
        self.block1 = nn.Sequential(
                nn.Conv2d(in_c, 64, 7, padding=3),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.MaxPool2d(2, stride=2)
                )
        self.block2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
                )
        self.block3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
                )
        self.block4 = nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 1, padding=0),
                )
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)


class Generator(nn.Module):
    def __init__(self, in_c, out_c):
        super(Generator, self).__init__()
        self.in_conv = nn.Conv2d(in_c, 64, 7, padding=3)
        self.in_prelu = nn.PReLU(64)
        
        self.down1 = nn.Sequential(
                EmbeddedBlock(64, 64),
                EmbeddedBlock(64, 64)
                )
        self.down2 = nn.Sequential(
                EmbeddedBlock(64, 64),
                EmbeddedBlock(64,64)
                )
        self.down3 = nn.Sequential(
                EmbeddedBlock(64, 64),
                EmbeddedBlock(64,64)
                )
        self.bottem_block = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.Conv2d(128, 64, 3, padding=1)
                )
        self.up1 = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                EmbeddedBlock(64, 64),
                EmbeddedBlock(64,64),
                )
        self.up2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                EmbeddedBlock(64, 64),
                EmbeddedBlock(64,64)
                )
        self.up3 = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                EmbeddedBlock(64, 64),
                EmbeddedBlock(64,64)
                )
        self.up_scale = PixelShuffle(64, 2)
        self.conv_block = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.PReLU(64),
                nn.Conv2d(64, out_c, 3, padding=1)
                )
          
    def forward(self, x):
        x = self.in_conv(x)
        x = self.in_prelu(x)
        down1 = self.down1(x)
        pool1 = F.avg_pool2d(down1, 2, stride=2, padding=0)

        down2 = self.down2(pool1)
        pool2 = F.avg_pool2d(down2, 2, stride=2, padding=0)

        down3 = self.down3(pool2)
        pool3 = F.avg_pool2d(down3, 2, stride=2, padding=0)

        bottem = self.bottem_block(pool3)

        concat1 = torch.cat((pool3, bottem), 1)
        up1 = self.up1(concat1)
        up1 = F.interpolate(up1, scale_factor=2)

        concat2 = torch.cat((pool2, up1), 1)
        up2 = self.up2(concat2)
        up2 = F.interpolate(up2, scale_factor=2)

        concat3 = torch.cat((pool1, up2), 1)
        up3 = self.up3(concat3)
        up3 = F.interpolate(up3, scale_factor=2)

        x = self.up_scale(up3)
        x = self.conv_block(x)

        return torch.tanh(x)/2 + 0.5


class HRTo3DHR(nn.Module):
    def __init__(self, in_c, out_c):
        super(HRTo3DHR, self).__init__()
        self.in_conv = nn.Conv2d(in_c, 64, 7, padding=3)
        self.in_prelu = nn.PReLU(64)

        self.down1 = nn.Sequential(
            EmbeddedBlock(64, 64),
            EmbeddedBlock(64, 64)
        )
        self.down2 = nn.Sequential(
            EmbeddedBlock(64, 64),
            EmbeddedBlock(64, 64)
        )
        self.down3 = nn.Sequential(
            EmbeddedBlock(64, 64),
            EmbeddedBlock(64, 64)
        )
        self.bottem_block = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Conv2d(128, 64, 3, padding=1)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            EmbeddedBlock(64, 64),
            EmbeddedBlock(64, 64),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            EmbeddedBlock(64, 64),
            EmbeddedBlock(64, 64)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            EmbeddedBlock(64, 64),
            EmbeddedBlock(64, 64)
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(64),
            nn.Conv2d(64, out_c, 3, padding=1)
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.in_prelu(x)
        down1 = self.down1(x)
        pool1 = F.avg_pool2d(down1, 2, stride=2, padding=0)

        down2 = self.down2(pool1)
        pool2 = F.avg_pool2d(down2, 2, stride=2, padding=0)

        down3 = self.down3(pool2)
        pool3 = F.avg_pool2d(down3, 2, stride=2, padding=0)

        bottem = self.bottem_block(pool3)

        concat1 = torch.cat((pool3, bottem), 1)
        up1 = self.up1(concat1)
        up1 = F.interpolate(up1, scale_factor=2)

        concat2 = torch.cat((pool2, up1), 1)
        up2 = self.up2(concat2)
        up2 = F.interpolate(up2, scale_factor=2)

        concat3 = torch.cat((pool1, up2), 1)
        up3 = self.up3(concat3)
        up3 = F.interpolate(up3, scale_factor=2)

        x = self.conv_block(up3)

        return torch.tanh(x) / 2 + 0.5


class AENet(nn.Module):
    """
    Attention embedding network
    """
    def __init__(self, SR=False, embed_num=2):
        super(AENet, self).__init__()
        self.SR = SR
        self.in_conv = nn.Conv2d(3, 64, 3, padding=1)
        self.in_prelu = nn.ReLU()

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU())
        self.down11 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU())
        self.down22 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU())
        self.down33 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )

        self.bottem_block = nn.Sequential(
            nn.Conv2d(512+embed_num, 512, 3, padding=1),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(1024+embed_num, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(512+embed_num, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(256+embed_num, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU()
        )
        if SR:
            self.up_scale = PixelShuffle(64, 2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x, pad):
        x = self.in_conv(x)
        x = self.in_prelu(x)
        down1 = self.down1(x)
        down1 = self.down11(torch.cat([x, x], dim=1)+down1)
        pool1 = F.avg_pool2d(down1, 2, stride=2, padding=0)

        down2 = self.down2(pool1)
        down2 = self.down22(torch.cat([pool1, pool1], dim=1)+down2)
        pool2 = F.avg_pool2d(down2, 2, stride=2, padding=0)

        down3 = self.down3(pool2)
        down3 = self.down33(torch.cat([pool2, pool2], dim=1)+down3)
        pool3 = F.avg_pool2d(down3, 2, stride=2, padding=0)
        embed_pad = torch.cat([pool3, F.interpolate(pad, scale_factor=1/8)], 1)

        bottem = self.bottem_block(embed_pad)

        concat1 = torch.cat((pool3, bottem, F.interpolate(pad, scale_factor=1/8)), 1)
        up1 = self.up1(concat1)
        up1 = F.interpolate(up1, scale_factor=2)

        concat2 = torch.cat((pool2, up1, F.interpolate(pad, scale_factor=1/4)), 1)
        up2 = self.up2(concat2)
        up2 = F.interpolate(up2, scale_factor=2)

        concat3 = torch.cat((pool1, up2, F.interpolate(pad, scale_factor=1/2)), 1)
        up3 = self.up3(concat3)
        x = F.interpolate(up3, scale_factor=2)
        if self.SR:
            x = self.up_scale(x)

        x = self.conv_block(x)

        return torch.sigmoid(x)


class GeneratorHR(nn.Module):
    def __init__(self, in_c, out_c):
        super(GeneratorHR, self).__init__()
        self.in_conv = nn.Conv2d(in_c, 64, 7, padding=3)
        self.in_prelu = nn.PReLU(64)

        self.down1 = nn.Sequential(
            EmbeddedBlock(64, 64),
            EmbeddedBlock(64, 64)
        )
        self.down2 = nn.Sequential(
            EmbeddedBlock(64, 64),
            EmbeddedBlock(64, 64)
        )
        self.down3 = nn.Sequential(
            EmbeddedBlock(64, 64),
            EmbeddedBlock(64, 64)
        )
        self.bottem_block = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Conv2d(128, 64, 3, padding=1)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            EmbeddedBlock(64, 64),
            EmbeddedBlock(64, 64),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            EmbeddedBlock(64, 64),
            EmbeddedBlock(64, 64)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            EmbeddedBlock(64, 64),
            EmbeddedBlock(64, 64)
        )
        # self.up_scale = PixelShuffle(64, 2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(64),
            nn.Conv2d(64, out_c, 3, padding=1)
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.in_prelu(x)
        down1 = self.down1(x)
        pool1 = F.avg_pool2d(down1, 2, stride=2, padding=0)

        down2 = self.down2(pool1)
        pool2 = F.avg_pool2d(down2, 2, stride=2, padding=0)

        down3 = self.down3(pool2)
        pool3 = F.avg_pool2d(down3, 2, stride=2, padding=0)

        bottem = self.bottem_block(pool3)

        concat1 = torch.cat((pool3, bottem), 1)
        up1 = self.up1(concat1)
        up1 = F.interpolate(up1, scale_factor=2)

        concat2 = torch.cat((pool2, up1), 1)
        up2 = self.up2(concat2)
        up2 = F.interpolate(up2, scale_factor=2)

        concat3 = torch.cat((pool1, up2), 1)
        up3 = self.up3(concat3)
        up3 = F.interpolate(up3, scale_factor=2)

        # x = self.up_scale(up3)
        x = self.conv_block(up3)

        return torch.tanh(x) / 2 + 0.5
