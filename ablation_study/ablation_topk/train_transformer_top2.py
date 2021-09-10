"""
author:majiabo
time:2019.2.8
function : 3dSR
"""

import sys

sys.path.append('../')
from attention_strategy.ablation_topk.configs import AblationTop2
from attention_strategy.ablation_topk.net import STTRNetAblationStudy
from lib.dataset import AttentionDataSet
from lib.dataset import wrap_multi_channel_img
from lib.eval import Evaluator
from lib.utilis import path_checker
from torch.utils.data import DataLoader
import tensorflow as tf

import os
import cv2
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


args = AblationTop2()
torch.manual_seed(args.random_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device_num = len(args.gpu_id.split(','))
device = torch.device('cuda')
_gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
[tf.config.experimental.set_memory_growth(_gpus[_], True) for _ in range(device_num)]

batch_size = args.batch_size * device_num
num_workers = device_num * args.num_workers
img_save_step = int(200 / batch_size)
model_save_step = int(10000 / batch_size)
lr = args.lr*device_num

path_checker(args.checkpoints_path)
path_checker(args.img_log_path)

Writer = SummaryWriter(args.tensorboard_path)
train_set = AttentionDataSet(args.data_root, 'train', args.pattern, args.in_img, args.out_img,
                             lr2hr=args.lr2hr, seed=args.random_seed, input_size=args.out_size, resolution=args.resolution)
test_set = AttentionDataSet(args.data_root, 'test', args.pattern, args.in_img, args.out_img,
                            lr2hr=args.lr2hr, seed=args.random_seed, input_size=args.out_size, resolution=args.resolution)
train_set.data_prefix = '/mnt/diskarray/mjb/Projects'
test_set.data_prefix = '/mnt/diskarray/mjb/Projects'
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# define evaluator
evaluator = Evaluator(test_set, ssim_psnr=True, tf_backend=True)

G = STTRNetAblationStudy(sr=True, out_num=len(args.out_img), top_k=args.top_k).cuda()
G.train()

if args.G_path:
    G.load_state_dict(torch.load(args.G_path), strict=args.strict_mode)
    # G.load_mode(args.G_path, tail=False)
optimizerG = torch.optim.Adam(G.parameters(), lr=lr)

if device_num>1:
    G = torch.nn.DataParallel(G).to(device)
else:
    G = G.to(device)

l1_loss = torch.nn.L1Loss()
img_save_counter = 0

for epoch in range(args.start_epoch, args.stop_epoch):
    for index, (lr_img, hr_img, blur_maps) in enumerate(train_loader):
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        gen = G(lr_img)

        G.zero_grad()
        gloss = l1_loss(gen, hr_img)
        gloss.backward()
        optimizerG.step()

        # process log
        if (index) % img_save_step == 0:
            tensor_list = [gen[0, :, :, :], hr_img[0, :, :, :]]

            wraped = wrap_multi_channel_img(tensor_list)
            Writer.add_scalar('scalar/train', gloss, img_save_counter)

            img_path = os.path.join(args.img_log_path, 'Img_{}_{}.jpg'.format(epoch, img_save_counter))
            mask_img_path = os.path.join(args.img_log_path, 'Mask_{}_{}.jpg'.format(epoch, img_save_counter))

            cv2.imwrite(img_path, wraped)
            img_save_counter += 1

        if (index + 1) % model_save_step == 0 and index != 0:
            gg_path = os.path.join(args.checkpoints_path, 'G_{}_{}.pth'.format(epoch, img_save_counter))
            if device_num>1:
                torch.save(G.module.state_dict(), gg_path)
            else:
                torch.save(G.state_dict(), gg_path)
        sys.stdout.write(
            "\r[Epoch {}/{}] [Batch {}/{}] [mae:{:.5f}]".format(
                epoch, args.stop_epoch, index, len(train_loader), gloss.item()))
        sys.stdout.flush()
    print()
    converter = lambda data: [np.clip(i.cpu().numpy(), a_min=0, a_max=1)*255 for i in data.permute((0, 2, 3, 1))]
    with torch.no_grad():
        print('Evaluating ...')
        psnrs = []
        ssims = []
        for index, (lr_img, hr_img, blur_maps) in enumerate(test_loader):

            if index*batch_size > args.test_patch:
                break
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            gen = G(lr_img)
            ssim, psnr = evaluator.ssim_psnr(converter(gen), converter(hr_img))
            psnrs += psnr
            ssims += ssim
        Writer.add_scalar('scalar/test_psnr', sum(psnrs)/len(psnrs), img_save_counter)
        Writer.add_scalar('scalar/test_ssim', sum(ssims)/len(ssims), img_save_counter)

    if epoch % args.lr_adjust_step == 0 and epoch != 0:
        lr = lr * args.decay_factor
        for param in optimizerG.param_groups:
            param['lr'] = lr
        print('learning rate is :', lr)
