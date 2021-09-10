"""
author:majiabo
time:2019.2.8
function : 3dSR predict
"""

import sys
sys.path.append('../')
from attention_strategy.ablation_sr.net import STTRNetAblationStudy
from lib.dataset import wrap_multi_channel_img, AttentionDataSet
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import cv2
import torch
from attention_strategy.ablation_sr.configs import AblationSR

args = AblationSR()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda')
# ================================
#  config path
# ================================
max_num = None

weight_path = [os.path.join(args.checkpoints_path, line+'.pth') for line in args.weight_id]

G = STTRNetAblationStudy(sr=True, out_num=len(args.out_img), top_k=args.top_k).cuda()

G.eval()
batch_size = 4
with torch.no_grad():
    for weight, weight_ in zip(weight_path, args.weight_id):
        train_set = AttentionDataSet(args.data_root, 'test', args.pattern, input_layer=args.in_img, out_layer=args.out_img,
                              seed=args.random_seed, lr2hr=args.lr2hr, input_size=args.out_size, resolution=args.resolution)
        train_set.data_prefix = '/mnt/diskarray/mjb/Projects'

        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=False)
        print('Wight:', weight)
        G.load_state_dict(torch.load(weight))

        img_save_counter = 0
        img_log_sub_path = os.path.join(args.test_img_log_path, weight_)
        if not os.path.exists(img_log_sub_path):
            os.makedirs(img_log_sub_path)
        else:
            print('{} exist, skip...'.format(img_log_sub_path))
        for (lr_img, hr_img, blur_maps) in tqdm(train_loader):

            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            gen = G(lr_img)
            # gen = G(lr_img)
            for i in range(lr_img.shape[0]):
                tensor_list = [gen[i, :, :, :], hr_img[i, :, :, :]]

                wraped = wrap_multi_channel_img(tensor_list)

                img_path = os.path.join(img_log_sub_path, '{}.png'.format(img_save_counter))
                cv2.imwrite(img_path, wraped)
                img_save_counter += 1
            if max_num is not None:
                if img_save_counter > max_num:
                    break