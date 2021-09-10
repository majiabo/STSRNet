#
import sys
sys.path.append('../')
from lib.dataset import wrap_multi_channel_img, AttentionDataSet
import os
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from SRMethods.net import EDSR
import cv2


class Args:
    n_resblocks = 16
    n_feats = 64
    scale = [2]
    rgb_range = 1
    n_colors = 3
    res_scale = 1


class MyArgs:
    lr = 1e-4
    decay_factor = 0.5
    decay_step = 20
    start_epoch = 0
    end_epoch = 60

    batch_size = 16
    test_patch = 2000
    num_workers = 2
    random_seed = 119
    out_size = 512
    resolution = 0.25
    in_img = [0]
    out_img = [0]
    lr2hr = True
    pattern = 'random'

    title = 'ablation_sr/EDSR'
    # data_root = '/home/kara/Data_temp'
    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    # prefix = '/home/kara/diskarray_mjb/Projects'
    prefix = '/mnt/diskarray/mjb/Projects'
    tensorboard_log = '{}/3DSR/log/tensorboards/{}'.format(prefix, title)
    checkpoints_path = '{}/3DSR/log/checkpoints/{}'.format(prefix, title)
    img_log_path = '{}/3DSR/log/imgs/train/{}'.format(prefix, title)
    test_img_log_path = '{}/3DSR/log/imgs/test/{}'.format(prefix, title)
    weight_id = [f'netG_epoch_2_{i}' for i in range(60, 61)]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda')

    arg = MyArgs()
    max_num = None

    G = EDSR(Args(), output_num=1).to(device)

    weight_path = [os.path.join(arg.checkpoints_path, line+'.pth') for line in arg.weight_id]
    G.eval()
    with torch.no_grad():
        for weight, weight_ in zip(weight_path, arg.weight_id):
            train_set = AttentionDataSet(arg.data_root, 'test', 'random', arg.in_img, arg.out_img,
                                        lr2hr=arg.lr2hr, seed=arg.random_seed, input_size=arg.out_size,
                                        resolution=arg.resolution)

            train_set.data_prefix = '/mnt/diskarray/mjb/Projects'  # dataset = RealMobileDataset(img_root, test_txt, True)
            train_loader = DataLoader(train_set, batch_size=arg.batch_size, shuffle=False, num_workers=0)

            G.load_state_dict(torch.load(weight))

            img_save_counter = 0
            img_log_sub_path = os.path.join(arg.test_img_log_path, weight_)
            if not os.path.exists(img_log_sub_path):
                os.makedirs(img_log_sub_path)
            else:
                print('{} exist, skip...'.format(img_log_sub_path))
            for (lr_img, hr_img, _) in tqdm(train_loader):

                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)
                gen = G(lr_img)
                for i in range(lr_img.shape[0]):
                    tensor_list = [gen[i, :, :, :], hr_img[i, :, :, :]]

                    wraped = wrap_multi_channel_img(tensor_list)

                    img_path = os.path.join(img_log_sub_path, '{}.png'.format(img_save_counter))
                    cv2.imwrite(img_path, wraped)
                    img_save_counter += 1
                if max_num is not None:
                    if img_save_counter > max_num:
                        break

