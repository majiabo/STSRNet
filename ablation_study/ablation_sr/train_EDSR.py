import sys
# sys.path.append('/home/kara/diskarray_mjb/Projects/3DSR/code')

from lib.dataset import AttentionDataSet, path_checker, wrap_multi_channel_img
from torch.utils.tensorboard import SummaryWriter
from lib.eval import Evaluator
import os
import cv2
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from SRMethods.loss import EDSRLoss
from SRMethods.net import EDSR
import numpy as np

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

    pretrain_ckpt = None


if __name__ == '__main__':
    # ================================
    #  config path
    # ================================
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda')

    args = MyArgs()

    path_checker(args.checkpoints_path)
    path_checker(args.img_log_path)
    summary_writer = SummaryWriter(args.tensorboard_log)

    train_set = AttentionDataSet(args.data_root, 'train', args.pattern, args.in_img, args.out_img,
                                 lr2hr=args.lr2hr, seed=args.random_seed, input_size=args.out_size,
                                 resolution=args.resolution)
    test_set = AttentionDataSet(args.data_root, 'test', args.pattern, args.in_img, args.out_img,
                                lr2hr=args.lr2hr, seed=args.random_seed, input_size=args.out_size,
                                resolution=args.resolution)
    train_set.data_prefix = args.prefix
    test_set.data_prefix = args.prefix
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # define evaluator
    evaluator = Evaluator(test_set, ssim_psnr=True, tf_backend=True)

    netG = EDSR(Args(), output_num=len(args.out_img))
    if args.pretrain_ckpt:
        netG.load_state_dict(torch.load(args.pretrain_ckpt))
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

    generator_criterion = EDSRLoss()

    if torch.cuda.is_available():
        netG.cuda()

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr)
    global_counter = 0

    for epoch in range(args.start_epoch, args.end_epoch + 1):
        netG.train()
        for index, (lr_img, hr_img, _) in enumerate(train_loader):
            print('Epoch: [{}/{}] | Batch: [{}/{}]'.format(epoch, args.end_epoch+1, index, len(train_loader)))
            real_img = hr_img.cuda()
            z = lr_img.cuda()
            netG.zero_grad()
            fake_img = netG(z)
            g_loss = generator_criterion(fake_img, real_img)
            g_loss.backward()
            optimizerG.step()
            
            global_counter += 1

            if global_counter % 100 == 0:
                summary_writer.add_scalar('ablation_sr/g_loss', g_loss, global_counter)
                tensor_list = [fake_img[0], hr_img[0]]
                wraped = wrap_multi_channel_img(tensor_list)
                img_path = os.path.join(args.img_log_path, 'Img_{}.jpg'.format(global_counter))
                cv2.imwrite(img_path, wraped)

        # eval
        converter = lambda data: [np.clip(i.cpu().numpy(), a_min=0, a_max=1) * 255 for i in data.permute((0, 2, 3, 1))]
        netG.eval()
        with torch.no_grad():
            print('Evaluating ...')
            psnrs = []
            ssims = []
            for index, (lr_img, hr_img, blur_maps) in enumerate(test_loader):

                if index * args.batch_size > args.test_patch:
                    break
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)
                gen = netG(lr_img)
                ssim, psnr = evaluator.ssim_psnr(converter(gen), converter(hr_img))
                psnrs += psnr
                ssims += ssim
            summary_writer.add_scalar('ablation_sr/test_psnr', sum(psnrs) / len(psnrs), epoch)
            summary_writer.add_scalar('ablation_sr/test_ssim', sum(ssims) / len(ssims), epoch)

        # save model parameters
        torch.save(netG.state_dict(), os.path.join(args.checkpoints_path, 'netG_epoch_%d_%d.pth' % (2, epoch)))

        if epoch % args.decay_step == 0 and epoch != 0:
            args.lr = args.lr * args.decay_factor
            for p in optimizerG.param_groups:
                p['lr'] = args.lr

