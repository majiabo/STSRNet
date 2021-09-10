from lib import eval
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

# title = 'stitch_data_check'
title = 'ablation_topk/t4_v4'
disk_prefix = 'diskarray/mjb/Projects'
img_size = 128
sample = None
target = []
img_num_per_line = 5
ssim_psnr_flag = True
mse_flag = True
mae_flag = True
lpips_flag = True

data_root = '/mnt/{}/3DSR/log/imgs/test/{}'.format(disk_prefix, title)
# data_root = '/mnt/diskarray/mjb/Projects/3DSR/data/stitch_data'
# fft_root = '/mnt/{}/3DSR/log/imgs/fft_img/{}'.format(disk_prefix, title)
# layer_map_root = '/mnt/{}/3DSR/log/imgs/layer_map/{}'.format(disk_prefix, title)
ssim_psnr_root = '/mnt/{}/3DSR/log/ssim_psnr'.format(disk_prefix)
# ssim_psnr_root = '/mnt/diskarray/mjb/Projects/3DSR/data'

if not os.path.exists(ssim_psnr_root):
    os.makedirs(ssim_psnr_root)
logger_path = os.path.join(ssim_psnr_root, '{}.txt'.format(title))

weights = os.listdir(data_root)
if not os.path.exists(os.path.split(logger_path)[0]):
    os.makedirs(os.path.split(logger_path)[0])
file_handle = open(logger_path, 'a')
for weight in weights:
    if target:
        if weight not in target:
            continue
    print('processing:', weight)
    data_set = eval.EvalDataSet(os.path.join(data_root, weight))
    # data_set = eval.EvalDataSetV1(os.path.join(data_root, weight), [f'{i}' for i in range(-2, 3)],
    #                              size=img_size, num_of_line=img_num_per_line)
    data_set.size = img_size
    print('Data size:', data_set.size)
    data_set.num_of_line = img_num_per_line

    # fft_path = os.path.join(fft_root, weight)
    # layer_map_path = os.path.join(layer_map_root, weight)
    fft_path = None
    layer_map_path = None
    evaluator = eval.Evaluator(data_set, ssim_psnr_flag, fft_path, layer_map_path, tf_backend=True, mse=mse_flag,
                               mae=mae_flag, lpips_flag=lpips_flag, sample_num=sample)
    avg_ssim, avg_psnr, avg_mse, avg_mae, avg_lpips = evaluator.eval()
    file_handle.write('weight:{}]\n'.format(weight))
    file_handle.write('avg_ssim:{}\n'.format(avg_ssim))
    file_handle.write('avg_psnr:{}\n'.format(avg_psnr))
    file_handle.write('avg_mse:{}\n'.format(avg_mse))
    file_handle.write('avg_mae:{}\n'.format(avg_mae))
    file_handle.write('avg_lpips:{}\n\n\n'.format(avg_lpips))
    file_handle.flush()

file_handle.close()
