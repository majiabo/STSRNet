
class Ablation1by1:
    """
    real SISR
    """
    gpu_id = '0'
    random_seed = 119
    title = 'ablation_ws/1by1_v2'

    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)

    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # for pretrain_model
    # D_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/AE3DHR/D_2_212.pth'
    D_path, G_path = None, None
    # G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/ablation_ws/1by1/G_89_7083.pth'
    strict_mode = True
    # =================================
    #       model config
    # =================================
    lr2hr = True
    SR = True
    out_size = 512
    top_k = 5
    window_size = 1
    resolution = 0.25
    start_epoch = 0
    stop_epoch = 200
    lr = 1e-4
    lr_adjust_step = 60
    decay_factor = 0.5
    batch_size = 32
    test_patch = 2000
    num_workers = 4
    in_img = [0]  # 输入图像的通道数
    pattern = 'random'
    out_img = [-2, -1, 0, 1, 2]
    if pattern == 'random':
        in_c = 3
    else:
        in_c = len(in_img) * 3
    out_c = len(out_img) * 3
    d_in_c = out_c

    # for predict
    test_img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # weight_id = [i.strip('.pth') for i in ['G_56_4476.pth', 'G_57_4555.pth', 'G_58_4634.pth', 'G_59_4713.pth']]
    weight_id = ['G_199', 'G_198', 'G_197']


class Ablation5by5(Ablation1by1):
    top_k = 5
    window_size = 5
    gpu_id = '1'
    title = 'ablation_wx/5by5_v2'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)
    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/ablation_wx/5by5/G_70_5583.pth'
    G_path = None
    lr = 1e-4
    decay_factor = 0.5
    lr_adjust_step = 60
    batch_size = 16
    start_epoch = 0
    stop_epoch = 200
    test_img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    weight_id = ['G_199', 'G_198', 'G_197']


if __name__ == '__main__':
    arg3 = Ablation1by1()
    arg2 = Ablation5by5()
    print()