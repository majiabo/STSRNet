
class MinusOne:
    """
    real SISR
    """
    gpu_id = '0'
    random_seed = 119
    title = 'ablation_input/minus_one_v1'

    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)

    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # for pretrain_model
    # D_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/AE3DHR/D_2_212.pth'
    D_path, G_path = None, None
    G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/ablation_input/minus_one/G_78_6214.pth'
    strict_mode = True
    # =================================
    #       model config
    # =================================
    lr2hr = True
    SR = True
    out_size = 512
    top_k = 5
    resolution = 0.25
    start_epoch = 0
    stop_epoch = 80
    lr = 1e-4
    lr_adjust_step = 30
    decay_factor = 0.5
    batch_size = 32
    test_patch = 2000
    num_workers = 4
    in_img = [-1]  # 输入图像的通道数
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
    weight_id = ['G_77_6135', 'G_78_6214', 'G_79_6293']


class One:
    """
    real SISR
    """
    gpu_id = '1'
    random_seed = 119
    title = 'ablation_input/one_v1'

    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)

    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # for pretrain_model
    # D_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/AE3DHR/D_2_212.pth'
    D_path, G_path = None, None
    G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/ablation_input/one/G_79_6293.pth'
    strict_mode = True
    # =================================
    #       model config
    # =================================
    lr2hr = True
    SR = True
    out_size = 512
    top_k = 5
    resolution = 0.25
    start_epoch = 0
    stop_epoch = 80
    lr = 1e-4
    lr_adjust_step = 30
    decay_factor = 0.5
    batch_size = 48
    test_patch = 2000
    num_workers = 4
    in_img = [1]  # 输入图像的通道数
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
    weight_id = ['G_79_6293']


if __name__ == '__main__':
    print()