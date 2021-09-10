
class AblationTop2:
    """
    real SISR
    """
    gpu_id = '0'
    random_seed = 119
    title = 'ablation_topk/top2-V4'

    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)

    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # for pretrain_model
    # D_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/AE3DHR/D_2_212.pth'
    D_path, G_path = None, None
    G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/ablation_topk/top2-V3/G_57_4555.pth'
    strict_mode = True
    # =================================
    #       model config
    # =================================
    lr2hr = True
    SR = True
    out_size = 512
    top_k = 2
    resolution = 0.25
    start_epoch = 0
    stop_epoch = 60
    lr = 1e-4
    lr_adjust_step = 20
    decay_factor = 0.2
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
    weight_id = [i.strip('.pth') for i in ['G_56_4476.pth', 'G_57_4555.pth', 'G_58_4634.pth', 'G_59_4713.pth']]


class AblationT3:
    """
    real SISR
    """
    gpu_id = '0'
    random_seed = 119
    title = 'ablation_topk/t3_v4'

    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)

    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # for pretrain_model
    # D_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/AE3DHR/D_2_212.pth'
    D_path, G_path = None, None
    G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/ablation_topk/t3_v3/G_59_4713.pth'
    strict_mode = True
    # =================================
    #       model config
    # =================================
    lr2hr = True
    SR = True
    out_size = 512
    top_k = [3]
    resolution = 0.25
    start_epoch = 0
    stop_epoch = 60
    lr = 1e-4
    lr_adjust_step = 20
    decay_factor = 0.2
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
    weight_id = [i.strip('.pth') for i in ['G_59_4713.pth']]


class AblationT4:
    """
    real SISR
    """
    gpu_id = '1'
    random_seed = 119
    title = 'ablation_topk/t4_v4'

    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)

    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # for pretrain_model
    # D_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/AE3DHR/D_2_212.pth'
    D_path, G_path = None, None
    G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/ablation_topk/t4_v3/G_59_4713.pth'
    strict_mode = True
    # =================================
    #       model config
    # =================================
    lr2hr = True
    SR = True
    out_size = 512
    top_k = [4]
    resolution = 0.25
    start_epoch = 0
    stop_epoch = 60
    lr = 1e-4
    lr_adjust_step = 20
    decay_factor = 0.2
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
    weight_id = [i.strip('.pth') for i in ['G_59_4713.pth']]


class AblationT5:
    """
    real SISR
    """
    gpu_id = '0'
    random_seed = 119
    title = 'ablation_topk/t5_v3'

    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)

    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # for pretrain_model
    # D_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/AE3DHR/D_2_212.pth'
    D_path, G_path = None, None
    G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/ablation_topk/t5_v2/G_59_4713.pth'
    strict_mode = True
    # =================================
    #       model config
    # =================================
    lr2hr = True
    SR = True
    out_size = 512
    top_k = [5]
    resolution = 0.25
    start_epoch = 0
    stop_epoch = 60
    lr = 1e-4
    lr_adjust_step = 20
    decay_factor = 0.2
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
    weight_id = [i.strip('.pth') for i in ['G_59_4713.pth']]


class AblationTop3(AblationTop2):
    top_k = 3
    gpu_id = '0'
    title = 'ablation_topk/top3-v4'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)
    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/ablation_topk/top3-v3/G_58_4634.pth'
    decay_factor = 0.25
    lr_adjust_step = 20
    start_epoch = 0
    test_img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    weight_id = [i.strip('.pth') for i in ['G_56_4476.pth', 'G_57_4555.pth', 'G_58_4634.pth', 'G_59_4713.pth']]


class AblationTop4(AblationTop2):
    top_k = 4
    gpu_id = '1'
    title = 'ablation_topk/top4-V4'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)
    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/ablation_topk/top4-V3/G_58_4634.pth'
    decay_factor = 0.25
    lr_adjust_step = 20
    start_epoch = 0
    test_img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    weight_id = [i.strip('.pth') for i in ['G_56_4476.pth', 'G_57_4555.pth', 'G_58_4634.pth', 'G_59_4713.pth']]


if __name__ == '__main__':
    arg3 = AblationTop3()
    arg2 = AblationTop2()
    print()