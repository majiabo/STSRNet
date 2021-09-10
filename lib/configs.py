class STSRNetSISRConfig:
    """
    real SISR
    """
    gpu_id = '0,1'
    random_seed = 119
    title = 'test_p'

    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)

    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # for pretrain_model
    # D_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/AE3DHR/D_2_212.pth'
    D_path, G_path = None, None
    G_path = '../assets/G_57_4555.pth'  # load pretrain weight to help train
    strict_mode = True
    # =================================
    #       model config
    # =================================
    lr2hr = True
    SR = True
    out_size = 512
    resolution = 0.25
    start_epoch = 0
    stop_epoch = 100  # adjust to make sure the model converge
    lr = 1e-4
    lr_adjust_step = 15  # adjust based on start_epoch and stop_epoch, or use lr
    decay_factor = 0.5  # similarly
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
    weight_id = [i.strip('.pth') for i in [
         # 'G_43_3449.pth',
         # 'G_42_3370.pth',
         # 'G_31_2501.pth',
         # 'G_36_2896.pth',
         # 'G_59_4713.pth',
         # 'G_34_2738.pth',
         # 'G_40_3212.pth',
         # 'G_45_3607.pth',
         # 'G_54_4318.pth',
         # 'G_53_4239.pth',
         # 'G_33_2659.pth',
         # 'G_48_3844.pth',
         'G_57_4555.pth',
         # 'G_58_4634.pth',
         # 'G_37_2975.pth',
         # 'G_56_4476.pth',
         # 'G_44_3528.pth',
         # 'G_49_3923.pth',
         # # 'G_46_3686.pth',
         # 'G_50_4002.pth',
         # 'G_51_4081.pth',
         # 'G_47_3765.pth',
         # 'G_55_4397.pth',
         # 'G_39_3133.pth',
         # 'G_52_4160.pth',
         # 'G_38_3054.pth',
         # 'G_32_2580.pth',
         # 'G_35_2817.pth',
         # 'G_41_3291.pth'
    ]
            ]


class STSRNetConfig:
    gpu_id = '1'
    random_seed = 119
    title = 'our/STTRNet1_SISR_rec'

    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)

    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # for pretrain_model
    # D_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/AE3DHR/D_2_212.pth'
    D_path, G_path = None, None
    #G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/AENet_V4/G_99_7575.pth'
    strict_mode = True
    # =================================
    #       model config
    # =================================
    lr2hr = True
    SR = True
    out_size = 256
    start_epoch = 0
    stop_epoch = 40
    lr = 1e-4
    lr_adjust_step = 10
    decay_factor = 0.6
    batch_size = 2
    num_workers = 0
    in_img = [0]  # 输入图像的通道数
    pattern = 'random'
    out_img = list(range(-2, 3))
    if pattern == 'random':
        in_c = 3
    else:
        in_c = len(in_img) * 3
    out_c = len(out_img) * 3
    d_in_c = out_c

    # for predict
    test_img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    weight_id = [ i.strip('.pth') for i in [
            'G_34_2634.pth']
    ]


class TransformerGANConfig:
    gpu_id = '0'
    random_seed = 119
    title = 'STTSRNet_256_perceptual'

    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)

    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # for pretrain_model
    # D_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/AE3DHR/D_2_212.pth'
    D_path, G_path = None, None
    G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/STTSRNet_256_rec/G_39_3014.pth'
    strict_mode = True
    # =================================
    #       model config
    # =================================
    realistic = True
    perceptual_decay = 0.025
    adver_decay = 0.025
    blur_weight = 0.05
    lr2hr = True
    SR = True
    out_size = 256
    start_epoch = 0
    stop_epoch = 40
    lr = 1e-4
    lr_adjust_step = 15
    momentum = 0.9
    decay_factor = 0.6
    loss = 'mae'
    D = 'single_discriminator'
    batch_size = 2
    num_workers = 2
    in_img = [0]  # 输入图像的通道数
    pattern = 'random'
    out_img = list(range(-2, 3))
    if pattern == 'random':
        in_c = 3
    else:
        in_c = len(in_img) * 3
    out_c = len(out_img) * 3
    d_in_c = out_c

    # for predict
    test_img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    weight_id = [i.rstrip('.pth') for i in [
        'G_0_50.pth',
         'G_10_810.pth',
         'G_11_886.pth',
         'G_12_962.pth',
         'G_13_1038.pth',
         'G_14_1114.pth',
         'G_15_1190.pth',
         'G_16_1266.pth',
         'G_17_1342.pth',
         'G_18_1418.pth',
         'G_19_1494.pth',
         'G_1_126.pth',
         'G_20_1570.pth',
         'G_21_1646.pth',
         'G_22_1722.pth',
         'G_23_1798.pth',
         'G_24_1874.pth',
         'G_25_1950.pth',
         'G_26_2026.pth',
         'G_27_2102.pth',
         'G_28_2178.pth',
         'G_29_2254.pth',
         'G_2_202.pth',
         'G_30_2330.pth',
         'G_31_2406.pth',
         'G_32_2482.pth',
         'G_33_2558.pth',
         'G_34_2634.pth',
         'G_35_2710.pth',
         'G_36_2786.pth',
         'G_37_2862.pth',
         'G_38_2938.pth',
         'G_39_3014.pth',
         'G_3_278.pth',
         'G_4_354.pth',
         'G_5_430.pth',
         'G_6_506.pth',
         'G_7_582.pth',
         'G_8_658.pth',
         'G_9_734.pth'
    ]]


class TransformerConfig:
    gpu_id = '0'
    random_seed = 119
    title = 'STTSRNet_256_rec'

    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)

    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # for pretrain_model
    # D_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/AE3DHR/D_2_212.pth'
    D_path, G_path = None, None
    #G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/AENet_V4/G_99_7575.pth'
    strict_mode = True
    # =================================
    #       model config
    # =================================
    realistic = False
    perceptual_decay = 0.025
    adver_decay = 0.025
    blur_weight = 0.05
    lr2hr = True
    SR = True
    out_size = 256
    start_epoch = 0
    stop_epoch = 40
    lr = 1e-4
    lr_adjust_step = 15
    momentum = 0.9
    decay_factor = 0.6
    loss = 'mae'
    D = 'single_discriminator'
    batch_size = 2
    num_workers = 2
    in_img = [0]  # 输入图像的通道数
    pattern = 'random'
    out_img = list(range(-2, 3))
    if pattern == 'random':
        in_c = 3
    else:
        in_c = len(in_img) * 3
    out_c = len(out_img) * 3
    d_in_c = out_c

    # for predict
    test_img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    weight_id = [i.rstrip('.pth') for i in [
    'G_39_3014.pth'
    ]]


class Config:
    gpu_id = '2'
    random_seed = 119
    title = 'AENet_V5_768_1'

    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'
    checkpoints_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/{}'.format(title)

    tensorboard_path = '/mnt/diskarray/mjb/Projects/3DSR/log/tensorboards/{}'.format(title)
    img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/train/{}'.format(title)
    test_img_log_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    # for pretrain_model
    # D_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/AE3DHR/D_2_212.pth'
    D_path, G_path = None, None
    #G_path = '/mnt/diskarray/mjb/Projects/3DSR/log/checkpoints/AENet_V4/G_99_7575.pth'
    strict_mode = True
    # =================================
    #       model config
    # =================================
    adver_decay = 0.025
    blur_weight = 0.05
    lr2hr = True
    SR = True
    input_size = 768
    start_epoch = 0
    stop_epoch = 40
    lr = 1e-4
    lr_adjust_step = 15
    momentum = 0.9
    decay_factor = 0.6
    loss = 'mae'
    D = 'single_discriminator'
    batch_size = 1
    num_workers = 0
    in_img = [0]  # 输入图像的通道数
    pattern = 'random'
    out_img = list(range(-2, 3))
    if pattern == 'random':
        in_c = 3
    else:
        in_c = len(in_img) * 3
    out_c = len(out_img) * 3
    d_in_c = out_c

    # for predict
    test_img_log_path = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/{}'.format(title)
    weight_id = ['G_0_50',
 'G_10_810',
 'G_11_886',
 'G_12_962',
 'G_13_1038',
 'G_14_1114',
 'G_15_1190',
 'G_16_1266',
 'G_17_1342',
 'G_18_1418',
 'G_19_1494',
 'G_1_126',
 'G_20_1570',
 'G_21_1646',
 'G_22_1722',
 'G_23_1798',
 'G_24_1874',
 'G_25_1950',
 'G_26_2026',
 'G_27_2102',
 'G_28_2178',
 'G_29_2254',
 'G_2_202',
 'G_30_2330',
 'G_31_2406',
 'G_32_2482',
 'G_33_2558',
 'G_34_2634',
 'G_35_2710',
 'G_36_2786',
 'G_37_2862',
 'G_3_278',
 'G_5_430',
 'G_6_506',
 'G_7_582',
 'G_8_658']


