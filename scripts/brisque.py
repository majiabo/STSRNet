"""
reference-less metrics
"""
from imquality import brisque
from multiprocessing import Pool
import os
import cv2


def func(arg):
    print(arg)
    p, GT = arg
    temp_score = []
    image = cv2.imread(p)[..., ::-1]
    for i in range(0, 5):
        try:
            if GT:
                patch = image[128:, i * 128:(i + 1) * 128]
            else:
                patch = image[:128, i * 128:(i + 1) * 128]
            s = brisque.score(patch)
        except:
            s = None
        temp_score.append(s)
    return temp_score


if __name__ == '__main__':
    log_file = '../log/metrics_brisque.txt'
    targets = {
        'HR': '../log/imgs/test/response/Liff/epoch-bes',
        'Liif': '../log/imgs/test/response/Liff/epoch-bes',
        'VDSR': '../log/imgs/test/SR-Refocus/VDSR/netG_epoch_2_39',
        'SRResNet': '../log/imgs/test/SR-Refocus/SRResNet/netG_epoch_2_55',
        'CARN': '../log/imgs/test/SR-Refocus/CARN/netG_epoch_2_59',
        'EDSR': '../log/imgs/test/SRSOTA/EDSR_0.25_512_depth/netG_epoch_2_36',
        'SRCNN': '../log/imgs/test/SR-Refocus/SRCNN/netG_epoch_2_59',
        'MWCNN': '../log/imgs/test/SR-Refocus/MWCNN/netG_epoch_2_59',
        'Deep-Z': '../log/imgs/test/SR-Refocus/Deep-Z-SR/netG_epoch_2_39_stitched', # remember to concat images
        'RCAN': '../log/imgs/test/SR-Refocus/RCAN/netG_epoch_2_59',
        'RFANet': '../log/imgs/test/response/RFANet_v1/G_99_7873',
        'STSRNet': '../log/imgs/test/our/STTRNet2-TransformerV5-plus-4/G_57_4555_used_in_paper',
    }
    scores = {}
    for key, dir in targets.items():
        if key == 'HR':
            GT = True
        else:
            GT = False
        print('processing:', key)
        image_files = os.listdir(dir)
        paths = [os.path.join(dir, i) for i in image_files]
        pool = Pool(16)
        paths = [[p, GT] for p in paths]
        res = pool.map(func, paths)
        pool.close()
        pool.join()
        # res = res.get()
        scores[key] = []
        for i in range(5):
            t = [j[i] for j in res if j[i] is not None]
            scores[key].append(sum(t)/len(t))
    with open(log_file, 'w') as f:
        for k, v in scores.items():
            f.write('{}:{}'.format(k, v))



