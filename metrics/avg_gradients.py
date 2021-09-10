"""
reference-less metrics
"""
from skimage import measure
import numpy as np
import os
import cv2


class EvalDataSet:
    size = 768
    num_of_line = 11

    def __init__(self, data_root, gen_layer=0):
        self.img_names = os.listdir(data_root)
        self.gen_layer = gen_layer
        self.data_root = data_root
        self.img_paths = [os.path.join(data_root, line) for line in self.img_names]

    def set_images(self, name):
        self.img_names = [line for line in self.img_names if line == name]
        self.img_paths = [os.path.join(self.data_root, line) for line in self.img_names]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        gen_imgs, real_imgs = self.read_img(img_path)
        return gen_imgs, real_imgs, self.img_names[item]

    def read_img(self, path):
        gen_imgs = []
        real_imgs = []
        img = cv2.imread(path)[..., ::-1]
        for i in range(self.num_of_line):
            gen = img[self.gen_layer*self.size:(self.gen_layer+1)*self.size, i*self.size:(i+1)*self.size]
            real = img[-self.size:, i*self.size:(i+1)*self.size]
            gen_imgs.append(gen)
            real_imgs.append(real)
        return gen_imgs, real_imgs


def avg_gradients(img):
    img = img.astype('float32')
    x = img[1:, :-1] - img[1:, 1:]
    y = img[:-1, 1:] - img[1:, 1:]
    delta = np.sqrt(x*x+y*y)
    return delta.mean()


if __name__ == '__main__':
    log_file = 'metrics_avg_gradient.txt'
    targets = {
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
    img_size = 128
    img_num_per_line = 5
    scores = {}
    GT = True
    for key, dir in targets.items():
        print('processing:', key)
        temp_score = {k:[] for k in range(0, 5)}
        gt_score = {k:[] for k in range(0, 5)}
        image_files = os.listdir(dir)
        for file in image_files:
            p = os.path.join(dir, file)
            image = cv2.imread(p)[..., ::-1]
            for i in range(0, 5):
                patch = image[:128, i*128:(i+1)*128]
                s = avg_gradients(patch)
                #s = brisque.score(patch)
                temp_score[i].append(s)
            if GT:
                for i in range(0, 5):
                    patch = image[128:, i*128:(i+1)*128]
                    s = avg_gradients(patch)
                    #s = brisque.score(patch)
                    gt_score[i].append(s)
           
        scores[key] = [sum(temp_score[i])/len(temp_score[i]) for i in range(0, 5)]
        if GT:
            scores['HR'] = [sum(gt_score[i])/len(gt_score[i]) for i in range(0, 5)]
            GT = False

    with open(log_file, 'w') as f:
        for k, v in scores.items():
            f.write('{}:{}'.format(k, v))



