"""
datalink: https://pan.baidu.com/share/init?surl=S6qVWCsEXC5jcZN6SDWGdA
code: iz73
"""
import openslide
from skimage import morphology
import random
import numpy as np
import json
import os
import cv2


def binary_sep(img, threColor=35, threVol=2000, Blocksize=101, C=10):
    '''
    执行图像分割，分割背景与前景
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    th1 = cv2.adaptiveThreshold(gray[50:462, 50:462], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, Blocksize,
                                C)
    a = th1
    a = a <= 0  # the foreground is 1,and background is 0,so inverse the color
    dst = morphology.remove_small_objects(a, min_size=threVol, connectivity=1)
    imgBin = dst > 0
    s = imgBin.sum()
    if s > 1:
        flag = True
    else:
        flag = False
    return flag


def read_images(handle: openslide.OpenSlide, left_tops: list, size: tuple):
    """
    read images at level 0
    :param handle:
    :param left_tops:
    :param size:
    :return:
    """
    images = []
    for x, y in left_tops:
        img = np.array(handle.read_region((x, y), 0, size).convert('RGB'))
        images.append(img)
    return images


def gen_coors(handle: openslide.OpenSlide, size, patches):
    slide_w, slide_h = handle.dimensions
    w, h = size
    left_tops = []
    while len(left_tops) <= patches:
        x = random.randint(w, slide_w - w)
        y = random.randint(h, slide_h - h)
        img = np.array(handle.read_region((x, y), 0, size).convert('RGB'))
        flag = binary_sep(img)
        if flag:
            left_tops.append((x, y))
            print('progress:[{}/{}]'.format(len(left_tops), patches))
    return left_tops


def save_images(images: list, left_tops: list, img_root: str, slide_id: str, layer: int):
    path_list = []
    for img, (x, y) in zip(images, left_tops):
        path = os.path.join(img_root, '{}_{}_{}_{}.jpg'.format(slide_id, x, y, layer))
        flag = cv2.imwrite(path, img[..., ::-1])
        path_list.append(path)
        if not flag:
            raise RuntimeError('img save failed, path:'.format(path))
    return path_list


if __name__ == '__main__':
    size = (768, 768)
    patches_num = 5000
    original_root = '/mnt/disk_8t/kara/DATA/mulit_layer_3d_img/20x_tiff'
    img_root = '/mnt/disk_8t/kara/3DSR/data/imgs'
    label_root = '/mnt/disk_8t/kara/3DSR/data/label'
    slides = [os.path.join(original_root, line) for line in os.listdir(original_root)]

    exclude = []

    for slide_dir in slides:
        print('processing:{}'.format(slide_dir))
        slide_names = [line for line in os.listdir(slide_dir) if 'Extended' not in line]
        p = [os.path.join(slide_dir, line) for line in os.listdir(slide_dir) if 'Extended' in line]
        handle = openslide.OpenSlide(p[0])
        print('Generate effective coors...')
        left_tops = gen_coors(handle, size, patches_num)

        for slide_name in slide_names:
            print('sub:', slide_name)
            if slide_name in exclude:
                print('exclude:', slide_name)
                continue
            slide_id, _, layer = slide_name.split('.')[0].split('_')
            img_slide_root = os.path.join(img_root, slide_id, layer)
            if not os.path.exists(img_slide_root):
                os.makedirs(img_slide_root)
            label_slide_root = os.path.join(label_root, slide_id)
            if not os.path.exists(label_slide_root):
                os.makedirs(label_slide_root)

            json_name = os.path.join(label_slide_root, 'layer_{}.json'.format(layer))
            if os.path.exists(json_name):
                print(json_name, 'exist, skip...')
                continue

            slide_path = os.path.join(slide_dir, slide_name)
            handle = openslide.OpenSlide(slide_path)
            images = read_images(handle, left_tops, size)
            path_list = save_images(images, left_tops, img_slide_root, slide_id, layer)
            with open(json_name, 'w') as f:
                json.dump(path_list, f)

