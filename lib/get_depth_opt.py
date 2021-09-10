# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:36:52 2019

@author: 42965
"""

import cv2
import numpy as np
from skimage import morphology
import os
from scipy import signal
from tqdm import tqdm

windowsize=9
color = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0),
         (0, 255, 255), (255, 0, 255), (255, 255, 0),
         (255, 255, 255), (100, 255, 0), (0, 100, 255), (100, 100, 255)]


def _BinarySP(img, threVol=1000, size=768):
        '''
        执行图像分割，分割背景与前景，img为灰度图
        暂时使用固定阈值二值化
        '''
        gray = cv2.resize(img, (size, size))
        #th1 = cv2.adaptiveThreshold(gray[:,:],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,Blocksize,C)
        _,th1 = cv2.threshold(gray,240,255,cv2.THRESH_BINARY)
        a = th1
        a = a<=0 #the foreground is 1,and background is 0,so inverse the color
        dst=morphology.remove_small_objects(a, min_size=threVol, connectivity=1)
        imgBin = dst > 0
        s = imgBin.sum()
        if s >1:
            flag = True
        else:
            flag = False
        return [flag,imgBin]
    

def _sameSize(img_std,img_cvt):
    """
    确保img_cvt的尺寸与img_std相等
    """
    x,y=img_std.shape[:2]
    img_cvt.resize((x,y))
    return img_cvt


def getLaplacePyr(img):
    """
    计算一层拉普拉斯金字塔，返回高频、低频两个逐像素数组
    """
    firstLevel=img.copy()
    secondLevel=cv2.pyrDown(firstLevel)
    lowFreq=cv2.pyrUp(secondLevel)
    highFreq=cv2.subtract(firstLevel, lowFreq)
    return lowFreq,highFreq


def analysis(img_list):
    """
    输入一个不同层图片组成的列表，得到一个对应的逐像素清晰度的C*H*W数组
    """
    num=len(img_list)
    #max_shift = (windowsize-1)//2
    result=[]
    for index in range(num):
        lowFreq,highFreq = getLaplacePyr(img_list[index])
        m=np.divide(highFreq,lowFreq)
        m=np.power(m,2)
        arr=np.ones([windowsize,windowsize])
        result.append(signal.convolve2d(m,arr,mode='same',boundary='fill'))

    return np.array(result)


def get_mask(result,imgbin):
    """
    result为ananlysis所输出的清晰度数组，imgbin为_BinarySP给出的前景分割二值图
    通过以上两者得到mask
    mask取值中，0表示背景，1开始的数字表示在第几张图上
    """
    #max_arr=np.max(result,axis=0)
    #x,y=max_arr.shape
    #mask=np.zeros([x,y])
    #for i in range(0,x):
    #    for j in range(0,y):
    #        if imgbin[i,j]:
    #            mask[i,j]=int(np.where(result[:,i,j]==max_arr[i,j])[0][0])+1
    max_arr=np.max(result,axis=0)
    x,y=max_arr.shape
    arr_bin=(result==max_arr).transpose((1,2,0))#返回一个H*W*C的二值数组，表明最大值出现位置
    arr_index=np.array(np.where(arr_bin))#返回一个3*(H*W)的数组
    _,indices=np.unique(arr_index[0:2,:],axis=1, return_index=True)#返回去除重复后arr_index索引数组
    mask=np.where(imgbin,arr_index[2,indices].reshape(x,y)+1,np.zeros([x,y]))
    return mask


def split_img(img):
    h = img.shape[0]
    w = img.shape[1]
    img_list = []
    for i in range(int(w/h)):
        img_list.append(img[:,i*h:(i+1)*h])
    return img_list


def generate_image_labe(num, img_size=768):
    step = int(img_size/num)
    standard = np.zeros([img_size, step, 3])

    if len(color) < num:
        raise ValueError('defined color less than predefined color table!!!')
    for i in range(num):
        standard[i*step: (i+1)*step, :, 0] = color[i][0]
        standard[i*step: (i+1)*step, :, 1] = color[i][1]
        standard[i*step: (i+1)*step, :, 2] = color[i][2]
    return standard


def color_brush(mask, num, img_size=768):
    color_board = np.zeros([img_size, img_size, 3])
    if len(color) < num:
        raise ValueError('defined color less than predefined color table!!!')
    for i in range(num):
        color_board[:, :, 0][mask == i+1] = color[i][0]
        color_board[:, :, 1][mask == i+1] = color[i][1]
        color_board[:, :, 2][mask == i+1] = color[i][2]
    return color_board


def gen_layer_map(img_list: list, num: int, size=768):
    """

    :param img_list: [ gen_list, real_list]
    :param num:
    :return:
    """
    results_img = []
    standard = generate_image_labe(num, size)
    for img in img_list:
        img_cl = np.concatenate(img, axis=1)
        img = [cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY) for mat in img]
        result = analysis(img)
        _, imgbin = _BinarySP(img[int(len(img) / 2)])
        mask = get_mask(result, imgbin)
        mask_cl = color_brush(mask, num)
        ss = np.hstack((img_cl, mask_cl, standard))
        results_img.append(ss)
    results_img = np.concatenate(results_img, axis=0)
    return results_img


if __name__ == '__main__':
    num = 7
    root_path = '/home/kara/SRLog/udense_light/test_img/5layers_ds/G_19_4'
    root_target_path = '/mnt/diskarray/mjb/SRLog/udense_light/depth_img/5layer_ds/G_19_4'
    if not os.path.isdir(root_target_path):
        os.makedirs(root_target_path)

    path_list = [line for line in os.listdir(root_path)]
    standard = generate_image_labe(num)
    for path in tqdm(path_list):
        img_cl = cv2.imread(os.path.join(root_path,path))
        img = cv2.cvtColor(img_cl,cv2.COLOR_BGR2GRAY)
        results_img = []
        for i in range(2):
            img_list = split_img(img[i*512:(i+1)*512,:])
            result = analysis(img_list)
            _,imgbin = _BinarySP(img_list[int(len(img_list)/2)])
            mask = get_mask(result,imgbin)
            mask_cl = color_brush(mask,num)
            ss = np.hstack((img_cl[i*512:(i+1)*512,:,:],mask_cl,standard))
            results_img.append(ss)
        results_img = np.concatenate(results_img,axis = 0)
        save_path = os.path.join(root_target_path,path)
        cv2.imwrite(save_path,results_img)

