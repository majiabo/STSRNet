import tensorflow as tf
from skimage import metrics
import numpy as np
import time
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
p1 = '/mnt/disk_8t/kara/3DSR/log/imgs/test/mae_v0/G_19_1519/0.jpg'
img = cv2.imread(p1)
size = 768
img1 = img[:size, :size]
img2 = img[size:, :size]
print('skimage ...')
start = time.time()
for i in range(100):
    psnr1 = metrics.peak_signal_noise_ratio(img1, img2)
    ssim1 = metrics.structural_similarity(img1, img2, multichannel=True, win_size=11, sigma=1.5)
end = time.time()
print('psnr=', psnr1)
print('ssim=', ssim1)
print('time=', end-start)

print('tf ...')
start = time.time()
for i in range(100):
    #ssim2 = tf.image.ssim_multiscale(img1, img2, max_val=255, filter_size=7)
    img1 = np.stack([img1, img1], axis=0)
    img2 = np.stack([img2, img2], axis=0)
    psnr2 = tf.image.psnr(img1, img2, max_val=255)
    ssim2 = tf.image.ssim(tf.convert_to_tensor(img1), tf.convert_to_tensor(img2), max_val=255)
end = time.time()
print('psnr=', psnr2)
print('ssim=', ssim2)
print('time=', end-start)


