"""
Extract right score from test log file
"""
import os

file_path = '/mnt/diskarray/mjb/Projects/3DSR/log/ssim_psnr/our/STTRNet1_SISR_rec.txt'

title = 'weight:G_34_2634]ALL'

targets = []
with open(file_path) as f:
    counter = 0
    flag = False
    for line in f:
        if flag:
            elements = line.split('[')[-1].split(']')[0].split()
            elements = [float(i) for i in elements]
            targets.append(elements)
            counter += 1
        if title in line:
            flag = True
        if counter >= 5:
            break

round_arg = [3, 2, 2, 2, 4]
names = ['ssim', 'psnr', 'mse', 'mae', 'lpips']
for a, n, name in zip(targets, round_arg, names):
    a = [round(i, n) for i in a]
    print(f'{name}:{a}')
