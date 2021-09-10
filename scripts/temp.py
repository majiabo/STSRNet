import cv2
import os
import numpy as np
from imquality import brisque

if __name__ == '__main__':
    p = '../metrics_entropy.txt'
    with open(p) as f:
        lines = [line.strip() for line in f]
    for line in lines:
        prefix, data = line.split(':')
        print(prefix, ['{:.2f}'.format(float(l)) for l in data[1:-1].split(',')])


