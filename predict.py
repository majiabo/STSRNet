from lib.net import STSRNet
import numpy as np
import cv2
import torch
# modified this line
from lib.configs import STSRNetSISRConfig
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = STSRNetSISRConfig()
    weight_path = './assets/G_57_4555.pth'
    img_path = './assets/LR.jpg'
    gt_path = './assets/GT.jpg'
    G = STSRNet(sr=args.SR, out_num=len(args.out_img))
    G.load_state_dict(torch.load(weight_path))
    G.eval()
    with torch.no_grad():
        img = cv2.imread(img_path)[..., ::-1].astype('float32')/255
        img = np.transpose(img, axes=[2, 0, 1])[np.newaxis, ...]
        img = torch.from_numpy(img)
        gen = np.transpose(G(img).numpy()[0], axes=[1, 2, 0])
        gen = np.concatenate([gen[..., i*3:(i+1)*3] for i in range(5)], axis=1)
        gt = cv2.imread(gt_path)[..., ::-1].astype('float32')/255
        plt.subplot(211)
        plt.imshow(gen)
        plt.subplot(212)
        plt.imshow(gt)
        plt.show()
