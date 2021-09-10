from skimage.measure import compare_psnr, compare_ssim
from lib.get_depth_opt import gen_layer_map
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import cv2
import os
import lpips
import torch


class EvalDataSetV1:
    """
    This class is used to process Test image with a middle directory.
    """

    def __init__(self, data_root, target_layers: list, gen_layer=0, size=128,
                 num_of_line = 1):
        self.target_layers = target_layers
        self.size = size
        self.num_of_line = num_of_line
        self.datasets = [EvalDataSet(os.path.join(data_root, index), gen_layer=gen_layer)
                                       for index in target_layers]
        for index, _ in enumerate(self.datasets):
            self.datasets[index].size = self.size
            self.datasets[index].num_of_line = self.num_of_line

    def set_images(self):
        raise NotImplementedError('EvalDataSet1.set_images method is not implemented ...')

    def __len__(self):
        for index, _ in enumerate(self.datasets):
            self.datasets[index].size = self.size
            self.datasets[index].num_of_line = self.num_of_line
        return len(self.datasets[0])

    def read_img(self, item):
        gen_imgs = []
        real_imgs = []
        for index, dataset in enumerate(self.datasets):
            gen, real, _ = dataset[item]
            gen_imgs.append(gen[0])
            real_imgs.append(real[0])
        return gen_imgs, real_imgs, _

    def __getitem__(self, item):
        return self.read_img(item)


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


class Evaluator:
    def __init__(self, data_set: EvalDataSet, ssim_psnr=True, fft_root=None, layer_map_root=None, tf_backend=False, mse=False, mae=False,
                 lpips_flag=False, sample_num=None):
        """

        :param data_set:
        :param fft_root:  None or 'full path', None to skip
        :param layer_map_root: same as fft_root
        """
        self.data_set = data_set
        self.fft_root = fft_root
        self.layer_map_root = layer_map_root
        self.tf_backend = tf_backend
        self.ssim_psnr_flag = ssim_psnr
        self.mse = mse
        self.mae = mae
        self.lpips_flag = lpips_flag
        self.check_path(fft_root)
        self.check_path(layer_map_root)
        self.sample_num = len(data_set) if sample_num is None else sample_num
        self.lpips_alex = lpips.LPIPS(net='alex').cuda()


    @staticmethod
    def check_path(path):
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                print(path, 'exists, skip...')

    def ssim_psnr(self, gens, reals):
        psnrs = []
        ssims = []
        for gen, real in zip(gens, reals):
            if self.tf_backend:
                psnr = tf.image.psnr(real, gen, max_val=255).numpy()
                ssim = tf.image.ssim(tf.convert_to_tensor(real), tf.convert_to_tensor(gen), max_val=255).numpy()
            else:
                psnr = compare_psnr(real, gen)
                ssim = compare_ssim(real, gen, multichannel=True)
            psnrs.append(psnr)
            ssims.append(ssim)
        return ssims, psnrs

    def cal_mse(self, gens, reals):
        mses = []
        for gen, real in zip(gens, reals):
            gen = gen.astype('float32')
            real = real.astype('float32')
            if self.tf_backend:
                mse = tf.reduce_mean(tf.square(gen - real)).numpy()
            else:
                mse = np.mean(np.square(gen-real))

            mses.append(mse)
        return mses

    def cal_mae(self, gens, reals):
        mses = []
        for gen, real in zip(gens, reals):
            gen = gen.astype('float32')
            real = real.astype('float32')
            if self.tf_backend:
                mse = tf.reduce_mean(tf.abs(gen - real)).numpy()
            else:
                mse = np.mean(np.abs(gen - real))

            mses.append(mse)
        return mses

    def cal_lpips(self, gens, reals):
        mses = []
        for gen, real in zip(gens, reals):
            gen = (gen.astype('float32')/255-0.5)*2
            real = (real.astype('float32')/255-.5)*2
            gen = torch.from_numpy(np.transpose(gen, axes=[2, 0, 1])[None, ...]).cuda()
            real = torch.from_numpy(np.transpose(real, axes=[2, 0, 1])[None, ...]).cuda()
            mse = self.lpips_alex(gen, real).cpu()
            mses.append(mse.item())
        return mses

    @staticmethod
    def fft2(gens, reals):
        fft_gens = []
        fft_reals = []
        for gen, real in zip(gens, reals):
            # 根据公式转成灰度图
            gen_gray = cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY)
            real_gray = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)

            # 进行傅立叶变换，并显示结果
            fft2_gen = np.fft.fft2(gen_gray)
            fft2_real = np.fft.fft2(real_gray)
            # 将图像变换的原点移动到频域矩形的中心，并显示效果
            shift2center_gen = np.fft.fftshift(fft2_gen)
            shift2center_real = np.fft.fftshift(fft2_real)

            # 对中心化后的结果进行对数变换，并显示结果
            log_shift2center_gen = np.log(1 + np.abs(shift2center_gen))
            log_shift2center_real = np.log(1 + np.abs(shift2center_real))
            fft_gens.append(log_shift2center_gen)
            fft_reals.append(log_shift2center_real)
        return fft_gens, fft_reals

    @staticmethod
    def multi_to_one(gens, reals):
        top = np.concatenate(gens, axis=1)
        bottom = np.concatenate(reals, axis=1)
        pad = np.concatenate((top, bottom), axis=0)
        return pad

    @staticmethod
    def average(metric):
        if metric:
            avgs = np.array(metric)
            print(avgs)
            avgs = avgs.sum(axis=0)/avgs.shape[0]
        else:
            avgs = None
        return avgs

    def eval(self, select=None):
        ssims = []
        psnrs = []
        mses = []
        maes = []
        lpips_scores = []
        if select is not None:
            self.data_set.set_images(select)
        for index, (gen_imgs, real_imgs, img_name) in tqdm(enumerate(self.data_set)):
            if index > self.sample_num:
                break
            if self.ssim_psnr_flag:
                ssim, psnr = self.ssim_psnr(gen_imgs, real_imgs)
            if self.fft_root is not None:
                gen_fft, real_fft = self.fft2(gen_imgs, real_imgs)
                fft_pad = self.multi_to_one(gen_fft, real_fft)
                fft_path = os.path.join(self.fft_root, img_name)
                fft_pad = (fft_pad - fft_pad.min())*(255/fft_pad.max())
                cv2.imwrite(fft_path, fft_pad.astype('uint8'))
            if self.layer_map_root is not None:
                layer_map = gen_layer_map([gen_imgs, real_imgs], 11, size=768)
                layer_map_path = os.path.join(self.layer_map_root, img_name)
                cv2.imwrite(layer_map_path, layer_map[..., ::-1])
            if self.ssim_psnr_flag:
                ssims.append(ssim)
                psnrs.append(psnr)

            if self.mse:
                mse = self.cal_mse(gen_imgs, real_imgs)
                mses.append(mse)

            if self.mae:
                mae = self.cal_mae(gen_imgs, real_imgs)
                maes.append(mae)

            if self.lpips_flag:
                lpips_score = self.cal_lpips(gen_imgs, real_imgs)
                lpips_scores.append(lpips_score)

        avg_ssim = self.average(ssims)
        avg_psnr = self.average(psnrs)
        avg_mse = self.average(mses)
        avg_mae = self.average(maes)
        avg_lpips = self.average(lpips_scores)

        return avg_ssim, avg_psnr, avg_mse, avg_mae, avg_lpips







if __name__ == '__main__':
    data_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/SR-Refocus/Deep-Z-SR/netG_epoch_2_38'
    e_data_root = '/mnt/diskarray/mjb/Projects/3DSR/log/imgs/test/SR-Refocus/MWCNN/netG_epoch_2_55'
    evalset = EvalDataSetV1(data_root, [f'{i}' for i in range(-2, 3)])
    eval_set = EvalDataSet(e_data_root, 0)
    eval_set.size = 128
    eval_set.num_of_line = 5
    gens, reals, _ = evalset[0]
    e_gens, e_reals, _ =eval_set[0]
    print()