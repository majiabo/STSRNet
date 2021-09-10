from torch.utils.data import Dataset
import numpy as np
import random
import torch
import json
import os
import cv2


class D3Dataset(Dataset):
    slides = {
            'train':['10140015','10140018','10140064'],'test':['10140071','10140074']
            }

    def __init__(self, data_root, mode, pattern, input_layer:list, out_layer:list, lr2hr=True, seed = 0, input_size=384):
        """

        :param data_root:
        :param mode:
        :param pattern: str, 'random' or 'all'
        :param input_layer:
        :param out_layer:
        :param seed:
        """
        self.input_size = input_size
        self.mode = mode
        assert pattern in ['random', 'all'], '{} pattern is not supported...'.format(pattern)
        self.pattern = pattern
        self.input_layer = input_layer
        self.out_layer = out_layer
        self.lr2hr = lr2hr
        self.label_root = os.path.join(data_root, 'label')
        self.layers_info = self.parser_jsons()
        self.indexer = list(range(len(self.layers_info['layer_0'])))

        random.seed(seed)
        random.shuffle(self.indexer)

    def parser_jsons(self):
        layers_info = {'layer_{}'.format(i):[] for i in range(-5,6)}
        if self.mode == 'train':
            slides = self.slides['train']
        elif self.mode == 'test':
            slides = self.slides['test']
        else:
            raise ValueError('Not Supported mode:{}'.format(self.mode))
        for s in slides:
            for i in range(-5,6):
                layer_item = 'layer_{}'.format(i)
                json_path = os.path.join(self.label_root, s, layer_item+'.json')
                with open(json_path) as f:
                    items = json.load(f)
                layers_info[layer_item] += items
        return layers_info
    
    @staticmethod
    def read_imgs(names, crop_function=None, down_sample=False, resolution=None):
        imgs = []
        for n in names:
            img = cv2.imread(n)[..., ::-1]
            img = cv2.resize(img, (512, 512))
            if crop_function is not None:
                img = crop_function(img)
            if down_sample:
                img = D3Dataset.degrade_img(img)
            if resolution:
                img = cv2.resize(img, None, fx=resolution, fy=resolution)
            img = img.astype('float32')
            img = img/255.
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=2)
        return imgs

    def crop_img(self, img):
        w, h = img.shape[:2]
        x = int(w/4)
        y = int(h/4)
        w = int(w/2)
        h = int(h/2)
        img = img[x:x+w, y:y+h]
        return img

    @staticmethod
    def degrade_img(img, blur=True):
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        if blur:
            img = cv2.GaussianBlur(img, (5, 5), 3)
        return img

    @staticmethod
    def blur_map(img1, img2):
        map = img1.mean(axis=-1) - img2.mean(axis=-1)
        return map

    def select_names(self, item):
        item = self.indexer[item]
        input_names = []
        output_names = []
        for n in self.input_layer:
            input_names.append(self.layers_info['layer_{}'.format(n)][item])

        for n in self.out_layer:
            output_names.append(self.layers_info['layer_{}'.format(n)][item])

        return input_names, output_names

    def __len__(self):
        return len(self.indexer)
        
    def __getitem__(self, item):
        input_names, output_names = self.select_names(item)
        if self.pattern == 'random' and self.mode == 'train':
            input_names = [random.choice(input_names)]
        if self.lr2hr:
            input_imgs = self.read_imgs(input_names, down_sample=True)
            output_imgs = self.read_imgs(output_names)
        else:
            input_imgs = self.read_imgs(input_names, crop_function=self.crop_img, down_sample=False)
            output_imgs = self.read_imgs(output_names, crop_function=self.crop_img, down_sample=False)

        input_imgs = np.transpose(input_imgs, axes=(2, 0, 1))
        output_imgs = np.transpose(output_imgs, axes=(2, 0, 1))
        
        input_imgs = torch.from_numpy(input_imgs)
        output_imgs = torch.from_numpy(output_imgs)

        return input_imgs, output_imgs


class AEDataset(D3Dataset):
    def __init__(self, data_root, mode, pattern='random', input_layer=None,
                 out_layer=None, lr2hr=False, seed=0, input_size=384, data_prefix=None, fix_layer=None, resolution=1):
        """
        fix_layer:当fix_layer为None时，随意选择layer，否则only choose fixe_layer
        """
        super(AEDataset, self).__init__(data_root, mode, pattern, input_layer, out_layer, lr2hr, seed)
        # self.layer_norm_factor = int((len(out_layer)-1)/2)
        self.layer_norm_factor = 2
        self.data_prefix = data_prefix
        self.fix_layer = fix_layer
        self.input_size = input_size
        self.resolution = resolution

    def crop_img(self, p):
        x, y = p

        def func(img):
            return img[y:y+self.input_size, x:x+self.input_size]
        return func

    def get_pad(self, img, size: int, layer: int):
        pad = -np.ones([2, size, size], dtype=np.float32)
        binary = binary_sep(img).astype(np.float32)
        pad[0] = binary
        layer_value = layer / self.layer_norm_factor
        pad[1] = layer_value
        return pad

    def get_depth_mask(self, size: int, layer: int):
        pad = -np.ones([1, size, size], dtype=np.float32)
        layer_value = (layer + 1e-4) / self.layer_norm_factor
        pad[0] = layer_value
        return pad

    def get_names(self, item, input_layer, out_layer):
        item = self.indexer[item]
        input_names = []
        output_names = []
        for n in input_layer:
            name = self.layers_info['layer_{}'.format(n)][item]
            if self.data_prefix:
                name = self.replace_prefix(name)
            input_names.append(name)

        for n in out_layer:
            name = self.layers_info['layer_{}'.format(n)][item]
            if self.data_prefix:
                name = self.replace_prefix(name)
                # print(name)
            output_names.append(name)
        return input_names, output_names

    def replace_prefix(self, path):
        head, tail = path.split('3DSR/')
        new_path = os.path.join(self.data_prefix, '3DSR', tail)
        return new_path

    def get_points(self, max_v):
        x, y = random.randint(0, max_v), random.randint(0, max_v)
        return x, y

    def __getitem__(self, item):
        if self.fix_layer is None:
            layer = random.choice(self.out_layer)
        else:
            layer = self.fix_layer
        input_names, output_names = self.get_names(item, self.input_layer, [layer])
        # if self.mode == 'train':
        #     x, y = self.get_points(768 - self.input_size)
        # else:
        #     x = int((768-self.input_size)/2)
        #     y = x
        x, y = 0, 0
        if self.lr2hr:
            input_imgs = self.read_imgs(input_names, crop_function=self.crop_img((x, y)), down_sample=True,
                                        resolution=self.resolution)
            output_imgs = self.read_imgs(output_names, crop_function=self.crop_img((x, y)), resolution=self.resolution)
        else:
            input_imgs = self.read_imgs(input_names, crop_function=self.crop_img((x, y)), down_sample=False,
                                        resolution=self.resolution)
            output_imgs = self.read_imgs(output_names, crop_function=self.crop_img((x, y)), down_sample=False,
                                         resolution=self.resolution)
        # pad = self.get_pad(input_imgs, self.input_size, layer)
        depth_mask_size = int(self.input_size*self.resolution) if self.resolution is not None else self.input_size
        pad = self.get_depth_mask(depth_mask_size, layer)

        input_imgs = np.transpose(input_imgs, axes=(2, 0, 1))
        output_imgs = np.transpose(output_imgs, axes=(2, 0, 1))

        input_imgs = torch.from_numpy(input_imgs)
        output_imgs = torch.from_numpy(output_imgs)
        depth_mask = torch.from_numpy(pad)

        return input_imgs, output_imgs, depth_mask


class AttentionDataSet(D3Dataset):
    def __init__(self, data_root, mode, pattern='random', input_layer=None,
                 out_layer=None, lr2hr=False, seed=0, input_size=128, data_prefix=None, resolution=None, print_real_name=False):
        super(AttentionDataSet, self).__init__(data_root, mode, pattern, input_layer, out_layer, lr2hr, seed)
        self.input_size = input_size
        self.data_prefix = data_prefix
        self.resolution = resolution
        self.print_real_name = print_real_name

    def replace_prefix(self, path):
        head, tail = path.split('3DSR/')
        new_path = os.path.join(self.data_prefix, '3DSR', tail)
        return new_path

    def get_weight_map(self, input_, output):
        if self.lr2hr:
            input_ = torch.nn.functional.interpolate(input_, scale_factor=2)
        num = int(output.shape[1]/3)
        temp = list()
        for i in range(num):
            temp = temp + [torch.mean(torch.abs(input_-output[:, i*3:(i+1)*3]), dim=1, keepdim=True) for _ in range(3)]
        return torch.cat(temp, dim=1)

    def crop_img(self, p):
        x, y = p

        def func(img):
            return img[y:y+self.input_size, x:x+self.input_size]
        return func

    def get_points(self, max_v):
        x, y = random.randint(0, max_v), random.randint(0, max_v)
        return x, y

    def read_images(self, names, crop_function=None, down_sample=False, resolution=None):
        imgs = []
        for n in names:
            img = cv2.imread(n)[..., ::-1]
            img = cv2.resize(img, (512, 512))
            if crop_function is not None:
                img = crop_function(img)
            if down_sample:
                img = D3Dataset.degrade_img(img)
            if resolution:
                img = cv2.resize(img, None, fx=resolution, fy=resolution)
            img = img.astype('float32')
            img = img/255.
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=2)
        return imgs

    def __getitem__(self, item):
        input_names, output_names = self.select_names(item)
        if self.data_prefix:
            input_names = [self.replace_prefix(p) for p in input_names]
            output_names = [self.replace_prefix(p) for p in output_names]

        if self.pattern == 'random' and self.mode == 'train':
            input_names = [random.choice(input_names)]
        # if self.mode == 'train':
        #     x, y = self.get_points(768 - self.input_size)
        # else:
        #     x = int((768-self.input_size)/2)
        #     y = x
        x, y = 0, 0
        if self.lr2hr:
            input_imgs = self.read_images(input_names, crop_function=self.crop_img((x, y)), down_sample=True, resolution=self.resolution)
            output_imgs = self.read_images(output_names, crop_function=self.crop_img((x, y)), resolution=self.resolution)
        else:
            input_imgs = self.read_images(input_names, crop_function=self.crop_img((x, y)), down_sample=False, resolution=self.resolution)
            output_imgs = self.read_images(output_names, crop_function=self.crop_img((x, y)), down_sample=False, resolution=self.resolution)

        blur_maps = []
        # for i in range(len(output_names)):
        #     gt = output_imgs[..., i*3:(i+1)*3]
        #     gt = cv2.resize(gt, None, fx=0.5, fy=0.5)
        #     map = self.blur_map(input_imgs, gt)
        #     blur_maps.append(map)
        #
        # blur_maps = np.stack(blur_maps, axis=0)
        input_imgs = np.transpose(input_imgs, axes=(2, 0, 1))
        output_imgs = np.transpose(output_imgs, axes=(2, 0, 1))

        input_imgs = torch.from_numpy(input_imgs)
        output_imgs = torch.from_numpy(output_imgs)
        # blur_maps = torch.from_numpy(blur_maps)
        blur_maps = output_imgs
        if self.print_real_name:
            return input_imgs, output_imgs, blur_maps, input_names
        return input_imgs, output_imgs, blur_maps


def wrap_multi_channel_img(tensor_list, rgb=True, bits=8):
    """
    tensor_list: list，内在元素是 pytorch tensor [c,h,w]
    其中，若c是多通道图，则排列之
    """
    img_list = []
    for tensor in tensor_list:
        img = tensor.cpu().detach().numpy()
        img = np.transpose(img, axes=(1, 2, 0))
        sub_list = []
        if rgb:
            num = img.shape[-1]//3
            step = 3
        else:
            num = img.shape[-1]
            step = 1
        for i in range(num):
            tmp = img[:, :, step*i:step*(i+1)]
            sub_list.append(tmp)
        joint_img = np.concatenate(sub_list, axis=1)
        img_list.append(joint_img)
    assemble_img = np.concatenate(img_list, axis=0)
    assemble_img = np.clip(assemble_img, a_min=0, a_max=1)
    if bits == 8:
        assemble_img = np.uint8(assemble_img*255)
    elif bits == 16:
        assemble_img = np.uint16(assemble_img*(2**bits-1))
    if rgb:
        assemble_img = cv2.cvtColor(assemble_img, cv2.COLOR_RGB2BGR)
    # assemble_img = np.clip(assemble_img, a_min=0, a_max=255)
    return assemble_img


def wrap_multi_channel_array(tensor_list):
    img_list = []
    for tensor in tensor_list:
        img = tensor.cpu().detach().numpy()
        img = np.transpose(img, axes=(1, 2, 0))
        sub_list = []
        for i in range(img.shape[-1]):
            sub_img = img[..., i]
            sub_list.append(sub_img)
        joint_img = np.concatenate(sub_list, axis=1)
        img_list.append(joint_img)

    assemble_img = np.concatenate(img_list, axis=0).squeeze()
    # assemble_img = assemble_img.astype(np.int16)
    return assemble_img


def binary_sep(img, threshold=230):
    '''
    执行图像分割，分割背景与前景
    '''
    gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mask = gray < threshold
    imgBin = np.zeros_like(mask)
    imgBin[mask] = 1
    return imgBin


def path_checker(path):
    """
    检查目录是否存在，不存在，则创建
    """
    if not os.path.isdir(path):
        os.makedirs(path)
        print('目录不存在，已创建...')
    else:
        print('目录已存在')


if __name__ == '__main__':
    from torchvision import transforms
    import matplotlib.pyplot as plt
    data_root = '/mnt/diskarray/mjb/Projects/3DSR/data'

    # def __init__(self, data_root, mode, pattern, input_layer:list, out_layer:list, lr2hr=True, seed = 0):
    data_set = AttentionDataSet(data_root, 'train', 'all', [0], list(range(-2, 3)), lr2hr=True,
                                input_size=512, resolution=0.25, print_real_name=True)
    data_set.data_prefix = '/mnt/diskarray/mjb/Projects'
    # data_set = D3Dataset(data_root, 'train', 'all',[0], list(range(-5,6)), lr2hr=False)
    # data_set = AEDataset(data_root, 'train', 'random', [0], list(range(-5,6)), lr2hr=True, seed=1110,
                         # resolution=0.25, input_size=512)
    # data_set.data_prefix = '/mnt/diskarray/mjb/Projects'

    in_img, out_imgs, maps, names = data_set[4]
    # z = data_set.get_weight_map(in_img, out_imgs)
    transforms.ToPILImage()(in_img).show()
    transforms.ToPILImage()(out_imgs[0:3]).show()
    transforms.ToPILImage()(maps).show()
    # transforms.ToPILImage()(maps[0]).show()
    # plt.imshow(maps[0])
    plt.show()

    print('in_img:')
    print(in_img.dtype)
    print(in_img.shape)
    print('out_img:')
    print(out_imgs.dtype)
    print(out_imgs.shape)
    print('Depth mask:')
    print(maps.dtype)
    print(maps.shape)
    print(maps.mean())



