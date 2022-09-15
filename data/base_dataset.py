import argparse
import os

import PIL
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from scipy import signal
import random
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from util import pose_utils
from PIL import Image
import pandas as pd
import torch
import math
import numbers


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--angle', type=float, default=False)
        parser.add_argument('--shift', type=float, default=False)
        parser.add_argument('--scale', type=float, default=False)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.image_dir, self.bone_file, self.name_pairs = self.get_paths(opt)
        self.dir_E = os.path.join(opt.dataroot, opt.phase + 'E')
        # print("image_dir",self.image_dir)
        # print("bone_file",self.bone_file)
        # print("name_pairs",self.name_pairs)
        size = len(self.name_pairs)
        self.dataset_size = size

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size

        transform_list = []
        # transform_list.append(transforms.Resize(size=self.load_size))
        # 数据归一化到【0,1】，（是将数据除以255）
        transform_list.append(transforms.ToTensor())
        rgb = transform_list.copy()
        edge = transform_list.copy()
        # 通过Normalize计算过后，将数据归一化到[-1, 1]
        # transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        # # 串联多个图片变换的操作
        # self.trans = transforms.Compose(transform_list)

        rgb += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        edge += [ transforms.Scale(self.load_size, Image.BICUBIC),
            transforms.Normalize(mean=[0.5], std=[0.5]), ]
        # 串联多个图片变换的操作
        self.trans_rgb = transforms.Compose(rgb)

        self.trans_edge = transforms.Compose(edge)
        # sep=':'  制定分隔符
        self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of MarkovAttnDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths


    def __getitem__(self, index):
        P1_name, P2_name = self.name_pairs[index]
        P1_path = os.path.join(self.image_dir, P1_name)  # person 1
        P2_path = os.path.join(self.image_dir, P2_name)  # person 2
        EP1_path = os.path.join(self.dir_E, 'diff',
                                P1_name.replace('.jpg', '.png').replace('fashion', ''))  # edge of person 1
        EP2_path = os.path.join(self.dir_E, 'diff',
                                P2_name.replace('.jpg', '.png').replace('fashion', ''))  # edge of person 2

        if self.opt.phase == 'test':
            EP1_path = EP1_path.replace('testE', 'trainE')
            EP2_path = EP2_path.replace('testE', 'trainE')
        elif self.opt.phase == 'val':
            EP1_path = EP1_path.replace('valE', 'trainE')
            EP2_path = EP2_path.replace('valE', 'trainE')

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')
        P1_edge = Image.open(EP1_path).convert('L')
        P2_edge = Image.open(EP2_path).convert('L')
        # PIL.ImageOps.invert 实现二值图像黑白反转
        person_edge = np.array(P1_edge.copy())
        P1_edge = PIL.ImageOps.invert(P1_edge)
        P2_edge = PIL.ImageOps.invert(P2_edge)
        P1_img = F.resize(P1_img, self.load_size)
        P2_img = F.resize(P2_img, self.load_size)
        # P1_edge = F.resize(P1_edge, self.load_size)
        # P2_edge = F.resize(P2_edge, self.load_size)

        angle, shift, scale = self.getRandomAffineParam()
        P1_img = F.affine(P1_img, angle=angle, translate=shift, scale=scale, shear=0, fillcolor=(128, 128, 128))
        center = (P1_img.size[0] * 0.5 + 0.5, P1_img.size[1] * 0.5 + 0.5)
        affine_matrix = self.get_affine_matrix(center=center, angle=angle, translate=shift, scale=scale, shear=0)
        BP1 = self.obtain_bone(P1_name, affine_matrix)
        P1 = self.trans_rgb(P1_img)
        EP1 = self.trans_edge(P1_edge)

        angle, shift, scale = self.getRandomAffineParam()
        angle, shift, scale = angle * 0.2, (
        shift[0] * 0.5, shift[1] * 0.5), 1  # Reduce the deform parameters of the generated image
        P2_img = F.affine(P2_img, angle=angle, translate=shift, scale=scale, shear=0, fillcolor=(128, 128, 128))
        center = (P1_img.size[0] * 0.5 + 0.5, P1_img.size[1] * 0.5 + 0.5)
        affine_matrix = self.get_affine_matrix(center=center, angle=angle, translate=shift, scale=scale, shear=0)
        BP2 = self.obtain_bone(P2_name, affine_matrix)
        P2 = self.trans_rgb(P2_img)
        EP2 = self.trans_edge(P2_edge)


        BEP2, masked_edge_c1 = crop_shift(sp=BP1.copy(), tp=BP2.copy(), person=person_edge,
                                          radius=self.opt.crop_radius, mask=False)
        BEP2 = self.trans_edge(BEP2)
        # BEP2 = transforms.Scale(self.load_size, Image.BICUBIC)



        # masked_edge_c1 = F.resize(masked_edge_c1, self.load_size)
        masked_edge_c1 = self.trans_edge(masked_edge_c1)

        BP1 = torch.as_tensor(BP1)
        BP2 = torch.as_tensor(BP2)

        return {'P1': P1, 'BP1': BP1, 'EP1': EP1,'P2': P2, 'BP2': BP2,'EP2': EP2,
                'BEP2': BEP2, 'BEP2c1': masked_edge_c1,
                'P1_path': P1_name, 'P2_path': P2_name}

    def obtain_bone(self, name, affine_matrix):
        string = self.annotation_file.loc[name]
        array = pose_utils.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose = pose_utils.cords_to_map(array, self.load_size, self.opt.old_size, affine_matrix)
        pose = np.transpose(pose, (2, 0, 1))
        # pose = torch.Tensor(pose)
        return pose

    def __len__(self):
        return self.dataset_size

    def name(self):
        assert False, "A subclass of BaseDataset must override self.name"

    def getRandomAffineParam(self):
        if self.opt.angle is not False:
            angle = np.random.uniform(low=self.opt.angle[0], high=self.opt.angle[1])
        else:
            angle = 0
        if self.opt.scale is not False:
            scale = np.random.uniform(low=self.opt.scale[0], high=self.opt.scale[1])
        else:
            scale = 1
        if self.opt.shift is not False:
            shift_x = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
            shift_y = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
        else:
            shift_x = 0
            shift_y = 0
        return angle, (shift_x, shift_y), scale

    def get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        # code from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#affine
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RSS is rotation with scale and shear matrix
        #       RSS(a, scale, shear) = [ cos(a + shear_y)*scale    -sin(a + shear_x)*scale     0]
        #                              [ sin(a + shear_y)*scale    cos(a + shear_x)*scale     0]
        #                              [     0                  0          1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        if isinstance(shear, (tuple, list)) and len(shear) == 2:
            shear = [math.radians(s) for s in shear]
        elif isinstance(shear, numbers.Number):
            shear = math.radians(shear)
            shear = [shear, 0]
        else:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " +
                "two values. Got {}".format(shear))
        scale = 1.0 / scale

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
            math.sin(angle + shear[0]) * math.sin(angle + shear[1])
        matrix = [
            math.cos(angle + shear[0]), math.sin(angle + shear[0]), 0,
            -math.sin(angle + shear[1]), math.cos(angle + shear[1]), 0
        ]
        matrix = [scale / d * m for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]
        return matrix

    def get_affine_matrix(self, center, angle, translate, scale, shear):
        matrix_inv = self.get_inverse_affine_matrix(center, angle, translate, scale, shear)

        matrix_inv = np.matrix(matrix_inv).reshape(2, 3)
        pad = np.matrix([0, 0, 1])
        matrix_inv = np.concatenate((matrix_inv, pad), 0)
        matrix = np.linalg.inv(matrix_inv)
        return matrix
def map_to_cord(pose_map, threshold=0.1):
    all_peaks = [[] for i in range(18)]
    pose_map = pose_map.transpose(1,2,0)
    pose_map = pose_map[..., :18]

    y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis = (0, 1)),
                                     pose_map > threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(18):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(-1)
            y_values.append(-1)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)
def GaussainKernel(kernlen=10, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d
def crop_shift(sp, tp, person, radius=10, size=(256, 256), mask=True):
    s_cord = map_to_cord(sp) #(18,256,256) >
    t_cord = map_to_cord(tp)
    r = radius
    plates = np.ones((size[0], size[1], 18), dtype='float32')
    plate = np.ones((size[0], size[1]), dtype='float32')
    for i in range(18):
        sx = s_cord[i][1]
        sy = s_cord[i][0]
        tx = t_cord[i][1]
        ty = t_cord[i][0]
        if not (sx < 0 or tx < 0):
            # source crop
            sy_start = 0 if sy - r < 0 else sy - r
            sy_end = size[0] if sy + r > size[0] else sy + r
            sx_start = 0 if sx - r < 0 else sx - r
            sx_end = size[1] if sx + r > size[1] else sx + r

            crop_height = sy_end - sy_start
            crop_width = sx_end - sx_start

            ty_start = 0 if (ty - int((crop_height)/2)) < 0 else (ty - int((crop_height)/2))
            ty_end = size[0] if (ty_start+crop_height) > size[0] else (ty_start+crop_height)
            tx_start = 0 if (tx - int((crop_width)/2)) < 0 else (tx - int((crop_width)/2))
            tx_end = size[1] if (tx_start + crop_width) > size[1] else (tx_start + crop_width)

            # target crop
            crop = person[sy_start:sy_end, sx_start:sx_end]

            if mask:
                M = GaussainKernel(kernlen=radius*2, std=4)
                M = np.resize(M, (crop.shape))
                crop = crop * M
            crop = np.resize(crop, ((ty_end-ty_start), (tx_end-tx_start)))
            # if ty_end - ty_start < crop.shape[0] and tx_end - tx_start < crop.shape[1]:
            #     plates[:, :, i][ty_start:ty_end, tx_start:tx_end] = crop
            #     plate[ty_start:ty_end, tx_start:tx_end] = crop
            # elif ty_end - ty_start < crop.shape[0]:
            #     pass
            # elif tx_end - tx_start < crop.shape[1]:
            #     pass
            # else:
            plates[:, :, i][ty_start:ty_end, tx_start:tx_end] = crop
            plate[ty_start:ty_end, tx_start:tx_end] = crop
            # plates[:, :, i][ty - r:(ty + r), tx - r:(tx + r)] = crop
            # plate[ty - r:(ty + r), tx - r:(tx + r)] = crop

    return 255.0-plates, 255.0-plate

if __name__ == '__main__':
    # from options.test_options import TestOptions
    # import data as Dataset
    base = BaseDataset()
    parser = argparse.ArgumentParser(description='show  all  statics')
    parser.add_argument('--dataroot', help='root path', type=str)
    parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
    args = parser.parse_args()
    for arg in vars(args):
        print('[%s] = ' % arg, getattr(args, arg))
    opt = args

    base.initialize(opt)
    # image_dir, boneslst, name_pairs = base.get_paths(opt)
    list = base.__getitem__(0)
    print(list)