import argparse
import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import pandas as pd
from util import pose_utils
import numpy as np
import torch

class FashionDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(old_size=(256, 176))
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)
        return parser


    # 从本地拿到图片和结构图
    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
        pairLst = os.path.join(root, 'fasion-pairs-%s.csv' % phase)
        # 获取图片对的名称
        name_pairs = self.init_categories(pairLst)
        #图片的路径
        image_dir = os.path.join(root, '%s' % phase)
        # 获取骨架信息
        bonesLst = os.path.join(root, 'fasion-annotation-%s.csv' % phase)
        return image_dir, bonesLst, name_pairs

    # 下载图片对
    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')
        return pairs

    def name(self):
        return "FashionDataset"

if __name__=='__main__':
    print("==========实例化===========")
    fashionD = FashionDataset()

    parser = argparse.ArgumentParser(description='show  all  statics')
    parser.add_argument('--dataroot', help='root path', type=str)
    # parser.add_argument('--dataset_mode', help='Path to output data', type=str)
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    # parser.add_argument('--name', help='name of the experiment', type=str)
    # parser.add_argument('--calculate_mask', action='store_true')
    args = parser.parse_args()
    for arg in vars(args):
        print('[%s] = ' % arg , getattr(args,arg))
    opt = args
    image_dir,boneslst,name_pairs =fashionD.get_paths(opt)
    i = 0
    for img in name_pairs:

        if(i<len(name_pairs)):
            i=i+1
            print("第%d个图片的信息:"%i,img)

