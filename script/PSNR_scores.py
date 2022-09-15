gfla = 'D:/代码/GFLA/other_generated_image/GFLA/test'
SCA = '../other_generated_image/SCA_GAN/test'
pise = '../other_generated_image/pise/latest'
POSE2pose = 'D:/代码/GFLA/other_generated_image/pose2pose/fake_A'
patn = '../eval_results'
pg = '../eval_results'

our_f = '../dataset/fashion/test_256'  # in

# tar_f = '/home/jins/my_grade1/ADGAN/generated_imgs/img_tar'
tar_f = './test_results/5000000'

import os
import cv2
from skimage.io import imread, imsave
import numpy as np
from skimage.measure import compare_ssim

def addBounding(image, bound=40):
    h, w, c = image.shape
    image_bound = np.ones((h, w + bound * 2, c)) * 255
    image_bound = image_bound.astype(np.uint8)
    image_bound[:, bound:bound + w] = image

    return image_bound


def psnr_and_ssim(orig, target, max_value):
    """Numpy implementation of PSNR."""
    mse = ((orig - target) ** 2).mean(axis=(-3, -2, -1))
    psnr = 20 * np.log10(max_value) - 10 * np.log10(mse)
    ssim = compare_ssim(orig, target, win_size=11, data_range=255, multichannel=True)
    return ssim, psnr


def _main(truth_path, generate_path):
    truth = cv2.imread(truth_path)
    # print(img2_path)
    generate = cv2.imread(generate_path)
    print(generate_path)
    # if generate.shape[1] == 256:
    #     generate = generate[:, 40:-40, :]
    mse,psnr = psnr_and_ssim(truth, generate, 255)
    return mse, psnr


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--folder", type=str, default="ou")
    args = parser.parse_args()

    score_list = []
    ssim_score = []
    a = os.listdir(tar_f)
    for i in a:
        genrate = os.path.join(tar_f, i)
        name = i
        if args.folder.startswith('ou'):
            folder = our_f
        elif args.folder.startswith('bi'):
            folder = bi
        elif args.folder.startswith('gf'):
            folder = gfla
        elif args.folder.startswith('ad'):
            folder = adgan
        elif args.folder.startswith('xing'):
            folder = xing
        elif args.folder.startswith('patn'):
            folder = patn
        elif args.folder.startswith('pg'):
            folder = pg
        name = name.split('_2_')[-1]
        name = name.split('_vis')[0] + '.jpg'
        # name = name.split('___')[-1]
        # name = name.split('_vis')[0]
        truth = os.path.join(folder, name)
        # os.rename(img1, os.path.join(folder, i))
        ssim, psnr = _main(truth, genrate)
        score_list.append(psnr)
        ssim_score.append(ssim)
    print(len(score_list))
    print(args.folder + ' psnr : ' + str(np.mean(score_list)) + '  ssim : ' + str(np.mean(ssim_score)))

