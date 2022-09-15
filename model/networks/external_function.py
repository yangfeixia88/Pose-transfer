import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from model.networks.resample2d_package.resample2d import Resample2d
from model.networks.block_extractor.block_extractor import BlockExtractor
from model.networks.local_attn_reshape.local_attn_reshape import LocalAttnReshape

import numpy as np

from util import util


class MultiAffineRegularizationLoss(nn.Module):
    def __init__(self, kz_dic):
        super(MultiAffineRegularizationLoss, self).__init__()
        self.kz_dic=kz_dic
        self.method_dic={}
        for key in kz_dic:
            instance = AffineRegularizationLoss(kz_dic[key])
            self.method_dic[key] = instance
        self.layers = sorted(kz_dic, reverse=True)

    def __call__(self, flow_fields):
        loss = 0
        for i in range(len(flow_fields)):
            method = self.method_dic[self.layers[i]]
            loss += method(flow_fields[i])
        return loss

class AffineRegularizationLoss(nn.Module):
    ''' docstring for AffineRegularizationLoss'''
    def __init__(self, kz):
        super(AffineRegularizationLoss, self).__init__()
        self.kz = kz
        self.criterion = torch.nn.L1Loss()
        self.extractor = BlockExtractor(kernel_size=kz)
        self.reshape = LocalAttnReshape()

        temp = np.arange(kz)
        A = np.ones([kz*kz, 3])
        A[:, 0] = temp.repeat(kz)
        A[:, 1] = temp.repeat(kz).reshape((kz,kz)).transpose().reshape(kz**2)
        AH = A.transpose()
        k = np.dot(A, np.dot(np.linalg.inv(np.dot(AH, A)), AH)) - np.identity(kz**2)
        self.kernel = np.dot(k.transpose(), k)
        self.kernel = torch.from_numpy(self.kernel).unsqueeze(1).view(kz**2, kz, kz).unsqueeze(1)

    def __call__(self, flow_fields):
        grid = self.flow2grid(flow_fields)

        grid_x = grid[:, 0, :, :].unsqueeze(1)
        grid_y = grid[:, 1, :, :].unsqueeze(1)
        weights = self.kernel.type_as(flow_fields)
        loss_x = self.calculate_loss(grid_x, weights)
        loss_y = self.calculate_loss(grid_y, weights)
        return loss_x+loss_y


    def calculate_loss(self, grid, weights):
        results = nn.functional.conv2d(grid, weights)
        b, c, h, w = results.size()
        kernel_new = self.reshape(results,self.kz)
        f = torch.zeros(b, 2, h, w).type_as(kernel_new) + float(int(self.kz/2))
        grid_H = self.extractor(grid, f)
        result = torch.nn.functional.avg_pool2d(grid_H*kernel_new, self.kz, self.kz)
        loss = torch.mean(result)*self.kz**2
        return loss

    def flow2grid(self, flow_field):
        b,c,h,w = flow_field.size()
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(flow_field).float()
        y = torch.arange(w).view(-1, 1).expand(-1, w).type_as(flow_field).float()
        grid = torch.stack([x,y],dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        return flow_field+grid

class AdversarialLoss(nn.Module):


    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            # 计算目标值与预测值之间的二进制交叉熵损失函数
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            # 均方损失函数
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()
        elif type == 'style_gan2':
            self.criterion = nn.softplus()
        # elif type == '':
        #     self.criterion == nn.

    def __call__(self, outputs, is_real, for_dis=None):
        if self.type == 'hinge':
            if for_dis:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class VGGLoss(nn.Module):
    '''
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    '''


    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(VGGLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f =x.view(b, ch, w * h)
        f_T = f.transpose(1,2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def __call__(self,x, y):
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return content_loss, style_loss

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss

class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss

class PerceptualCorrectness(nn.Module):
    r"""

    """

    def __init__(self, layer=['rel1_1','relu2_1','relu3_1','relu4_1']):
        super(PerceptualCorrectness, self).__init__()
        self.add_module('vgg', VGG19())
        self.layer = layer
        self.eps=1e-8
        self.resample = Resample2d(4, 1, sigma=2)

    def __call__(self, target, source, flow_list, used_layers,  mask=None, use_bilinear_sampling=False):
        used_layers=sorted(used_layers, reverse=True)
        # self.target=target
        # self.source=source
        # Let vs and vt denote the features generated by a specific layer of VGG19.
        self.target_vgg, self.source_vgg = self.vgg(target), self.vgg(source)
        loss = 0
        for i in range(len(flow_list)):
              loss += self.calculate_loss(flow_list[i],self.layer[used_layers[i]],mask, use_bilinear_sampling)

        return loss

    def calculate_loss(self, flow, layer, mask=None, use_bilinear_sampling=False):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        [b, c, h, w] = target_vgg.shape

        # maps = F.interpolate(maps, [h,w]).view(b,-1)  将flow上下采样到指定的大小[h,w]
        flow = F.interpolate(flow, [h,w])

        target_all = target_vgg.view(b, c, -1)                      #[b C N2]
        source_all = source_vgg.view(b, c, -1).transpose(1,2)       #[b N2 C]


        source_norm = source_all/(source_all.norm(dim=2, keepdim=True)+self.eps)
        target_norm = target_all/(target_all.norm(dim=1, keepdim=True)+self.eps)
        try:
            correction = torch.bmm(source_norm, target_norm)                       #[b N2 N2]
        except:
            print("An exception occurred")
            print(source_norm.shape)
            print(target_norm.shape)
        (correction_max,max_indices) = torch.max(correction, dim=1)  #维度为1 的correction的最大值  max_indices为索引值

        # interple with bilinear sampling
        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow).view(b, c, -1)
        else:
            input_sample = self.resample(source_vgg, flow).view(b, c, -1)    #这里的值都是0

        correction_sample = F.cosine_similarity(input_sample, target_all)    #[b 1 N2]
        loss_map = torch.exp(-correction_sample/(correction_max+self.eps))  #论文中的采样正确行损失的公式  这里的值都是1
        if mask is None:
            loss = torch.mean(loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))
        else:
            mask=F.interpolate(mask, size=(target_vgg.size(2), target_vgg.size(3)))
            mask=mask.view(-1, target_vgg.size(2)*target_vgg.size(3))
            loss_map = loss_map - torch.exp(torch.tensor(-1).type_as(loss_map))
            loss = torch.sum(mask * loss_map)/(torch.sum(mask)+self.eps)

        # print(correction_sample[0,2076:2082])
        # print(correction_max[0,2076:2082])
        # coor_x = [32,32]
        # coor = max_indices[0,32+32*64]
        # coor_y = [int(coor%64), int(coor/64)]
        # source = F.interpolate(self.source, [64,64])
        # target = F.interpolate(self.target, [64,64])
        # source_i = source[0]
        # target_i = target[0]
        #
        # source_i = source_i.view(3, -1)
        # source_i[:,coor]=-1
        # source_i[0,coor]=1
        # source_i = source_i.view(3,64,64)
        # target_i[:,32,32]=-1
        # target_i[0,32,32]=1
        # lists = str(int(torch.rand(1)*100))
        # img_numpy = util.tensor2im(source_i.data)
        # util.save_image(img_numpy, 'source'+lists+'.png')
        # img_numpy = util.tensor2im(target_i.data)
        # util.save_image(img_numpy, 'target'+lists+'.png')
        return loss

    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid - 1
        flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid).view(b, c, -1)
        return input_sample

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])
        for x in range(2, 4):
            self.relu1_2.add_module((str(x)), features[x])
        for x in range(4,7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out

class L1_plus_perceptualLoss(nn.Module):
    def __init__(self, lambda_L1, lambda_perceptual, perceptual_layers, gpu_ids, percep_is_l1):
        super(L1_plus_perceptualLoss, self).__init__()

        self.lambda_L1 = lambda_L1
        self.lambda_perceptual = lambda_perceptual
        self.gpu_ids = gpu_ids

        self.percep_is_l1 = percep_is_l1

        # vgg = models.vgg19(pretrained=True).features
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('./dataset/fashion/vgg19-dcbb9e9d.pth'))
        vgg = vgg19.features


        self.vgg_submodel = nn.Sequential()
        for i,layer in enumerate(list(vgg)):
            self.vgg_submodel.add_module(str(i),layer)
            if i == perceptual_layers:
                break
        if len(gpu_ids) > 1:
            self.vgg_submodel = torch.nn.DataParallel(self.vgg_submodel, device_ids=gpu_ids)
        self.vgg_submodel = self.vgg_submodel.cuda()
        #self.vgg_submodel = torch.nn.DataParallel(self.vgg_submodel, device_ids=gpu_ids).cuda()

        # print(self.vgg_submodel)

    def forward(self, inputs, targets):
        if self.lambda_L1 == 0 and self.lambda_perceptual == 0:
            return Variable(torch.zeros(1)).cuda(), Variable(torch.zeros(1)), Variable(torch.zeros(1))
        # normal L1
        loss_l1 = F.l1_loss(inputs, targets) * self.lambda_L1

        # perceptual L1
        mean = torch.FloatTensor(3)
        mean[0] = 0.485
        mean[1] = 0.456
        mean[2] = 0.406
        mean = Variable(mean)
        mean = mean.view(1, 3, 1, 1).cuda()

        std = torch.FloatTensor(3)
        std[0] = 0.229
        std[1] = 0.224
        std[2] = 0.225
        std = Variable(std)
        std = std.view(1, 3, 1, 1).cuda()

        fake_p2_norm = (inputs + 1)/2 # [-1, 1] => [0, 1]
        fake_p2_norm = (fake_p2_norm - mean)/std

        input_p2_norm = (targets + 1)/2 # [-1, 1] => [0, 1]
        input_p2_norm = (input_p2_norm - mean)/std


        fake_p2_norm = self.vgg_submodel(fake_p2_norm)
        input_p2_norm = self.vgg_submodel(input_p2_norm)
        input_p2_norm_no_grad = input_p2_norm.detach()

        if self.percep_is_l1 == 1:
            # use l1 for perceptual loss
            loss_perceptual = F.l1_loss(fake_p2_norm, input_p2_norm_no_grad) * self.lambda_perceptual
        else:
            # use l2 for perceptual loss
            loss_perceptual = F.mse_loss(fake_p2_norm, input_p2_norm_no_grad) * self.lambda_perceptual

        loss = loss_l1 + loss_perceptual

        # return loss, loss_l1, loss_perceptual
        return loss



class CxLoss(nn.Module):

    def __init__(self,sigma):
        super(CxLoss, self).__init__()
        self.CX_Loss0 =CXLoss0(sigma)
        self.vgg = VGG()
        self.vgg.load_state_dict(torch.load('dataset/fashion/vgg_conv.pth'))
        for param in self.vgg.parameters():
            param.requires_grad = False
    def forward(self,input_P2,img_gen):
        style_layer = ['r32','r42']
        vgg_style = self.vgg(input_P2,style_layer)
        vgg_fake = self.vgg(img_gen,style_layer)
        loss =0
        for i ,val in enumerate(vgg_fake):
            loss += self.CX_Loss0(vgg_style[i],vgg_fake[i])
        return loss
class CXLoss0(nn.Module):

    def __init__(self, sigma=0.1, b=1.0, similarity="consine"):
        super(CXLoss0, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCxHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''
        # NCHW
        # print(featureI.shape)

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            # See the torch document for functional.conv2d
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)

        CX = CX.max(dim=3)[0].max(dim=2)[0]
        CX = CX.mean(1)
        CX = -torch.log(CX)
        CX = torch.mean(CX)
        return CX
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]