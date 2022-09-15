import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
from model.base_model import BaseModel
from model.networks import base_function,external_function
import model.networks as network
from util import task, util
import itertools

import os

class poseEdgenet(BaseModel):

    def name(self):
        return "Pose-Guided Person Image Generation"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--attn_layer', action=util.StoreList, metavar="VAL1,VAL2...",
                            help="The number layers away from output layer")
        parser.add_argument('--kernel_size', action=util.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                            help="Kernel Size of Local Attention Block")

        parser.add_argument('--layers', type=int, default=3, help='number of layers in G')
        parser.add_argument('--netG', type=str, default='poseEdgenet', help='The name of net Generator')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')

        parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=5.0, help='weight for generation loss')
        parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')
        parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='weight for perceptual L1 loss')
        parser.add_argument('--perceptual_layers', type=int, default=3,help='index of vgg layer for extracting perceptual features.')
        parser.add_argument('--percep_is_l1', type=int, default=1, help='type of perceptual loss: l1 or l2')

        parser.add_argument('--use_spect_g', action='store_false',help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false', help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['app_gen', 'ad_gen', 'dis_img_gen', 'perceptual']

        self.visual_names = ['input_EP1', 'input_EP2', 'img_edge_gen','input_BP1','input_BP2']
        self.model_names = ['G', 'D']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids) > 0 \
            else torch.FloatTensor

        # define the generator
        input_nc=1
        output_nc=1
        self.net_G = network.define_g(opt,input_nc=input_nc,output_nc=output_nc,structure_nc=opt.structure_nc,ngf=64,use_spect=opt.use_spect_g)
        # self.edge_net = network.EdgeContentEncoder(input_dim=input_nc,output_dim=output_nc,structure_nc=opt.structure_nc,ngf=64,use_spect=opt.use_spect_g)
        # 修改为SCA中的posedge
        # input_nc = 18+1+18
        # self.net_G = network.define_g(opt,input_nc=input_nc)
        # define the discriminator
        if self.opt.dataset_mode == 'fashion':
            self.net_D = network.define_d(opt,input_nc=1, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
        elif self.opt.dataset_mode == 'market':
            self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=3, use_spect=opt.use_spect_d)

        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).to(opt.device)
            # self.GANloss = external_function.AdversarialLoss(opt.gan_mode)

            self.L1loss = torch.nn.L1Loss()

            # self.Vggloss = external_function.VGGLoss().to(opt.device)
            # self.Vggloss = external_function.VGGLoss()

            self.Perceptualloss = external_function.L1_plus_perceptualLoss(opt.lambda_L1, opt.lambda_perceptual, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_G.parameters())),
                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_D = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_D.parameters())),
                lr=opt.lr * opt.ratio_g2d, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_D)

        # load the pre-trained model and schedulers
        self.setup(opt)

    def set_input(self, input):
        # move to GPU and change data types
        self.input = input
        input_P1,input_EP1, input_BP1 = input['P1'],input['EP1'], input['BP1']
        input_P2,input_EP2, input_BP2 = input['P2'],input['EP2'], input['BP2']

        # self.input_BEP2, self.input_BEP2c1 = input['BEP2'], input['BEP2c1']
        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]
        # 将数据于CUDA中处理
        if len(self.gpu_ids) > 0:
            self.input_P1 = input_P1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_EP1 = input_EP1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_P2 = input_P2.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_EP2 = input_EP2.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP2 = input_BP2.cuda(self.gpu_ids[0], non_blocking=True)
            # self.input_BEP2 = self.input_BEP2.cuda(self.gpu_ids[0], non_blocking=True)
            # self.input_BEP2c1 = self.input_BEP2c1.cuda(self.gpu_ids[0], non_blocking=True)
        # print(input_BP1.shape)
        else:
            self.input_EP1 = input_EP1
            self.input_BP1 = input_BP1
            self.input_EP2 = input_EP2
            self.input_BP2 = input_BP2

        self.image_paths = []
        for i in range(self.input_EP1.size(0)):
            # splitext（）分割路径，返回路径名和文件扩展名的元组
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_2_' + input['P2_path'][i])
    def test(self):
        # real_map_AB = torch.cat((self.input_BP1, self.input_BP2), dim=1)
        # img_edge_gen = self.net_G(self.input_EP1, real_map_AB)
        # img_edge_gen = self.net_G(self.input_EP1, self.input_BP2, self.input_BEP2)
        img_edge_gen = self.net_G(self.input_EP1, self.input_BP1, self.input_BP2)
        # self.save_results(img_edge_gen, data_name='vis')
        result = torch.cat([self.input_EP1, img_edge_gen, self.input_EP2], 3)
        self.save_results(result, data_name='all')
    def forward(self):
        """Run forward processing to get the inputs"""
        # real_map_AB = torch.cat((self.input_BP1, self.input_BP2), dim=1)
        # self.img_edge_gen = self.net_G(self.input_EP1, real_map_AB)
        self.img_edge_gen = self.net_G(self.input_EP1, self.input_BP1, self.input_BP2)
        # self.img_edge_gen = self.net_G(self.input_EP1, self.input_BP2, self.input_BEP2)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        D_fake = netD(fake.detach())  # fake 切断反向传播
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.input_EP2, self.img_edge_gen)

    def backward_G(self):
        """Calculate training loss for the generator"""
        # Calculate l1 loss
        loss_app_gen = self.L1loss(self.img_edge_gen, self.input_EP2)
        self.loss_app_gen = loss_app_gen * self.opt.lambda_rec

        # Calculate GAN loss
        base_function._freeze(self.net_D)
        D_fake = self.net_D(self.img_edge_gen)
        self.loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # Calculate perceptual loss
        self.loss_perceptual = self.Perceptualloss(self.img_edge_gen, self.input_EP2)


        total_loss = 0

        for name in self.loss_names:
            if name != 'dis_img_gen':
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()

    def optimize_parameters(self):
        """update network weights"""
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
