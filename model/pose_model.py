import torch
from torch.cuda.amp import GradScaler
import torch.nn as nn
from model.base_model import BaseModel
from model.networks import base_function,external_function
import model.networks as network
from util import task, util
import itertools
import torch.nn.functional as F
import os
import numpy as np
from script.train_metrics import Reconstruction_Metrics,LPIPS
from torch.cuda.amp import autocast

class Pose(BaseModel):

    def name(self):
        return "Pose-Guided Person Image Generation"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--attn_layer', action=util.StoreList, metavar="VAL1,VAL2...",
                            help="The number layers away from output layer")
        parser.add_argument('--kernel_size', action=util.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                            help="Kernel Size of Local Attention Block")

        parser.add_argument('--layers', type=int, default=3, help='number of layers in G')
        parser.add_argument('--netG', type=str, default='pose', help='The name of net Generator')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_rec_edge', type=float, default=2.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')
        parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='weight for perceptual L1 loss')
        parser.add_argument('--perceptual_layers', type=int, default=3,
                            help='index of vgg layer for extracting perceptual features.')
        parser.add_argument('--percep_is_l1', type=int, default=1, help='type of perceptual loss: l1 or l2')
        parser.add_argument('--lambda_correct', type=float, default=5.0,
                            help='weight for the Sampling Correctness loss')
        parser.add_argument('--lambda_style', type=float, default=500.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--lambda_cx', type=float, default=0.1, help='weight of CX loss')
        parser.add_argument('--lambda_regularization', type=float, default=0.0025,
                            help='weight for the affine regularization loss')

        parser.add_argument('--use_spect_g', action='store_false',
                            help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false',
                            help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # self.loss_names = ['app_gen', 'correctness_gen', 'content_gen', 'style_gen', 'regularization',
        #                    'ad_gen', 'dis_img_gen','cx_gen','l1_edge','perceptual_edge']
        self.loss_names = ['app_gen', 'correctness_gen', 'content_gen', 'style_gen', 'regularization',
                           'ad_gen', 'dis_img_gen','cx_gen']

        self.visual_names = ['input_P1', 'input_P2', 'img_gen', 'flow_fields', 'masks','input_EP1', 'input_EP2','fake_EP2']
        # self.visual_names = ['input_P1', 'input_P2', 'img_gen', 'flow_fields', 'masks']
        self.model_names = ['G', 'D']

        self.eval_metric_name =['psnr_score','lpips_score']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids) > 0 \
            else torch.FloatTensor
        # self.noise_var = nn.Parameter(torch.zeros(3), requires_grad=True).cuda()
        # define the generator
        self.net_G = network.define_g(opt, image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
                                      layers=opt.layers, num_blocks=2, use_spect=opt.use_spect_g,
                                      attn_layer=opt.attn_layer,
                                      norm='instance', activation='LeakyReLU', extractor_kz=opt.kernel_size)
        # input_nc = 1
        # output_nc = 1
        # self.edge_net = network.PoseEdgeNetGenerator(input_nc=input_nc, output_nc=output_nc,
        #                                            structure_nc=opt.structure_nc, ngf=64, use_spect=opt.use_spect_g)
        # define the discriminator
        if self.opt.dataset_mode == 'fashion':
            self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
        elif self.opt.dataset_mode == 'market':
            self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=3, use_spect=opt.use_spect_d)
        #     给流场上颜色
        self.flow2color = util.flow2color()

        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).to(opt.device)
            # self.GANloss = external_function.AdversarialLoss(opt.gan_mode)
            self.L1loss = torch.nn.L1Loss().to(opt.device)
            self.Correctness = external_function.PerceptualCorrectness().to(opt.device)
            # self.Correctness = external_function.PerceptualCorrectness()
            self.Regularization = external_function.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).to(opt.device)
            # self.Regularization = external_function.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size)
            self.Vggloss = external_function.VGGLoss().to(opt.device)

            self.CX_loss = external_function.CxLoss(sigma=0.5).to(opt.device)
            # self.Perceptualloss = external_function.L1_plus_perceptualLoss(opt.lambda_L1, opt.lambda_perceptual,
            #                                                                opt.perceptual_layers, self.gpu_ids,
            #                                                                opt.percep_is_l1)

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
        input_P1, input_BP1 ,input_EP1= input['P1'], input['BP1'] ,input['EP1']
        input_P2, input_BP2 ,input_EP2= input['P2'], input['BP2'],input['EP2']
        # self.input_BEP2, self.input_BEP2c1 = input['BEP2'], input['BEP2c1']
        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]
        # 将数据于CUDA中处理
        if len(self.gpu_ids) > 0:
            self.input_P1 = input_P1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_EP1 = input_EP1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_P2 = input_P2.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP2 = input_BP2.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_EP2 = input_EP2.cuda(self.gpu_ids[0], non_blocking=True)
            # self.input_BEP2 = self.input_BEP2.cuda(self.gpu_ids[0], non_blocking=True)
            # self.input_BEP2c1 = self.input_BEP2c1.cuda(self.gpu_ids[0], non_blocking=True)
        else:
            self.input_P1 = input_P1
            self.input_BP1 = input_BP1
            self.input_EP1 = input_EP1
            self.input_P2 = input_P2
            self.input_BP2 = input_BP2
            self.input_EP2 = input_EP2
        # print(input_BP1.shape)
        # self.input = input
        # input_P1, input_BP1 = input['P1'], input['BP1']
        # input_P2, input_BP2 = input['P2'], input['BP2']
        # # 将数据于CUDA中处理
        # if len(self.gpu_ids) > 0:
        #     self.input_P1 = input_P1.cuda(self.gpu_ids[0], non_blocking=True)
        #     self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], non_blocking=True)
        #     self.input_P2 = input_P2.cuda(self.gpu_ids[0], non_blocking=True)
        #     self.input_BP2 = input_BP2.cuda(self.gpu_ids[0], non_blocking=True)

        self.image_paths = []
        for i in range(self.input_P1.size(0)):
            # splitext（）分割路径，返回路径名和文件扩展名的元组
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_2_' + input['P2_path'][i])
    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid - 1
        flow = 2*flow/torch.as_tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_warp = F.grid_sample(source, grid).view(b, c, -1)
        return input_warp
    def test(self):
        """Forward function used in test time"""
        # img_gen, flow_fields, masks = self.net_G(self.input_P1, self.input_BP1, self.input_BP2)
        # img_gen = img_gen*(1-masks[-1])+ self.bilinear_warp(self.input_P1,flow_fields[-1])*(masks[-1])
        img_gen, flow_fields, masks ,fake_EP2= self.net_G(self.input_P1, self.input_BP1, self.input_BP2,self.input_EP1)
        # fake_EP2 = self.net_G(self.input_P1, self.input_BP1, self.input_BP2,self.input_EP1)
        self.save_results(img_gen, data_name='vis')
        # if self.opt.save_input or self.opt.phase == 'test':
        #     result = torch.cat([self.input_P1, img_gen, self.input_P2], 3)
        #     # result_edge = torch.cat([self.input_EP1, fake_EP2, self.input_EP2], 3)
        #     self.save_results(result, data_name='all')
            # self.save_results(result_edge, data_name='alledge')
        if self.opt.save_input or self.opt.phase == 'val':
            self.save_results(self.input_P1, data_name='ref')
            self.save_results(self.input_P2, data_name='gt')
            result = torch.cat([self.input_P1, img_gen, self.input_P2], 3)
            result_edge = torch.cat([self.input_EP1, fake_EP2, self.input_EP2], 3)
            # print(result)
            # visualize_feature_map(result)
            self.save_results(result, data_name='all')
            self.save_results(result_edge, data_name='alledge')

    def forward(self):
        """Run forward processing to get the inputs"""
        self.img_gen, self.flow_fields, self.masks, self.fake_EP2 = self.net_G(self.input_P1, self.input_BP1, self.input_BP2, self.input_EP1)
        # self.fake_EP2 = self.net_G(self.input_P1, self.input_BP1, self.input_BP2, self.input_EP1)
        # self.img_gen = self.img_gen * (1 - self.masks[-1]) + self.bilinear_warp(self.input_P1, self.flow_fields[-1]) * (self.masks[-1])
        # self.img_gen = self.img_gen  + self.bilinear_warp(self.input_P1, self.flow_fields[-1])
        # self.img_gen, self.flow_fields, self.masks = self.net_G(self.input_P1, self.input_BP1, self.input_BP2)
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # added_noise = (torch.randn(real.shape[0], real.shape[3], real.shape[2], 1).cuda() * self.noise_var).transpose(1, 3)
        # real = real+added_noise
        # fake = fake+added_noise
        # Real
        D_real = netD(real)#[1,1,16,16]
        D_real_loss = self.GANloss(D_real, True, True)
        D_fake = netD(fake.detach())  # fake 切断反向传播
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty
        # scaler.scale(D_loss).backward()
        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.input_P2, self.img_gen)
        # return self.loss_dis_img_gen

    def backward_G(self):
        """Calculate training loss for the generator"""
        # scaler = GradScaler()
        # Calculate Edge loss
        # self.loss_l1_edge = self.L1loss(self.fake_EP2, self.input_EP2) * self.opt.lambda_rec_edge
        # self.loss_perceptual_edge = self.Perceptualloss(self.fake_EP2, self.input_EP2)
        # added_noise = (torch.randn(self.img_gen.shape[0], self.img_gen.shape[3], self.img_gen.shape[2], 1).cuda() * self.noise_var).transpose(1,
        #                                                                                                               3)
        # self.img_gen = self.img_gen + added_noise
        # self.input_P2 = self.input_P2 + added_noise

        # Calculate l1 loss
        loss_app_gen = self.L1loss(self.img_gen, self.input_P2)
        self.loss_app_gen = loss_app_gen * self.opt.lambda_rec

        # Calculate Sampling Correctness Loss
        loss_correctness_gen = self.Correctness(self.input_P2, self.input_P1, self.flow_fields, self.opt.attn_layer)
        self.loss_correctness_gen = loss_correctness_gen * self.opt.lambda_correct

        # Calculate GAN loss
        base_function._freeze(self.net_D)
        D_fake = self.net_D(self.img_gen)
        self.loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # Calculate regularization term
        loss_regularization = self.Regularization(self.flow_fields)
        self.loss_regularization = loss_regularization * self.opt.lambda_regularization

        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(self.img_gen, self.input_P2)
        self.loss_style_gen = loss_style_gen * self.opt.lambda_style
        self.loss_content_gen = loss_content_gen * self.opt.lambda_content
        # Calculate CX_loss
        loss_cx = self.CX_loss(self.input_P2,self.img_gen)
        self.loss_cx_gen = loss_cx * self.opt.lambda_cx


        total_loss = torch.as_tensor([0.0]).float().cuda()

        for name in self.loss_names:
            if name != 'dis_img_gen':
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()
        # scaler.scale(total_loss).backward()

    def metrics(self):
        rec = Reconstruction_Metrics()
        lpips = LPIPS()
        img_gen = np.array([util.tensor2im(self.img_gen.data).astype(np.float32)])
        input_P2 = np.array([util.tensor2im(self.input_P2.data).astype(np.float32)])
        self.eval_psnr_score = rec.calculate_psnr_image(img_gen, input_P2)
        self.eval_lpips_score = lpips.calculate_lpips_image(img_gen, input_P2)
    # 先调用forward再D   backward， 更新D之后； 再G  backward， 再更新G
    def optimize_parameters(self):
        """update network weights"""
        # scaler = GradScaler()
        # with autocast():
        self.forward()
        self.optimizer_D.zero_grad(set_to_none=True)
        self.backward_D()

        self.optimizer_D.step()
        # scaler.step(self.optimizer_D)
        # scaler.update()

        self.optimizer_G.zero_grad(set_to_none=True)

        # with autocast:
        self.backward_G()
        self.optimizer_G.step()
        # scaler.step(self.optimizer_G)
        # scaler.update()
        self.metrics()
