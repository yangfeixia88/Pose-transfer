import  torch
from model.base_model import BaseModel
from model.networks import base_function,external_function
import model.networks as network
from util import util
import itertools
import os
class Poseinit(BaseModel):
    def name(self):
        return "Pose transfer"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--kernel_size',action=util.StoreDictKeyPair,metavar="KEY1=VAL1,KEY2=VAL2...",
                            help="Kernel Size of Local Attention Block")

        parser.add_argument('--netG',type=str,default='Poseinit',help='The Name of net Generator')
        parser.add_argument('--netD',type=str,default='res',help='The Name of net Discriminator')
        parser.add_argument('--init_type',type=str,default='orthogonal',help='Initial type')

        # if is_train
        parser.add_argument('--ratio_g2d',type=float,default=0.1,help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec',type=float,default=5.0,help='weight for image reconstructon loss')
        parser.add_argument('--lambda_g',type=float,default=2.0,help='weight for generation loss')
        parser.add_argument('--lambda_correct',type=float,default=5.0,help='weight for the Sampling Correctness loss')
        parser.add_argument('--lambda_style',type=float,default=500.0,help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content',type=float,default=0.5,help='weight for the VGG19  content loss')
        parser.add_argument('--lambda_regularization',type=float,default=0.0025,help='weight for the affine regularization loss')


        parser.add_argument('--use_spect_g', action='store_false',help='whether use spectral normalizaton in generator')
        parser.add_argument('--use_spect_d',action='store_false',help='whether use spectral normalization in discriminator')
        parser.add_argument('--save_input',action='store_false',help='whether save the input images when testing')

        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser

    def __init__(self,opt):
        BaseModel.__init__(self,opt)
        self.loss_names = ['app_gen','content_gen','style_gen',
                           'ad_gen','dis_img_gen']
        self.visual_names =['input_P1','input_P2','img_gen','input_BP1','input_BP2']
        self.model_names =[ 'G','D']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids) > 0 else torch.FloatTensor

        # define the generator
        self.net_G = network.define_g(opt,structure_nc=opt.structure_nc,ngf=64,use_spect=opt.use_spect_g,
                                      norm='instance',activation='LeakyReLU')

        if self.opt.dataset_mode == 'fashion':
            self.net_D = network.define_d(opt, ndf=32, img_f=128,layers=4,use_spect=opt.use_spect_d)
        elif self.opt.dataset_mode == 'market':
            self.net_D = network.define_d(opt, ndf=32,img_f=128, layers=3,use_spect=opt.use_spect_d)


        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).to(opt.device)
            self.L1loss = torch.nn.L1Loss()
            self.Correctness = external_function.PerceptualCorrectness().to(opt.device)
            self.Regularization = external_function.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).to(opt.device)
            self.Vggloss = external_function.VGGLoss().to(opt.device)

            # define the optimizer
            self.optimizers_G = torch.optim.Adam(itertools.chain(
                filter(lambda p:p.requires_grad,self.net_G.parameters())),
                lr=opt.lr,betas=(0.0,0.999)
            )
            self.optimizers.append(self.optimizers_G)

            self.optimizers_D = torch.optim.Adam(itertools.chain(
                filter(lambda  p:p.requires_grad,self.net_D.parameters())),
                lr=opt.lr, betas=(0.0,0.999))
            self.optimizers.append(self.optimizers_D)

        self.setup(opt)
    def set_input(self, input):
        self.input = input
        input_P1, input_BP1 = input['P1'],input['BP1']
        input_P2, input_BP2 = input['P2'],input['BP2']

        if len(self.gpu_ids) > 0:
            self.input_P1 = input_P1.cuda(self.gpu_ids[0],non_blocking=True)
            self.input_BP1 = input_BP1.cuda(self.gpu_ids[0],non_blocking=True)
            self.input_P2 = input_P2.cuda(self.gpu_ids[0],non_blocking=True)
            self.input_BP2 = input_BP2.cuda(self.gpu_ids[0],non_blocking=True)
        # print(input_BP1.shape)
        self.image_paths = []
        for i in range(self.input_P1.size(0)):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0]+'_2_'+input['P2_path'][i])

    def test(self):
        img_gen =self.net_G(self.input_P1,self.input_BP1,self.input_BP2)
        self.save_results(img_gen, data_name='vis')
        if self.opt.save_input or input.opt.phase == 'val':
            self.save_results(self.input_P1,data_name='ref')
            self.save_results(self.input_P2,data_name='gt')
            result = torch.cat([self.input_P1,img_gen,self.input_P2],3)
            self.save_results(result, data_name='all')

    def forward(self):
        self.img_gen = self.net_G(self.input_P1,self.input_BP1,self.input_BP2)

    def backward_D_basic(self,netD, real, fake):

        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake,False, True)
        D_loss = (D_real_loss + D_fake_loss) * 0.5

        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        D_loss.backward()

        return D_loss

    def backward_D(self):
        base_function._unfreeze(self.net_D)
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.input_P2,self.img_gen)

    def backward_G(self):
        loss_app_gen = self.L1loss(self.img_gen, self.input_P2)
        self.loss_app_gen = loss_app_gen * self.opt.lambda_rec

        base_function._freeze(self.net_D)
        D_fake = self.net_D(self.img_gen)
        self.loss_ad_gen = self.GANloss(D_fake,True,False) *self.opt.lambda_g

        loss_conten_gen, loss_style_gen = self.Vggloss(self.img_gen,self.input_P2)
        self.loss_style_gen = loss_style_gen * self.opt.lambda_style
        self.loss_content_gen = loss_conten_gen * self.opt.lambda_content

        total_loss = 0

        for name in self.loss_names:
            if name!='dis_img_gen':
                total_loss += getattr(self,"loss_" +name)
        total_loss.backward()


    def optimize_parameters(self):
        self.forward()

        self.optimizers_D.zero_grad()
        self.backward_D()
        self.optimizers_D.step()

        self.optimizers_G.zero_grad()
        self.backward_G()
        self.optimizers_G.step()