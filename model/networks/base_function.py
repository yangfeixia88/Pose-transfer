import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler

import torch.nn.functional as F


from model.networks.resample2d_package.resample2d import Resample2d

from model.networks.block_extractor.block_extractor import BlockExtractor
from model.networks.local_attn_reshape.local_attn_reshape import LocalAttnReshape
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

# from util.util import muti_gpus_load_dict

# def visi(flow_field,image):
#     [b, _, h, w] = flow_field.size()
#     # 将图片采样到指定的大小
#     source_copy = torch.nn.functional.interpolate(image, (h, w))
#     # 用于指定横坐标x 的位置
#     x = torch.arange(w).view(1, -1).expand(h, -1).float()
#     # 用于指定纵坐标y的位置
#     y = torch.arange(h).view(-1, 1).expand(-1, w).float()
#     x = 2 * x / (w - 1) - 1
#     y = 2 * y / (h - 1) - 1
#     # 要创建一个[-1, 1]的网格backwarp_tenGrid
#     grid = torch.stack([x, y], dim=0).float().cuda()
#     grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
#     # 对flow进行缩放， u_scale = ( u * (2/(width-1)),  v * （ 2/(height-1)) )
#     flow_x = (2 * flow_field[:, 0, :, :] / (w - 1)).view(b, 1, h, w)
#     flow_y = (2 * flow_field[:, 1, :, :] / (h - 1)).view(b, 1, h, w)
#     flow = torch.cat((flow_x, flow_y), 1)
#     # 网格  grid = (backwarp_tenGrid + u_scale)
#     grid = (grid + flow).permute(0, 2, 3, 1)
#     # 双线性采样函数  实现图片的空间变形  ////  光流算法中的warp函数
#     warp = torch.nn.functional.grid_sample(source_copy, grid)
#     return warp
def get_norm_layer(norm_type='batch'):
    '''
    Get the normalization layer for the networks
    :param norm_type:
    :return:
    '''

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, momentum=0.1, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'adain':
        norm_layer = functools.partial(nn.ADAIN)
    elif norm_type == 'spade':
        norm_layer = functools.partial(SPADE, config_text='spadeinstance')
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %norm_type)

    if norm_type != 'none':
        norm_layer.__name__ = norm_type

    return norm_layer

def _freeze(*args):
    """freeze the network for forward process 冻结网络进行前向传播"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False

def _unfreeze(*args):
    """ unfreeze the network for parameter update 解冻网络进行参数更新"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer

def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+1+opt.iter_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def spectral_norm(module, use_spect=True):
    ''' use spectral normal layer to stable the training process'''

    if use_spect:
        return SpectralNorm(module)
    else:
        return module
def coord_conv(input_nc, output_nc, use_spect=False, use_coord=False, with_r=False, **kwargs):
    ''' 使用协调卷积层添加位置信息'''

    if use_coord:
        print("选择CoordConv")
        return CoordConv(input_nc, output_nc, with_r, use_spect, **kwargs)
    else:
        return spectral_norm(nn.Conv2d(input_nc,output_nc, **kwargs), use_spect)


class AddCoords(nn.Module):
    '''
    给张良加上坐标标注
    '''
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        '''

        :param x: shape (batch, channel, x_din, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        '''
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 -1
        yy_cahnnel = yy_cahnnel * 2 -1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):
    '''
    CoordConv operation
    '''
    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret

class EncoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(EncoderBlock, self).__init__()


        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_down)
        conv2 = coord_conv(output_nc, output_nc, use_spect, use_coord, **kwargs_fine)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(output_nc), nonlinearity, conv2)

    def forward(self,x):
        out = self.model(x)
        return out
class StyleEncoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(StyleEncoderBlock, self).__init__()


        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_down)
        conv2 = coord_conv(output_nc, output_nc, use_spect, use_coord, **kwargs_fine)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(output_nc), nonlinearity, conv2)

    def forward(self,x):
        out = self.model(x)
        return out

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
class ResBlock(nn.Module):
    ''' 定义残差块'''

    def __init__(self,input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 learnable_shortcut=False, use_spect=False, use_coord=False):
        super(ResBlock, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc
        self.learnable_shortcut = True if input_nc != output_nc else learnable_shortcut

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        conv2 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(hidden_nc), nonlinearity, conv2,)

        if self.learnable_shortcut:
            bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)
            self.shortcut = nn.Sequential(bypass,)


    def forward(self,x):

        if self.learnable_shortcut:
            out = self.model(x) + self.shortcut(x)
        else:
            out = self.model(x) + x
        return out

class ResBlocks(nn.Module):

    def __init__(self, num_blocks, input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d,
                 nonlinearity= nn.LeakyReLU(), learnable_shorcut=False, use_spect=False,use_coord=False):
        super(ResBlocks, self).__init__()
        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc

        self.model=[]
        if num_blocks == 1:
            self.model += [ResBlock(input_nc, output_nc, hidden_nc,
                                    norm_layer, nonlinearity, learnable_shorcut, use_spect, use_coord)]

        else:
            self.model += [ResBlock(input_nc, hidden_nc, hidden_nc,
                                    norm_layer, nonlinearity, learnable_shorcut, use_spect, use_coord)]
            for i in range(num_blocks-2):
                self.model += [ResBlock(hidden_nc, output_nc, hidden_nc,
                                        norm_layer, nonlinearity,learnable_shorcut,use_spect,use_coord)]
            self.model += [ResBlock(hidden_nc, output_nc, hidden_nc,
                                    norm_layer, nonlinearity, learnable_shorcut, use_spect, use_coord)]

        self.model = nn.Sequential(*self.model)

    def forward(self, inputs):

        return self.model(inputs)
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = nn.AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
class ResidualConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(ResidualConv2dBlock, self).__init__()
        mid_dim = max(input_dim, output_dim)
        self.conv1 = Conv2dBlock(input_dim, mid_dim, kernel_size, stride, padding, norm=norm, activation=activation, pad_type=pad_type)
        self.skip = Conv2dBlock(input_dim, output_dim, kernel_size, stride, padding, norm=norm, activation=activation, pad_type=pad_type)
        self.conv2 = Conv2dBlock(mid_dim, output_dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + skip
        return x
class ResBlockDecoder(nn.Module):
    '''
    Define a decoder Block
    '''

    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        bypass = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv1)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, norm_layer(hidden_nc), nonlinearity, conv2,)

        self.shortcut = nn.Sequential(bypass)

    def forward(self, x):

        out = self.model(x) + self.shortcut(x)
        return out
class ResBlockDecoder1(nn.Module):
    '''
    Define a decoder Block
    '''

    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockDecoder1, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=1, padding=1, ), use_spect)
        bypass = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, ), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv1)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, norm_layer(hidden_nc), nonlinearity, conv2,)

        self.shortcut = nn.Sequential(bypass)

    def forward(self, x):

        out = self.model(x) + self.shortcut(x)
        return out
class ResBlockEncoder(nn.Module):
    '''
    Define a decoder Block
    '''

    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockEncoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1  = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1),use_spect)
        conv2 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=4,stride=2, padding=1),use_spect)
        bypass = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1,stride=1, padding=0),use_spect)
        # print(type(norm_layer))
        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(hidden_nc), nonlinearity, conv2, )

        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),bypass)


    def forward(self,x):

        out = self.model(x) + self.shortcut(x)
        return out

class Output(nn.Module):
    '''
    Define the output layer
    '''
    def __init__(self, input_nc, output_nc, kernel_size=3, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Output, self).__init__()

        kwargs = {'kernel_size':kernel_size, 'padding':0, 'bias': True}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1, nn.Tanh())
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1, nn.Tanh())

    def forward(self, x):

        out = self.model(x)

        return out

class Jump(nn.Module):
    '''
    Define the output layer
    '''
    def __init__(self,input_nc, output_nc, kernel_size=3, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Jump, self).__init__()

        kwargs ={'kernel_size': kernel_size, 'padding':0, 'bias': True}
        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1)
        else:
            self.model == nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1)

    def forward(self, x):

        out = self.model(x)
        return out

class ExtractorAttn(nn.Module):
    def __init__(self, feature_nc, kernel_size=4, nonlinearity=nn.LeakyReLU(), softmax=None):
        super(ExtractorAttn, self).__init__()
        self.kernel_size=kernel_size
        hidden_nc = 128
        softmax = nonlinearity if softmax is None else nn.Softmax(dim=1)

        self.extractor = BlockExtractor(kernel_size=kernel_size)
        self.reshape = LocalAttnReshape()
        # kernel prediction network M
        self.fully_connect_layer = nn.Sequential(
            nn.Conv2d(2*feature_nc, hidden_nc, kernel_size= kernel_size, stride=kernel_size, padding=0),
            nonlinearity,
            nn.Conv2d(hidden_nc, kernel_size*kernel_size, kernel_size=1, stride=1, padding=0),
            softmax,)
    # 局部扭曲操作
    def forward(self, source, target, flow_field):
        block_source = self.extractor(source, flow_field)
        # print(block_source.size())
        block_target = self.extractor(target, torch.zeros_like(flow_field))
        # print(block_target.size())
        attn_param = self.fully_connect_layer(torch.cat((block_source, block_target),1))
        attn_param = self.reshape(attn_param, self.kernel_size)
        result = torch.nn.functional.avg_pool2d(attn_param*block_source, self.kernel_size, self.kernel_size)
        return  result


class  Self_Attn(nn.Module):
    """
    自注意力模块
    """
    def __init__(self,in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8,kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x0):
        """

        :param x:
        :param x0:
        :return:
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width * height).permute(0,2,1) # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize,-1,width * height)
        energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width * height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma * out + self.beta * x + x0
        return out


class AppearanceBlock(nn.Module):
    def __init__(self, input_nc=3, ngf=64,img_f=1024,layers=5,norm='batch',activation='RELU',use_spect=True, use_coord=False):
        super(AppearanceBlock, self).__init__()
        self.layers =layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.softmax = nn.Softmax(dim=-1)
        self.query_conv = nn.Conv2d(in_channels=input_nc,out_channels=input_nc // 8,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_nc, out_channels=input_nc //8 ,kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_nc,out_channels=input_nc ,kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,source,pose):
        B,C,W,H = source.size()
        proj_query = self.query_conv(pose).view(B,-1,W*H).permute(0,2,1)   #[B,W * H,C]
        proj_key = self.key_conv(source).view(B,-1,W*H)                #[B,C,W*H]
        energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)   # B * (WH)*(WH)

        proj_value = self.value_conv(source).view(B,-1,W*H)   # B  *C * （W * H）

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(B,C,W,H)

        source_update = self.gamma * out + source

        return source_update


class Pose_Guided_Block(nn.Module):

    def __init__(self,in_dim,num_blocks=2):
        super(Pose_Guided_Block, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim //8,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim //8,kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

        self.block0 = ResBlocks(num_blocks,in_dim)
        self.block1 = ResBlocks(num_blocks,in_dim)
        self.self_attn = Self_Attn(in_dim*2)

    def foward(self,source,spose,tpose):

        B,C,W,H = source.size()
        proj_query = self.query_conv(spose).view(B,-1,W*H).permute(0,2,1)
        proj_key = self.key_conv(tpose).view(B,-1,W*H)
        energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)

        proj_value = self.value_conv(source).view(B,-1,W*H)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(B,C*2,W,H)
        out = self.gamma * out + torch.cat((spose,tpose),dim=1)

        out = torch.cat((out,source),dim=1)
        pose_update = self.self_attn(out,out)

        spose_update = self.block0(spose)
        tpose_update = self.block1(tpose)

        return pose_update,spose_update,tpose_update



class Pose_Style_Block(nn.Module):
    def __init__(self, in_dim, norm='batch',activation='ReLU',use_spect=True,use_coord=False):
        super(Pose_Style_Block, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.softmax = nn.Softmax(dim=-1)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv1 = nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim // 2, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.block0 = ResBlock( in_dim//2,in_dim//2,in_dim//2,norm_layer,nonlinearity,False,use_spect,use_coord)
        self.block1 = ResBlock( in_dim//2,in_dim//2,in_dim//2,norm_layer,nonlinearity,False,use_spect,use_coord)
        self.self_attn = Self_Attn(in_dim * 2)
        self.Contran = ResBlockDecoder1(in_dim * 2,in_dim,in_dim * 2,norm_layer,nonlinearity,use_spect,use_coord)

    def forward(self,source,spose,tpose,pose):
        B, C, W, H = source.size()   #[1,256,32,32]

        proj_query = self.query_conv(pose).view(B,-1,W*H).permute(0,2,1)   #[B,W * H,C]  [1,1024,32 ]
        proj_key = self.key_conv(source).view(B,-1,W*H)                #[B,C,W*H]  ,[1,32,1024]
        energy = torch.bmm(proj_query,proj_key) #[1,1024,1024]
        attention = self.softmax(energy)   # B * (WH)*(WH) #[1,1024,1024]

        proj_value = self.value_conv(source).view(B,-1,W*H)   # B  *C * （W * H） [1,256,1024]

        out = torch.bmm(proj_value,attention.permute(0,2,1))  #[1,256,1024]
        out = out.view(B,C,W,H)   #[1,256,32,32]

        source_update = self.gamma * out + source    #

        B1, C1, W1, H1 = spose.size()       #[1,128,32,32]
        proj_query = self.query_conv1(spose).view(B1, -1, W1 * H1).permute(0, 2, 1) #[1,1024,32]
        proj_key = self.key_conv1(tpose).view(B1, -1, W1 * H1) #[1,32,1024]
        energy = torch.bmm(proj_query, proj_key)  #[1,1024,1024]
        attention = self.softmax(energy)  #[1,1024,1024]

        proj_value = self.value_conv(source).view(B1, -1, W1 * H1)  #[1,256,1024]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  #[1,256,1024]
        out = out.view(B1, C1 * 2, W1, H1)                           #[1,256,32,32]
        out = self.gamma * out + torch.cat((spose, tpose), dim=1)   #[1,256,32,32]

        out = torch.cat((out, source_update), dim=1)    #[1,512,32,32]
        pose_update = self.self_attn(out, out)   #[1,512,32,32]
        pose_update =self.Contran(pose_update)  #[1,256,32,32]
        spose_update = self.block0(spose) #[1,128,32,32]
        tpose_update = self.block1(tpose)  #[1,128,32,32]

        return  source_update,spose_update,tpose_update,pose_update




class SPADE(nn.Module):
    """
    利用的是SPG和SCA中的空间自适应的方法
    """
    def __init__(self,norm_nc,label_nc):
        super(SPADE, self).__init__()
        param_free_norm_type = 'instance'
        ks = 3

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = nn.SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128


        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x):

        # Part 1. generate parameter-free normalized activations
        # normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # cs = F.interpolate(cs, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(x)    #[1,128,32,32]
        gamma = self.mlp_gamma(actv)  #[1,256,32,32]
        beta = self.mlp_beta(actv)  #[1,256,32,32]

        # apply scale and bias
        # out = normalized * (1 + gamma) + beta

        return gamma, beta
class SEAN(nn.Module):
    def __init__(self,norm_nc,style_nc,poseoredge_nc):
        super(SEAN, self).__init__()

        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(torch.zeros(norm_nc), requires_grad=True)
        param_free_norm_type = 'instance'
        ks = 3
        pw =1
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        # self.conv_gamma = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)
        # self.conv_beta = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)
        # if poseoredge_nc != norm_nc
        #     self.mlp_shared = nn.Sequential(
        #         nn.Conv2d(poseoredge_nc, norm_nc, kernel_size=ks, padding=pw),
        #         nn.ReLU()
        #     )
        self.spade_style = SPADE(norm_nc,style_nc)
        self.spade_poseOredge = SPADE(norm_nc, poseoredge_nc)
        # self.spade = SPADE(norm_nc,style_nc)
    def forward(self, x,style_codes,pose_edge):

        # Part 1. generate parameter-free normalized activations
        added_noise = (torch.randn(x.shape[0], x.shape[3], x.shape[2], 1).cuda() * self.noise_var).transpose(1, 3)  #x[1,256,32,32] [1,128,64,64][1,64,128,128]
        normalized = self.param_free_norm(x + added_noise)

        # Part 2. produce scaling and bias conditioned on semantic map
        pose_edge = F.interpolate(pose_edge, size=x.size()[2:], mode='nearest')  #[1,1,32,32]
        # style_codes = F.interpolate(style_codes, size=x.size()[2:], mode='nearest')  #[1,256,32,32]

        gamma_spade_style, beta_spade_style = self.spade_style(style_codes)
        gamma_spade, beta_spade = self.spade_poseOredge(pose_edge)

        gamma_alpha = F.sigmoid(self.blending_gamma)
        beta_alpha = F.sigmoid(self.blending_beta)

        gamma_final = gamma_alpha * gamma_spade_style + (1 - gamma_alpha) * gamma_spade
        beta_final = beta_alpha * beta_spade_style + (1 - beta_alpha) * beta_spade
        out = normalized * (1 + gamma_final) + beta_final

        return out


class SEANResnetBlock(nn.Module):
    def __init__(self, fin, style_nc, poseoredge_nc, weight):
        super().__init__()
        # Attributes
        # self.learned_shortcut = (fin != fout)
        # fmiddle = min(fin, fout)
        self.W = weight

        self.conv_e = nn.Conv2d(fin, style_nc, kernel_size=3, padding=1)
        self.conv_p = nn.Conv2d(fin, style_nc, kernel_size=3, padding=1)

        # define normalization layers
        edge_nc = poseoredge_nc
        self.norm_e = SEAN(fin, style_nc, edge_nc)
        pose_nc = poseoredge_nc
        self.norm_p = SEAN(fin, style_nc, pose_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, style, content_pose, content_edge):#[1,256,32,32][1,128,64,64],[1,64,128,128]

        px = self.conv_p(self.actvn(self.norm_p(x,style, content_pose)))
        ex = self.conv_e(self.actvn(self.norm_e(x, style, content_edge)))
        if self.W > 0:
            out = ex * self.W + px * (1 - self.W)
        else:
            out = ex + px
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)





