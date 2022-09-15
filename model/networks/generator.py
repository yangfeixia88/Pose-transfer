import torch
import torch.nn as nn
import torch.nn.functional as F
from model.networks.base_network import  BaseNetwork
from model.networks.base_function import  *
# from util.get_featuremap import feature_map, every_feature, feature_map_jgp

from util.util import multi_gpus_load_dict, my_multi_gpus_load_dict


# class EncoderNet(BaseNetwork):
#     def __init__(self,input_nc=3,ngf=64, norm='batch',
#                  activation='ReLU', use_spect=True, use_coord=False):
#         super(EncoderNet, self).__init__()
#
#         norm_layer = get_norm_layer(norm_type=norm)
#         nonlinearity = get_nonlinearity_layer(activation_type=activation)
#
#         #外观、姿态编码
#         self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
#                                    nonlinearity, use_spect, use_coord)
#
#         self.block1 = EncoderBlock(ngf, ngf*2, norm_layer,
#                                  nonlinearity, use_spect, use_coord)
#
#         self.block2 = EncoderBlock(ngf*2, ngf * 4, norm_layer,
#                                nonlinearity, use_spect, use_coord)
#
#     def forward(self,input):
#
#         output = self.block0(input)
#
#         output= self.block2(self.block1(output))
#
#         return output
# class EncoderNet1(BaseNetwork):
#     def __init__(self,input_nc=3,ngf=64, norm='batch',
#                  activation='ReLU', use_spect=True, use_coord=False):
#         super(EncoderNet1, self).__init__()
#
#         norm_layer = get_norm_layer(norm_type=norm)
#         nonlinearity = get_nonlinearity_layer(activation_type=activation)
#
#         #外观、姿态编码
#         self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
#                                    nonlinearity, use_spect, use_coord)
#
#         self.block1 = EncoderBlock(ngf, ngf*2, norm_layer,
#                                  nonlinearity, use_spect, use_coord)
#         self.block2 = EncoderBlock(ngf*2, ngf * 2, norm_layer,
#                                    nonlinearity, use_spect, use_coord)
#
#     def forward(self,input):
#
#         output = self.block0(input)
#
#         output= self.block2(self.block1(output))
#
#         return output
#
# class Poseinitgenerator(BaseNetwork):
#     def __init__(self,input_nc=3,structure_nc=18,output_nc=3,ngf=64,layers=6,num_blocks=5,
#                  norm='batch',activation='ReLU',use_spect=True,ues_coord=False):
#         super(Poseinitgenerator, self).__init__()
#
#         self.layers = layers
#
#         norm_layer = get_norm_layer(norm_type=norm)
#         nonlinearity = get_nonlinearity_layer(activation_type=activation)
#
#         self.encoder_source = EncoderNet(input_nc,ngf,norm,activation,use_spect,ues_coord)
#         self.encoder_spose = EncoderNet1(structure_nc,ngf,norm,activation,use_spect,ues_coord)
#         self.encoder_tpose = EncoderNet1(structure_nc,ngf,norm,activation,use_spect,ues_coord)
#         self.up1 = ResBlockDecoder(ngf*4,ngf*2,ngf*4,norm_layer,nonlinearity,use_spect,ues_coord)
#         self.up2 = ResBlock(ngf*2,ngf*2,ngf*2,norm_layer,nonlinearity,use_spect,ues_coord)
#         self.up3 = ResBlockDecoder(ngf*2,ngf,ngf*2,norm_layer,nonlinearity,use_spect,ues_coord)
#         self.up4 = ResBlockDecoder(ngf,ngf,ngf,norm_layer,nonlinearity,use_spect,ues_coord)
#         self.outconv =Output(ngf,output_nc,3,None,nonlinearity,use_spect,ues_coord)
#
#         attBlock = nn.ModuleList()
#         for i in range(num_blocks):
#             attBlock.append(Pose_Style_Block(ngf*4,norm,activation,use_spect,ues_coord))
#
#         self.attn = attBlock
#         # self.atten = Pose_Style_Block(ngf*4,norm,activation,use_spect,ues_coord)
#         #
#         # self.atten1 = Pose_Style_Block(ngf * 4, norm, activation, use_spect, ues_coord)
#
#
#     def forward(self,source,source_B,target_B):
#         source = self.encoder_source(source)  # [1,256,32,32]
#         source_B = self.encoder_spose(source_B) #[1,128,32,32]
#         # print(source_B.size())
#         target_B = self.encoder_tpose(target_B)  #[1,128,32,32]
#         # print(target_B.size())
#         pose = torch.cat((source_B,target_B),dim=1) #[1,256,32,32]
#         for model in self.attn:
#             source,source_B,target_B,pose = model(source,source_B,target_B,pose)
#
#         # print(source.size())
#         # source, source_B, target_B, pose = self.atten(source, source_B, target_B, pose)
#         # source, source_B, target_B, pose = self.atten1(source, source_B, target_B, pose)
#         x = self.up1(source)
#         x = self.up2(x)
#         x =self.up3(x)
#         x = self.outconv(self.up4(x))
#
#         return x



# class PoseGenerator(BaseNetwork):
#     def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2,content_weight=0.005, norm='batch',activation='ReLU',
#                  attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True,use_coord=False):
#         super(PoseGenerator, self).__init__()
#         self.edge_net = EdgeContentEncoder(input_dim=image_nc-2,output_dim=image_nc-2,structure_nc=structure_nc,ngf=64,use_spect=False)   #与预训练的参数信息一样
#         # self.edge_net = PoseEdgeNet(image_nc-2,output_nc-2, structure_nc,ngf, use_spect,use_coord)
#         # self.edge_net = EdgeNet(image_nc-2,structure_nc,image_nc-2,ngf,norm,layers, img_f, activation,use_spect,use_coord)
#         # self.edge_code = EdgeNet(image_nc-2, ngf, img_f, layers,norm, activation, use_spect, use_coord)
#
#     def forward(self, source, source_B, target_B,input_EP1):   #加上input_EP1
#           #[1,256,32,32][1,128,64,64][1,64,128,128]
#         fake_EP2 =self.edge_net(input_EP1, source_B, target_B)  #[1,512,8,8][1,256,16,16][1,128,32,32][1,64,64,64]
#
#         return fake_EP2

class PoseGenerator(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2,content_weight=0.005, norm='batch',activation='ReLU',
                 attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True,use_coord=False):
        super(PoseGenerator, self).__init__()
        self.source = PoseSourceNet(image_nc, ngf, img_f, layers,
                                                   norm, activation, use_spect, use_coord)
        self.target = PoseTargetNet(image_nc, structure_nc, output_nc, ngf, img_f, layers, num_blocks,content_weight,
                                    norm, activation, attn_layer, extractor_kz, use_spect, use_coord)
        self.flow_net = PoseFlowNet(image_nc, structure_nc,ngf=32,img_f=256,encoder_layer=5, attn_layer=attn_layer,
                                    norm=norm, activation=activation,
                                    use_spect=use_spect, use_coord=use_coord)
        # self.edge_net = EdgeContentEncoder(input_dim=image_nc-2,output_dim=image_nc-2,structure_nc=structure_nc,ngf=64,use_spect=False)   #与预训练的参数信息一样
        self.edge_net = PoseEdgeNet(image_nc-2,output_nc-2, structure_nc,ngf, use_spect,use_coord)
        # self.edge_net = EdgeNet(image_nc-2,structure_nc,image_nc-2,ngf,norm,layers, img_f, activation,use_spect,use_coord)
        self.edge_code = EdgeNet(image_nc-2, ngf, img_f, layers,norm, activation, use_spect, use_coord)   #边缘图特征提取

    def forward(self, source, source_B, target_B,input_EP1):   #加上input_EP1
        feature_list = self.source(source)   #[1,256,32,32][1,128,64,64][1,64,128,128]
        # fake_EP2 =self.edge_net(input_EP1, source_B, target_B)  #[1,512,8,8][1,256,16,16][1,128,32,32][1,64,64,64]
        flow_fields, masks = self.flow_net(source, source_B, target_B)
        for param in self.edge_net.parameters():
            param.requires_grad = False
        fake_EP2 = self.edge_net(input_EP1, torch.cat((source_B, target_B), dim=1))
        edge_codes = self.edge_code(fake_EP2)
         #[1,2,32,32][1,2,64,64]
        # content_edge = fake_EP2
        image_gen = self.target(target_B, feature_list,flow_fields, masks,edge_codes)

        return image_gen, flow_fields, masks,fake_EP2
        # # image_gen = self.target(target_B, feature_list, flow_fields, masks)
        # # return image_gen, flow_fields, masks
        # return fake_EP2

# 残差网络提取源图像的风格特征
class PoseSourceNet(BaseNetwork):

    def __init__(self, input_nc=3, ngf=64, img_f=1024, layers=6, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False):
        super(PoseSourceNet, self).__init__()
        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)


        # encoder part CONV_BLOCKS
        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                   nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2**(i+1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)


    def forward(self, source):

        feature_list=[source]  # [1,3,256,256]
        out = self.block0(source) #[1,64,128,128]
        feature_list.append(out)
        for i in range(self.layers-1):  #i=0,[1,128,64,64]  ,i=1 ,[1,256,32,32]

            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature_list.append(out)
            # feature_map(out.cpu())
            # feature_map_jgp(out.cpu())
        feature_list = list(reversed(feature_list)) # [1,256,]
        return  feature_list
# 残差网络提取源图像的风格特征
# class StyleNet(BaseNetwork):
#     def __init__(self,input_dim,dim,norm,activate,pad_type):
#         super(StyleNet, self).__init__()
#
#         self.block0 = ResidualConv2dBlock(input_dim,dim,7,1,3, norm=norm,activation=activate,pad_type=pad_type)
#
#         self.block1 = ResidualConv2dBlock(dim,dim *2, 3, 2, 1, norm=norm, activation=activate, pad_type=pad_type)  #[1,64]
#         self.block2 = ResidualConv2dBlock(dim *2, dim*4, 3,2,1, norm=norm, activation=activate,pad_type=pad_type )  #[1,128]
#         self.block3 = ResidualConv2dBlock(dim *4,dim*8,3,2,1,norm=norm,activation=activate,pad_type=pad_type)#[1,256]
#         self.block4 = ResidualConv2dBlock(dim *8,dim*8,3,2,1,norm=norm, activation=activate,pad_type=pad_type)#[1,256]
#         # self.block5 = ResidualConv2dBlock(dim * 8, dim * 8, 3, 2, 1, norm=norm, activation=activate, pad_type=pad_type)
#
#         # self.model = []
#         # self.model += [self.block0]
#         # self.model += [self.block1]
#         # self.model += [self.block2]
#         # self.model += [self.block3]
#         # self.model += [self.block4]
#         # self.model += [self.block5]
#
#         # self.model = nn.Sequential(*self.model)
#     def forward(self, x):
#         # out = self.model(x)
#         feature_list =[x]
#         out = self.block0(x)
#         # feature_list.append(out)
#         out = self.block1(x)
#         # feature_list.append(out)
#         out = self.block2(out)
#         feature_list.append(out)
#         out = self.block3(out)
#         # feature_list.append(out)
#         out = self.block4(out)
#         feature_list.append(out)
#         return feature_list

class EdgeNet(BaseNetwork):
    def __init__(self,input_nc=1,ngf=32,img_f=1024,layers =6,norm='batch',activation='ReLU',use_spect=True,use_coord=False):
        super(EdgeNet, self).__init__()

        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self.block0 =EncoderBlock(input_nc, ngf, norm_layer,
                                   nonlinearity, use_spect, use_coord)
        # self.edge_generator =EdgeContentEncoder(input_nc, output_nc)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = EncoderBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)
    def forward(self,fake_EP2):
        # real_pose = torch.cat((input_BP1,input_BP2),dim=1)
        # fake_EP2 = self.edge_generator(input_EP1,real_pose)
        # fake_EP2 = self.edge_generator(input_EP1,input_BP1,input_BP2)
        edge_codes=[fake_EP2]
        out = self.block0(fake_EP2)  #[1,32,128,128]
        edge_codes.append(out)
        for i in range(self.layers-1):
            model = getattr(self,'encoder' + str(i))
            out = model(out)
            edge_codes.append(out)
        edge_codes.reverse()
        return edge_codes

class PoseTargetNet(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2, content_weight=0.005,
                 norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):
        super(PoseTargetNet, self).__init__()

        self.layers = layers
        self.attn_layers = attn_layer

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self.block0 = EncoderBlock(structure_nc, ngf, norm_layer,
                                   nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)


        mult = min(2 ** (layers-1), img_f//ngf)
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers-i-2), img_f//ngf) if i != layers-1 else 1
            # if num_blocks ==1:
            #     up = nn.Sequential(ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer,
            #                                        nonlinearity, use_spect, use_coord))
            # else:
            #     up = nn.Sequential(ResBlocks(num_blocks-1, ngf*mult_prev, None, None, norm_layer,
            #                                  nonlinearity, False, use_spect, use_coord),
            #
            #         ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer,
            #                                        nonlinearity, use_spect,use_coord))
            up = nn.Sequential(ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer,
                                                    nonlinearity, use_spect, use_coord))
            setattr(self, 'decoder' + str(i), up)
            res = ResBlocks(num_blocks - 1, ngf * mult_prev, None, None, norm_layer,
                                                        nonlinearity, False, use_spect, use_coord)
            setattr(self,'res'+ str(i),res)
            sean = SEANResnetBlock(ngf * mult_prev, ngf * mult_prev, ngf * mult_prev, content_weight) #[256,256][,128,128][64,64]
            setattr(self,'sean' + str(i),sean)
            if layers-i in attn_layer:
                attn = ExtractorAttn(ngf*mult_prev, extractor_kz[str(layers-i)], nonlinearity, softmax=True)
                setattr(self, 'attn' + str(i), attn)

        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)

    # posetargetnet
    def forward(self, target_B, source_feature, flow_fields, masks,fake_EP2):  #加上fake_EP2
        pose_codes = []
        out = self.block0(target_B) # [1, 64, 128, 128]#
        pose_codes.append(out)
        # feature_map(out.cpu())
        for i in range(self.layers-1):    #i=0 [1,128,64,64] i=1 [1,256,32,32] layers=3
            model = getattr(self, 'encoder' + str(i))
            out = model(out) # [1, 256, 32, 32]
            pose_codes.append(out)
            # feature_map(out.cpu())
        pose_codes.reverse()
        counter=0
        for i in range(self.layers):  #layers=3

            if self.layers-i in self.attn_layers:  #attn_layers=[2,3]
                model = getattr(self, 'attn' + str(i))  #i=0 ,[1, 256, 32, 32]   i=1 ,[1, 128, 64, 64]
                out_attn = model(source_feature[i], out, flow_fields[counter])  #i=0,1  source_feature[1,256,32,32],[1,128,64,64],[1,64,128,128],[1,3,256,256]
                # print(type(out_attn),type(out),type(masks))
                out = out*(1-masks[counter]) + out_attn*masks[counter]  #[1,256,32,32]，[1,128,64,64]
                # feature_map(out.cpu())
                counter += 1

            # if i == 1:
            #     out = out + visi(flow_fields[-1],source_feature[i])
            # out =model(out)

            model =getattr(self, 'res' +str(i))
            out = model(out)
            model = getattr(self, 'sean' + str(i))  #i=0  [1,256,32,32],i=1[1,128,64,64], i=2 [1,64,128,128]
            out = model(out, source_feature[i], pose_codes[i],fake_EP2[i])
            model = getattr(self, 'decoder' + str(i)) # i=0  [1, 128, 64, 64] i=1,[1, 64, 128, 128]   i=2 [1, 64, 256, 256]
            out = model(out)
        # feature_map(out.cpu())
        out_image = self.outconv(out)
        # feature_map_jgp(out_image.cpu())
        return out_image


class PoseFlowNet(nn.Module):
    ''' docstring for FlowNet'''
    def __init__(self, image_nc, structure_nc, ngf=64, img_f=1024, encoder_layer=5, attn_layer=[1], norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False):
        super(PoseFlowNet, self).__init__()

        self.encoder_layer = encoder_layer
        self.decoder_layer = encoder_layer - min(attn_layer)
        self.attn_layer = attn_layer
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = 2*structure_nc + image_nc

        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                   nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(encoder_layer-1):

            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect,use_coord)
            setattr(self, 'encoder' + str(i), block)


        for i in range(self.decoder_layer):
            mult_prev = mult
            mult = min(2 ** (encoder_layer-i-2), img_f//ngf) if i != encoder_layer-1 else 1
            up =ResBlockDecoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer,
                                nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), up)

            jumpconv = Jump(ngf*mult, ngf*mult, 3, None, nonlinearity, use_spect, use_coord)
            setattr(self, 'jump' + str(i), jumpconv)

            if encoder_layer-i-1 in attn_layer:
                flow_out = nn.Conv2d(ngf*mult, 2, kernel_size=3, stride=1, padding=1,bias=True)
                setattr(self, 'output' + str(i), flow_out)

                flow_mask = nn.Sequential(nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'mask' + str(i), flow_mask)

    def forward(self, source, source_B, target_B):
        flow_fields=[]
        masks=[]
        inputs = torch.cat((source, source_B, target_B),1) #[1,39,256,256]

        out = self.block0(inputs)  #[1,32,128,128]
        result=[out]
        # feature_map(out.cpu())
        for i in range(self.encoder_layer-1):

            model = getattr(self, 'encoder' + str(i)) # i=0,[1,64,64,64]  i=1,[1,128,32,32]  i=2, [1,256,16,16]   i=3,[1,256,8,8]
            out = model(out)
            result.append(out)
            # feature_map(out.cpu())
        for i in range(self.decoder_layer):    #    decoder_layer =3

            model = getattr(self, 'decoder' + str(i))    #out   i=0, [1,256,16,16]  i=1   [1,128,32,32]   i=2 [1,64,,64,64]
            out = model(out)
            # feature_map(out.cpu())
            model = getattr(self, 'jump' + str(i))        #jump  i=0, [1,256,16,16]  i=1   [1,128,32,32]    i=2  [1,64,64,64]
            jump = model(result[self.encoder_layer-i-2])

            out = out+jump
            # feature_map(out.cpu())
            if self.encoder_layer-i-1 in self.attn_layer:    # att =2  [1,2,32,32]   attn=3  [1,2,64,64]

                flow_field, mask = self.attn_output(out, i)  #
                flow_fields.append(flow_field)
                masks.append(mask)

        return flow_fields, masks

    def attn_output(self,out, i):
        model = getattr(self, 'output' + str(i))
        flow = model(out)
        model = getattr(self, 'mask' + str(i))
        mask = model(out)
        return flow, mask
# 流估计器
# class PoseFlowNet(nn.Module):
#     def __init__(self,input_nc, structure_nc =18,ngf=32, norm='instance',activation='ReLU', use_spect=True, use_coord=False):
#         super(PoseFlowNet, self).__init__()
#         norm_layer = get_norm_layer(norm_type=norm)
#         nonlinearity = get_nonlinearity_layer(activation_type=activation)
#         self.softmax = nn.Softmax(dim=-1)
#         # 图像
#         self.encoder_image_block = EncoderBlock(input_nc, ngf, norm_layer,nonlinearity, use_spect, use_coord)
#
#         self.encoder_block0 = EncoderBlock(ngf, ngf *2, norm_layer,nonlinearity, use_spect, use_coord) #(32,64)
#         self.encoder_block1 = EncoderBlock(ngf*2, ngf * 4, norm_layer, nonlinearity, use_spect, use_coord)  #(64,128)
#         self.encoder_block2 = EncoderBlock(ngf * 4, ngf * 8, norm_layer, nonlinearity, use_spect, use_coord)  #(128,256)
#         self.encoder_block3 = EncoderBlock(ngf * 8, ngf * 8, norm_layer, nonlinearity, use_spect, use_coord)
#         # 姿态
#         self.encoder_structure = EncoderBlock(structure_nc, ngf, norm_layer,nonlinearity, use_spect, use_coord)
#
#         self.query_conv = nn.Conv2d(in_channels=ngf *4,out_channels=ngf  *4,kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=ngf *4, out_channels=ngf *4, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=ngf *4, out_channels=ngf *4, kernel_size=1)
#         self.query_conv1 = nn.Conv2d(in_channels=ngf * 8, out_channels=ngf *8 , kernel_size=1)
#         self.key_conv1 = nn.Conv2d(in_channels=ngf * 8, out_channels=ngf *8, kernel_size=1)
#         self.value_conv1 = nn.Conv2d(in_channels=ngf *8, out_channels=ngf *8, kernel_size=1)
#
#         self.up0 = ResBlockDecoder(ngf*8, ngf*8, ngf*8, norm_layer,nonlinearity, use_spect, use_coord)
#         self.up1 = ResBlockDecoder(ngf *8, ngf*4 , ngf*4 , norm_layer, nonlinearity, use_spect, use_coord)
#         self.up2 = ResBlockDecoder(ngf*4 , ngf*2, ngf*2, norm_layer, nonlinearity, use_spect, use_coord)
#         self.jumpconv1 = Jump(ngf*4, ngf*4, 3, None, nonlinearity, use_spect, use_coord)
#         self.jumpconv2 = Jump(ngf * 8, ngf * 8, 3, None, nonlinearity, use_spect, use_coord)
#         self.flow_out1 = nn.Conv2d(ngf*2, 2, kernel_size=3, stride=1, padding=1, bias=True)
#         self.flow_mask1 = nn.Sequential(nn.Conv2d(ngf*2, 1, kernel_size=3, stride=1, padding=1, bias=True),nn.Sigmoid())
#         self.flow_out0 = nn.Conv2d(ngf * 4, 2, kernel_size=3, stride=1, padding=1, bias=True)
#         self.flow_mask0 = nn.Sequential(nn.Conv2d(ngf * 4, 1, kernel_size=3, stride=1, padding=1, bias=True),
#                                        nn.Sigmoid())
#
#     def forward(self,source,source_B,target_B):
#         flow_fields = []
#         masks=[]
#         # 提取源图像特征
#         source_out = self.encoder_image_block(source)  #[1,32,128,128]
#         source_out0 = self.encoder_block0(source_out)   #[1,64,64,64]
#         source_out1 = self.encoder_block1(source_out0)   #[1,128,32,32]
#         # 源姿态
#         source_B_out = self.encoder_structure(source_B) #[1,32,128,128]
#         source_B_out0 = self.encoder_block0(source_B_out)  #[1,64,64,64]
#         source_B_out1 = self.encoder_block1(source_B_out0) #[1,128,32,32]
#         source_B_out2 = self.encoder_block2(source_B_out1)   #[1,256,16,16]
#         # 目标姿态
#         target_B_out = self.encoder_structure(target_B)  #[1,32,128,128]
#         target_B_out0 = self.encoder_block0(target_B_out)    #[1,64,64,64]
#         target_B_out1 = self.encoder_block1(target_B_out0)  #[1,128,32,32]
#         target_B_out2 = self.encoder_block2(target_B_out1)  #[1,256,16,16]
#
#         B, C, W, H = source_out1.size()     # [1,32,128,128]
#         target_B_out11 = self.key_conv(target_B_out1).view(B,-1,W*H) #[1,16,1024]
#         source_B_out11 = self.query_conv(source_B_out1).view(B, -1, W * H).permute(0, 2, 1)  #[1,1024,16]
#         attention = self.softmax(torch.bmm(source_B_out11,target_B_out11))   #[1,1024,1024]
#         source_out11 = self.value_conv(source_out1).view(B,-1,W*H)      #[1,128,1024]
#         out0 = torch.bmm(source_out11,attention.permute(0,2,1))  #[1,128,1024]
#         out0 = out0.view(B,C,W,H)
#
#         source_out2 = self.encoder_block2(out0)  ##[1,256,16,16]
#         B1, C1, W1, H1 = source_out2.size()
#         source_B_out22 = self.query_conv1(source_B_out2).view(B1, -1, W1 * H1).permute(0, 2, 1) #[1,256,256]
#         target_B_out22 = self.key_conv1(target_B_out2).view(B1, -1, W1 * H1)  #[1,256,256]
#         attention = self.softmax(torch.bmm(source_B_out22, target_B_out22))   #[1,256,256]
#         source_out22 = self.value_conv1(source_out2).view(B1, -1, W1 * H1)    #[1,256,256]
#         out1 = torch.bmm(source_out22, attention.permute(0, 2, 1))
#         out1 = out1.view(B1, C1, W1, H1)                                      #[1,256,16,16]
#
#         out =  self.encoder_block3(out1)  #[1,256,8,8]
#         out=self.up0(out)                #[1,256,16,16]
#         jump = self.jumpconv2(source_out2)  ##[1,256,16,16]
#         out = out+jump                        ##[1,256,16,16]
#         out = self.up1(out)
#         #[1,128,32,32]
#         flow0 = self.flow_out0(out)
#         mark0 = self.flow_mask0(out)
#         flow_fields.append(flow0)            #[1,128,32,32]
#         masks.append(mark0)
#
#         jump = self.jumpconv1(source_out1)  #[1,128,32,32]
#         out= out+jump                        #[1,128,32,32]
#         out = self.up2(out)
#         #[1,64,64,64]
#         flow1 = self.flow_out1(out)
#         mark1 = self.flow_mask1(out)
#         flow_fields.append(flow1)          #[1,64,64,64]
#         masks.append(mark1)
#         return flow_fields,masks

class PoseFlowNetGenerator(BaseNetwork):
    def __init__(self, image_nc=3,structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, norm='instance',
                 activation='ReLU', encoder_layer=5, attn_layer=[1,2], use_spect=True, use_coord=False):
        super(PoseFlowNetGenerator, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer

        self.flow_net = PoseFlowNet(image_nc, structure_nc,ngf,img_f,
                                    encoder_layer, attn_layer=attn_layer,
                                    norm=norm, activation=activation,
                                    use_spect=use_spect, use_coord=use_coord)

    def forward(self, source, source_B, target_B):
        flow_fields, masks = self.flow_net(source, source_B, target_B)
        return flow_fields, masks
class PoseEdgeNetGenerator(BaseNetwork):
    def __init__(self,input_nc = 1,output_nc= 1,structure_nc=18, ngf=64,use_spect=True,ues_coord=False):
        super(PoseEdgeNetGenerator, self).__init__()

        self.edge_net = PoseEdgeNet(input_nc,output_nc,structure_nc,ngf,use_spect,ues_coord)
    def forward(self, input_EP1,source_B, target_B):
        fake_EP2 = self.edge_net(input_EP1,torch.cat((source_B,target_B),dim=1))
        return fake_EP2
# 边缘图生成
def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)


def downconv2x(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)


def upconv2x(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)

class ResidualBlock(nn.Module):

    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        layers = [
            conv3x3(num_channels, num_channels),
            # nn.BatchNorm2d(num_channels),
            nn.InstanceNorm2d(num_channels),
            nn.ReLU(inplace=True),
            conv3x3(num_channels, num_channels),
            # nn.BatchNorm2d(num_channels)
            nn.InstanceNorm2d(num_channels)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x) + x
        return y

class PoseEdgeNet(nn.Module):

    def __init__(self, input_nc=1, output_nc=1,structure_nc=18, ngf=64,use_spect=True,use_coord=False):
        super(PoseEdgeNet, self).__init__()

        in1_channels = input_nc
        out_channels = output_nc
        self.in1_conv1 = self.inconv(in1_channels, ngf)  #[1,1,256,256] -> [1.64,256,256]
        self.in1_down1 = self.down2x(ngf, ngf * 2)    #  [1.64,256,256] —> [1,128,128,128]
        self.in1_down2 = self.down2x(ngf * 2, ngf * 4)   # [1,128,128,128] -> [1,256,64,64]
        self.in1_down3 = self.down2x(ngf * 4, ngf * 8)   #[1,256,64,64]  ->[1,512,32,32]
        # self.in1_down4 = self.down2x(ngf * 8, ngf * 16)

        self.in2_conv1 = self.inconv(structure_nc * 2, ngf)
        self.in2_down1 = self.down2x(ngf, ngf * 2)
        self.in2_down2 = self.down2x(ngf * 2, ngf * 4)
        self.in2_down3 = self.down2x(ngf * 4, ngf * 8)
        # self.in2_down4 = self.down2x(ngf * 8, ngf * 16)

        # self.out_up1 = self.up2x(ngf * 16, ngf * 8)
        self.out_up2 = self.up2x(ngf * 8, ngf * 4)
        self.out_up3 = self.up2x(ngf * 4, ngf * 2)
        self.out_up4 = self.up2x(ngf * 2, ngf)
        self.out_conv1 = self.outconv(ngf, out_channels)

    def inconv(self, in_channels, out_channels):
        return nn.Sequential(
            conv3x3(in_channels, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def outconv(self, in_channels, out_channels):
        return nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            conv1x1(in_channels, out_channels),#将全连接层修改了
            nn.Tanh()
        )

    def down2x(self, in_channels, out_channels):
        return nn.Sequential(
            downconv2x(in_channels, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )

    def up2x(self, in_channels, out_channels):
        return nn.Sequential(
            upconv2x(in_channels, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )

    def forward(self, x1, x2):
        x1_c1 = self.in1_conv1(x1)
        x1_d1 = self.in1_down1(x1_c1)
        x1_d2 = self.in1_down2(x1_d1)
        x1_d3 = self.in1_down3(x1_d2)
        # x1_d4 = self.in1_down4(x1_d3)

        x2_c1 = self.in2_conv1(x2)
        x2_d1 = self.in2_down1(x2_c1)
        x2_d2 = self.in2_down2(x2_d1)
        x2_d3 = self.in2_down3(x2_d2)
        # x2_d4 = self.in2_down4(x2_d3)

        # y = x1_d4 * torch.sigmoid(x2_d4)
        # y = self.out_up1(y)
        y = x1_d3 * torch.sigmoid(x2_d3)
        y = self.out_up2(y)
        y = y * torch.sigmoid(x2_d2)
        y = self.out_up3(y)
        y = y * torch.sigmoid(x2_d1)  #(128,128,128)
        y = self.out_up4(y)  #(1,64,256,256)
        y = self.out_conv1(y)

        return y
# class PoseEdgeGenerator(BaseNetwork):
#     """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
#     We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
#     """
#
#     def __init__(self, input_nc=18+1+18, output_nc=1, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
#         """Construct a Resnet-based generator
#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#         """
#         assert(n_blocks >= 0)
#         super(PoseEdgeGenerator, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         prepro = [nn.ReflectionPad2d(3),
#                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]
#
#         encoder = []
#         n_downsampling = 2
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i
#             encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(True)]
#         residual = []
#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):       # add ResNet blocks
#
#             residual += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
#         decoder = []
#         for i in range(n_downsampling):  # add upsampling layers
#             mult = 2 ** (n_downsampling - i)
#             decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                          kernel_size=3, stride=2,
#                                          padding=1, output_padding=1,
#                                          bias=use_bias),
#                       norm_layer(int(ngf * mult / 2)),
#                       nn.ReLU(True)]
#         postpro = [nn.ReflectionPad2d(3)]
#         postpro += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
#         postpro += [nn.Tanh()]
#
#         self.prepro = nn.Sequential(*prepro)
#         self.encoder = nn.Sequential(*encoder)
#         self.residual = nn.Sequential(*residual)
#         self.decoder = nn.Sequential(*decoder)
#         self.postpro = nn.Sequential(*postpro)
#
#     def forward(self, input_EP1, input_BP2, input_BEP2):
#         """Standard forward"""
#         x = torch.cat([input_EP1, input_BP2, input_BEP2], dim=1)
#         x = self.prepro(x)
#         x = self.encoder(x)
#         x = self.residual(x)
#         x = self.decoder(x)
#         x = self.postpro(x)
#         return x
 ###################################
        # 边缘内容
 ###################################
class EdgeContentEncoder(nn.Module):
    def __init__(self, input_dim, output_dim,structure_nc=18, ngf=64,use_spect=True,use_coord=False):
        super(EdgeContentEncoder, self).__init__()
        self.model_edge = PoseEdgeNetGenerator(input_dim,output_dim,structure_nc, ngf,use_spect,use_coord)
        self.model_edge = my_multi_gpus_load_dict(self.model_edge,'result/posedge_fashion_checkpoints/110000_net_G.pth')
        for param in self.model_edge.parameters():
            param.requirs_grad = False
    def forward(self,input_EP1,source_B, target_B):
        return self.model_edge(input_EP1,source_B, target_B)
    # def forward(self,input_EP1,realpose):
    #     return self.model_edge(input_EP1,realpose)

