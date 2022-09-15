import torch.nn as nn
from torch.nn import init


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        
    @staticmethod
    def modify_commandline_option(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for params in self.parameters():
            num_params += params.numel() #计算元素的数目
            
        print('Network [%s] was created. Total number of parameters: %.d million. '
              'To see the architevture, do print(netwaork).' 
              %(type(self).__name__, num_params))
        
    def init_weights(self,init_type='normal', gain=0.02):
        def init_func(m):
          classname = m.__class__.__name__
          if classname.find('BatchNorm2d') != -1:
              if hasattr(m,'weight') and m.weight is not None:
                  init.normal_(m.weight.data, 1.0, gain)
              if hasattr(m, 'bias') and m.bias is not None:
                  init.constant_(m.bias.data, 0.0)
          elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
              if init_type == 'normal':
                  init.normal_(m.weight.data, 0.0, gain)
              elif init_type =='xavier':
                  init.xavier_normal_(m.weight.data, gain=gain)
              elif init_type == 'xavier_uniform':
                  init.xavier_uniform_(m.weight.data, gain=1.0)
              elif init_type == 'kaiming':
                  init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
              #   正交初始化：  用以解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用的参数初始化方法
              elif init_type == 'orthogonal':
                  init.orthogonal_(m.weight.data, gain=gain)
              elif init_type == 'none':
                  m.reset_parameters()
              else:
                  raise NotImplementedError('initialization method [%s] is not implemented ' % init_type)
              if hasattr(m,'bias') and m.bias is not None:
                  init.constant_(m.bias.data, 0.0)
                  
        self.apply(init_func)
        
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)