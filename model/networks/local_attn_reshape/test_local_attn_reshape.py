from local_attn_reshape import LocalAttnReshape
import torchvision.transforms as transforms
from PIL import Image
import torch
import imageio
import numpy as np


def tensor2im(image_tensor,bytes=255.0,imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0, :3,:,:].cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_numpy, (1,2,0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)

transform_list = []
label_path_1 = './mytest/1.jpeg'
label_path_2 = './mytest/2.jpeg'
if __name__ == '__main__':
    kernel_size = 1
    extractor = LocalAttnReshape()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # forward check


    transform_list += [transforms.Resize(256)]
    transform_list += [transforms.CenterCrop((256,256))]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    tran = transforms.Compose(transform_list)

    label_1 = Image.open(label_path_1)
    # label_2 = Image.open(label_path_2)
    # label_1 = readIamge(label_path_1)
    # label_2 = readIamge(label_path_2)
    tensor_1 = tran(label_1).unsqueeze(0)
    source0 = tensor_1.cuda()
    tensor_1 = torch.cat((tensor_1,tensor_1,tensor_1),1)
    print(tensor_1.size())
    # tensor_2 = tran(label_2).unsqueeze(0)
    # image = torch.cat((tensor_1, tensor_2), 0)
    # image = torch.cat((tensor_1, tensor_2), 0)

    # print(image.shape)
    # source = image.cuda()
    source = tensor_1.cuda()
    print(source.size())
    # flow = torch.zeros(source.shape).cuda() * 0
    # flow = flow[:, :2, :, :]

    # out = extractor(source, flow)
    out = extractor(source)
    print(out.shape)
    image = tensor2im(out)
    save_image(image, 'test.png')
    image = tensor2im(source0)
    save_image(image, 'source.png')







    inputs = torch.tensor(range(9))
    print(inputs)
    inputs = inputs.view(1,-1,1,1)
    print(inputs.size())
    inputs = inputs.view(1, -1, 1, 1).repeat(2,1,10,10).float()
    print(inputs.size())

    source = torch.rand(4,9,14,10).double().cuda()
    source.requires_grad=True
    print(torch.autograd.gradcheck(extractor,(source)))