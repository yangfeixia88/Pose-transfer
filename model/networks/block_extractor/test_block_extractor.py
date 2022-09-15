from block_extractor import BlockExtractor
import torch
from PIL import Image
import torchvision.transforms as transforms
import imageio
import numpy as np
from visdom import Visdom
def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0,:3,:,:].cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)
def readIamge(path,size=128):
    mode = Image.open(path)
    transform_s = transforms.Compose([
        transforms.Scale(size),
        transforms.CenterCrop((size,size)),
        # transforms.ToTensor()
    ])
    mode = transform_s(mode)
    return mode
transform_list=[]
label_path_1 = './mytest/1.jpeg'
label_path_2 = './mytest/5.jpeg'
if __name__ == "__main__":
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # forward check
    kernel_size = 3
    extractor = BlockExtractor(kernel_size)

    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    tran = transforms.Compose(transform_list)
    print(type(tran))
    # label_1 = Image.open(label_path_1)
    # label_2 = Image.open(label_path_2)
    label_1 = readIamge(label_path_1)
    label_2 = readIamge(label_path_2)
    tensor_1 = tran(label_1).unsqueeze(0)
    tensor_2 = tran(label_2).unsqueeze(0)
    # image = torch.cat((tensor_1, tensor_2), 0)
    image = torch.cat((tensor_2, tensor_1), 0)

    # print(image.shape)
    source = image.cuda()
    # source = tensor_2.cuda()
    flow = torch.ones(source.shape).cuda()*0
    flow = flow[:,:2,:,:]

    out = extractor(source, flow)
    image = tensor2im(out)
    save_image(image, 'test3.png')
    image = tensor2im(source)
    save_image(image, 'source3.png')
    # viz.image(image)
    # print(torch.sum(torch.abs(out[0,0,3:6,3:6]-source[0,0,0:3,0:3])))
    # image = tensor2im(out-source)
    # save_image(image, 'mine.png')





    # source = torch.rand(8,3,64,64).cuda()
    # flow = torch.rand(8,2,64,64).cuda()*0
    # source.requires_grad=True
    # flow.requires_grad=True
    # out = markovattn(source, flow)
    # print(torch.max(torch.abs(out-source)))
    # print(torch.min(torch.abs(out-source)))
    # image = tensor2im(out-source)
    # save_image(image, 'test.png')

    # backward check
    # with torch.no_grad():
    #     source = torch.randn(1,3,3,3).float().cuda()
    #     flow = torch.randn(1,2,3,3).float().cuda()*1.8
    #     source.requires_grad=True
    #     flow.requires_grad=True
    # print(torch.autograd.gradcheck(extractor, (source, flow),1e-6))



