import numpy as np
import cv2
import torch
from pylab import *
# 每个通道的feature map 灰度图
def every_feature(input):
    # print(input.type())
    feature_map_combination = []
    for i in range(input.shape[1]):
        feature =input[:,i,:,:]
        feature_map_combination.append(feature)
        # print(type(feature.shape[1]))
        # feature = feature.numpy()
        feature = feature.view(feature.shape[1], feature.shape[2])
        feature = feature.data.numpy()
        feature = 1.0 / (1 + np.exp(-1 * feature))
        # to [0,255]
        feature = np.round(feature * 255)
        # feature = np.transpose(feature,[1,2,0]).astype(np.unit8)
        print("feature.shape:",feature.shape)
        cv2.imwrite("./feature_img/"+str(i)+".jpg",cv2.cvtColor(feature,cv2.COLORMAP_JET))
    feature_map_sum = sum(ele for ele in feature_map_combination).squeeze(0)
    # print(feature_map_sum.size())
    plt.imshow(feature_map_sum)
    # plt.axis('off')
    plt.savefig("./feature_img/feature_map_sum.png")
# 每个 通道的 黄蓝彩色图
def feature_map(output):
    for feature_map in output:
        #[N,C,H,W] -> [C,H,W]
        img = np.squeeze(feature_map.detach().numpy())
        #[C,H,W] -> [H,W,C]
        img = np.transpose(img,[1,2,0])
        plt.figure()
        # for i in range(img.shape[2]):
        #     ax = plt.subplot(img.shape[2]/8,8,i+1)
        #     #[H,W,C]
        #     plt.axis('off')
        #     plt.imshow(img[:,:,i])
        for i in range(10):
            ax = plt.subplot(5,5,i+1)
            plt.axis('off')
            plt.imshow(img[:,:,i])
        plt.show()
#通道叠加后的heapmap
def feature_map_jgp(output):
    # print(output.size())
    heat = output.squeeze(0)  #降维操作,尺寸变为[128,64,64]
    # print(heat.size())
    heat_mean = torch.mean(heat,dim=0)  #对各卷积层（128）求平均值，尺寸变为（64,64）
    heatmap = heat_mean.numpy()   #转换为numpy数组
    heatmap /= np.max(heatmap)   #minmax 归一化处理
    heatmap =np.uint8(255*heatmap)  #像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET) #颜色变换
    plt.imshow(heatmap)
    plt.show()
    cv2.imwrite('./feature_img/heatmap.jpg', heatmap)