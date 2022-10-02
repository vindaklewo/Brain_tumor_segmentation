from sklearn import datasets
import torch.utils.data as data
import PIL.Image as Image
import os 
import numpy as np
import torch

def make_dataset(rootdata,roottarget):   # 加载数据存储的文件
    imgs=[]
    filename_data=[x for x in os.listdir(rootdata)]
    #print('filename_data',filename_data)
    for name in filename_data:
        #print('name',name)
        img=os.path.join(rootdata,name)
        mask=os.path.join(roottarget,name)
        imgs.append((img,mask))
    return imgs 

class MyDataset(data.Dataset):   # 获取数据集
    def __init__(self,rootdata,roottarget,transform=None,target_transform=None):
        imgs=make_dataset(rootdata,roottarget)
        self.imgs=imgs 
        self.transform=transform
        self.target_transform=target_transform

    def __getitem__(self,index):
        x_path,y_path=self.imgs[index]
        img_x=np.load(x_path)
        img_y=np.load(y_path)
        # import matplotlib.pyplot as plt   # 画出一组数据集
        # plot_x=torch.tensor(img_x).permute(2,0,1)
        # plot_y=torch.tensor(img_y)
        # plt.subplot(1,5,1)
        # plt.imshow(plot_x[0])
        # plt.subplot(1,5,2)
        # plt.imshow(plot_x[1])
        # plt.subplot(1,5,3)
        # plt.imshow(plot_x[2])
        # plt.subplot(1,5,4)
        # plt.imshow(plot_x[3])
        # plt.subplot(1,5,5)
        # .imshow(plot_y)
        # plt.show()
        img_x=img_x.astype('uint8')
        img_y=img_y.astype('uint8')
        if self.transform is not None:
            img_x=self.transform(img_x)
        if self.target_transform is not None:
            img_y=self.target_transform(img_y)
        return img_x,img_y  # 返回数据集的输入数据(img_x) 以及标签(img_y)

    def __len__(self):
        return len(self.imgs)
