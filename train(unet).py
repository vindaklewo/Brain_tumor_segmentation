import torch
from torch.utils.data import DataLoader
from torch import autograd,optim
from torchvision import transforms
from unet import UNet 
from datasource import MyDataset
from datasource import make_dataset
import time
import torch.nn as nn
import torch.nn.functional as F

x_transforms=transforms.Compose([transforms.ToTensor()])  
y_transforms=transforms.ToTensor()
# torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起

class FocalLoss(nn.Module):  # 定义FocalLoss损失函数，解决数据不平衡造成的模型性能问题
    def __init__(self,alpha=1,gamma=2,logits=False,reduce=True):
        super(FocalLoss,self).__init__()
        self.alpha=alpha 
        self.gamma=gamma
        self.logits=logits
        self.reduce=reduce 

    def forward(self,inputs,targets):
        if self.logits:
            BCE_loss=F.binary_cross_entropy_with_logits(inputs,targets,reduce=False)
        else:
            BCE_loss=F.binary_cross_entropy(inputs,targets,reduce=False)
        pt=torch.exp(-BCE_loss)
        F_loss=self.alpha*(1-pt)**self.gamma*BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def train_model(model,criterion,optimizer,dataload,num_epochs=150):  # 训练UNet3+模型
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        epoch_loss=0
        step=0
        for x,y in dataload:
            step+=1
            inputs=x
            labels=y*100
            optimizer.zero_grad()  # 把模型中参数的梯度设为0
            outputs=model(inputs)
            loss=criterion(outputs,labels)  # 计算损失
            loss.backward()  # .backward()里面的参数实际上就是每一个输出对输入求导后的权重
            optimizer.step()  # 更新模型
            epoch_loss+=loss.item()  # .item()返回高精度数值
        print('epoch %d loss:%0.6f'%(epoch,epoch_loss))
    torch.save(model.state_dict(),'weights_%d.pth'%epoch)
    torch.save(model,'weights_%d_dc.pth'%epoch)  # 保存模型
    return model 

def train(train_data_path,train_gt_path):   # 训练模型
    batch_size=10
    liver_dataset=MyDataset(train_data_path,train_gt_path,transform=x_transforms,target_transform=y_transforms)
    dataloaders=DataLoader(liver_dataset,batch_size=batch_size,shuffle=True)
    # 获取训练数据
    model=UNet()   # 实例化模型
    criterion=FocalLoss()   # 调用损失函数
    optimizer=torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)#optim.Adam(model.parameters())
    # 调用Adagrad优化器
    train_model(model,criterion,optimizer,dataloaders)

def test(test_data_path,test_gt_path,save_pre_path):
    liver_dataset=MyDataset(test_data_path,test_gt_path,transform=x_transforms,target_transform=y_transforms)
    print(liver_dataset)
    model=UNet()
    dataloaders=DataLoader(liver_dataset,batch_size=1)
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        count=0
        for x,_ in dataloaders:
            start=time.clock()
            y=model(x)
            img_y=torch.squeeze(y).numpy()
            elapsed=time.clock()-start   # 计算程序运行的cpu时间
            print('Time used:',elapsed)
            plt.plot(img_y)   # 画出分割模型
            count+=1
            plt.show()

if __name__=='__main__':
    pretrained=False 
    train_data_path='C:/Users/MSI1/work/Brain/Brats2018FoulModel2D/trainImage'   # 训练集数据存储路径
    train_gt_path='C:/Users/MSI1/work/Brain/Brats2018FoulModel2D/trainMask'   # 训练集标签存储路径
    train(train_data_path,train_gt_path)