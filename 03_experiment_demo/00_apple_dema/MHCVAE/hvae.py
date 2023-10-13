#这个文件也是hvae但是没有加mcmc作为对比实验
import torch
import torch.nn as nn
from tqdm import tqdm
#from UHA import UHA
import torch.nn.functional as F



class HVAE(nn.Module):
    def __init__(self, input_size, hidden_size1, latent_size1, hidden_size2, latent_size2):
        super(HVAE, self).__init__()
        
        # First Encoder 1000 -> 2
        self.encoder1 = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, 64),
            nn.ReLU(),
            nn.Linear(64, latent_size1*2)
        )
        
        # Second Encoder 2000 -> 2
        self.encoder2 = nn.Sequential(
            nn.Linear(latent_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, latent_size2)
        )
        
        # First Decoder 2->
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, input_size)
        )
        
        # Second Decoder
        self.decoder2 = nn.Sequential(
            nn.Linear(1, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, latent_size1),
            nn.ReLU(),
            nn.Linear(latent_size1, input_size)
        )
        
    def encode(self, x):
        z1 = self.encoder1(x)
        z2 = self.encoder2(z1)
        return z1, z2
    
    def decode(self, z1, z2):
        x_recon1 = self.decoder1(z1)
        x_recon2 = self.decoder2(z2)
        return x_recon1, x_recon2
    
    #重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # Encode 两次编码
        mu1, logvar1 = self.encoder1(x).chunk(2, dim=-1)
        z1 = self.reparameterize(mu1, logvar1)
        mu2, logvar2 = self.encoder2(z1).chunk(2, dim=-1)
        z2 = self.reparameterize(mu2, logvar2)

        # Decode 两次解码
        x_recon1 = self.decoder1(z1)
        x_recon2 = self.decoder2(z2)
        
        # Compute Loss
        #计算重构误差，就是经过vae前的数据和vae后的数据的区别 这一项就是ELBO中的交叉熵
        #重构误差1：计算输入数据和租后一次解码输出之间的重构误差
        recon_loss1 = nn.functional.mse_loss(x_recon1, x, reduction='sum') 
        #重构误差2：计算第二次编码的输入数据和第一次解码的输出之间的鸿沟误差
        recon_loss2 = nn.functional.mse_loss(x_recon2, x, reduction='sum')
        #计算两次的kl散度，也就是q(z)和p(z)之间的差距
        # KL(q(z|x) || p(z|x)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # 在这里我们一般认为p(z|x)为N(0,1)
        kld_loss1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
        kld_loss2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
        total_loss = recon_loss1 + recon_loss2 + kld_loss1 + kld_loss2
        
        return total_loss



import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 训练模型
def main():
    losses = [] #将训练中loss的变化情况保存在losses列表中
    # 随机生成数据
    x = torch.randn(1000, 1)
    x = x.reshape((1, 1000))
    # 创建模型、优化器和损失函数
    model = HVAE(input_size=1000, hidden_size1=256, latent_size1 =2, hidden_size2= 2, latent_size2= 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 开始训练
    num_epochs = 5000
    pbar = tqdm(range(num_epochs))
    for epoch in range(num_epochs):
        # 前向传播
        loss = model.forward(x)
        losses.append(loss.detach().numpy())
        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        description = f"Epoch {epoch}: Loss={loss:.2f}"
        pbar.set_description(description)
        pbar.update(1)
    
    #绘制losses变化曲线图，并将其保存
    # 绘制曲线图
    x = np.arange(len(losses))
    plt.plot(x, losses)
    
    # 添加标题和轴标签
    plt.title('Loss over time')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('./01_img/losses_5000.jpg')
    #将数据保存在excel表格中
    df = pd.DataFrame(losses[3000:])
    df.to_excel('./00_data/losses.xlsx', index=False)
    #print('Epoch [{}/{}], Loss: {:.4f},'.format(epoch+1, num_epochs, loss.item()))
    # 返回训练好的模型
    return model

#主程序
main()