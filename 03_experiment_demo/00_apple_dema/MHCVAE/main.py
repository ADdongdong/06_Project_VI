from mcmc_hvae import HVAE
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
    model = HVAE(input_size=1000, hidden_size1=256, latent_size1 =8, hidden_size2= 2, latent_size2= 4)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 开始训练
    num_epochs = 400
    #pbar = tqdm(range(num_epochs))
    for epoch in range(num_epochs):
        # 前向传播
        loss = model.forward(x)
        losses.append(loss.detach().numpy())
        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch[{epoch}/{num_epochs}], batch_loss:{loss}")
        # description = f"Epoch {epoch}: Loss={loss:.2f}"
        # pbar.set_description(description)
        # pbar.update(1)
    
    #绘制losses变化曲线图，并将其保存
    # 绘制曲线图
    x = np.arange(len(losses))
    plt.plot(x, losses)
    
    # 添加标题和轴标签
    plt.title('Loss over time')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('./01_img/mcmc_losses.jpg')
    #将数据保存在excel表格中
    df = pd.DataFrame(losses[2000:])
    df.to_excel('./00_data/mcmc_losses.xlsx', index=False)
    #print('Epoch [{}/{}], Loss: {:.4f},'.format(epoch+1, num_epochs, loss.item()))
    # 返回训练好的模型
    return model


#主程序
main()