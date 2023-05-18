from hvae import HVAE
import torch
import torch.nn as nn
from tqdm import tqdm

# 训练模型
def main():
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

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        description = f"Epoch {epoch}: Loss={loss:.2f}"
        pbar.set_description(description)
        pbar.update(1)

    #print('Epoch [{}/{}], Loss: {:.4f},'.format(epoch+1, num_epochs, loss.item()))
    # 返回训练好的模型
    return model


#主程序
main()