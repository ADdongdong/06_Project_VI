import torch
import torch.nn as nn

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1000)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, latent_dim=2):
        # 编码
        z_mean_logvar = self.encoder(x)
        z_mean = z_mean_logvar[:, :latent_dim]
        z_logvar = z_mean_logvar[:, latent_dim:]
        z = self.reparameterize(z_mean, z_logvar)

        # 解码
        recon_x = self.decoder(z)

        # 返回重构结果、均值和对数方差
        return recon_x, z_mean, z_logvar

# 训练模型
def train_vae():
    # 随机生成数据
    x = torch.randn(1000, 1)
    x = x.reshape((1, 1000))
    # 创建模型、优化器和损失函数
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 开始训练
    num_epochs = 1000
    for epoch in range(num_epochs):
        # 前向传播
        recon_x, z_mean, z_logvar = model.forward(x)

        # 计算重构误差和 KL 散度
        recon_loss = criterion(recon_x, x)
        kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        # 计算总损失
        loss = recon_loss + kl_divergence

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失信息
        print('Epoch [{}/{}], Loss: {:.4f}, Recon Loss: {:.4f}, KL Divergence: {:.4f}'.format(
            epoch+1, num_epochs, loss.item(), recon_loss.item(), kl_divergence.item()))

    # 返回训练好的模型
    return model

train_vae()