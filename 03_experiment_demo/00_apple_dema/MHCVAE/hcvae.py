import torch
from torch import nn
import torch.nn.functional as F


class HCVAE(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""

    def __init__(self, feature_size, class_size, latent_size):
        super(HCVAE, self).__init__()

        # 定义均方差对象
        self.Loss_MSE = torch.nn.MSELoss()

        # 定义网络
        self.fc1 = nn.Linear(feature_size + class_size, 200)
        self.fc2_mu = nn.Linear(200, latent_size)
        self.fc2_log_std = nn.Linear(200, latent_size)
        self.fc3 = nn.Linear(latent_size + class_size, 200)

    def encode(self, x, y):
        # concat features and labels
        h1 = F.relu(self.fc1(torch.cat([x, y], dim=1)))
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, z, y):
        # concat latents and labels
        h3 = F.relu(self.fc3(torch.cat([z, y], dim=1)))
        # 这里decoder也是先decoder出来均值和方差，因为，后面计算loss函数要用
        # 在decoder后再使用reparametrize重采样出来一个z放入下一层解码
        de_mu = self.fc2_mu(h3)
        de_log_std = self.fc2_log_std(h3)
        return de_mu, de_log_std

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x, y):
        # 第一次条件编码
        mu_1, log_std_1 = self.encode(x, y)
        z_1 = self.reparametrize(mu_1, log_std_1)

        # 第二次条件编码
        mu_2, log_std_2 = self.encode(z_1)
        z2 = self.reparametrize(mu_2, log_std_2)

        # 第三次条件编码
        mu_3, log_std_3 = self.encode(z2)
        z3 = self.reparametrize(mu_3, log_std_3)

        # 第一次条件解码
        de_mu1, de_log1 = self.decode(z3, y)
        de_z1 = self.reparametrize(de_mu1, de_log1)

        # 第二次条件解码
        de_mu2, de_log2 = self.decode(de_z1, y)
        de_z2 = self.reparametrize(de_mu2, de_log2)

        # 第三次条件解码
        de_mu3, de_log3 = self.decode(de_z2, y)
        de_z3 = self.reparametrize(de_mu3, de_log3)

        # 根据计算loss函数所用到的内容
        # 将编码得到的方差打包成数组
        en_log = [log_std_1, log_std_2, log_std_3]
        # 将编码和解码得到的均值打包成方差
        en_mu = [mu_1, mu_2, mu_3]
        de_mu = [de_mu1, de_mu2, de_mu3]
        return en_log, en_mu, de_mu

    def loss_function(self, en_log, en_mu, de_mu) -> torch.Tensor:
        # 根据hcvae的loss公式来计算loss函数
        # 方差部分为编码器的方差之和
        logvar_sum = -torch.sum(torch.log(en_log[0]))
        logvar_sum += -torch.sum(torch.log(en_log[1]))
        logvar_sum += -torch.sum(torch.log(en_log[2]))

        #  均值部分为编码器和解码器d对应的均值的均方差
        mu_sum = self.Loss_MSE(en_mu[0], de_mu[1])
        mu_sum += self.Loss_MSE(en_mu[1], de_mu[0])

        # 计算整体的loss函数
        loss = logvar_sum + mu_sum

        return loss
