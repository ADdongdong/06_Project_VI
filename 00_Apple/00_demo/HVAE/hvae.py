import torch
from torch import nn


class HVAE(nn.Module):

    def __init__(self):
        super(HVAE, self).__init__()
        #编码器1从784维降维到128维，拆分两个64维的μ1与σ1
        # [b, 784] => [b, 128]
        # u1: [b, 64]
        # sigma1: [b, 64]
        self.encoder1 = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # 编码器2从128维降维到20维，拆分两个10维的μ2与σ2
        # [b, 128] => [b, 20]
        # u2: [b, 10]
        # sigma2: [b, 10]
        self.encoder2 = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 20),
            nn.ReLU()
        )
        #解码器1，20维升维到128维
        # [b, 20] => [b,128]

        self.decoder1 = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 48),
            nn.ReLU(),
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        #解码器2，128维升为到784维
        # [b, 128] => [b,784]
        self.decoder2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

        self.criteon = nn.MSELoss()

    def forward(self, x):
        """
        :param x: [b, 1, 28, 28]
        :return:
        """
        batchsz = x.size(0)
        # flatten压平
        x = x.view(batchsz, 784)

        # encoder1
        # [b, 128], including μ1 and σ1
        h_ = self.encoder1(x)
        # [b, 128] => [b, 64] and [b, 64]
        mu1, sigma1 = h_.chunk(2, dim=1)
        # reparametrize trick（重参数化）, epison~N(0, 1)torch.randn该函数就是生成正态分布
        z1 = mu1 + sigma1 * torch.randn_like(sigma1)


        # encoder2
        # [b, 20], including μ2 and σ2
        h_ = self.encoder2(z1)
        # [b, 20] => [b, 10] and [b, 10]
        mu2, sigma2 = h_.chunk(2, dim=1)
        # reparametrize trick, epison~N(0, 1)同上
        z2 = mu2 + sigma2 * torch.randn_like(sigma2)

        # decoder1
        x_hat = self.decoder1(z2)
        # reshape



        # decoder2
        x_hat = self.decoder2(x_hat)
        # reshape重塑图片
        x_hat = x_hat.view(batchsz, 1, 28, 28)
        #计算KL散度
        kld2 = 0.5 * torch.sum(
            torch.pow(mu1, 2) +
            torch.pow(sigma1, 2) -
            torch.log(1e-8 + torch.pow(sigma1, 2)) - 1
        ) / (batchsz * 28 * 28)
        return x_hat, kld2