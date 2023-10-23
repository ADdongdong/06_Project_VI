import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
# import torchkeras

# hidden_size = 564


class CVAE(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""

    def __init__(self, feature_size: object, class_size: object, latent_size: object) -> object:
        # “”“这段代码的含义是首先找到CVAE的父类，然后将CVAE类的对象转化为父类的对象，
        super(CVAE, self).__init__()
        # 让后让这个“被转化”的对象调用自己的__init__()函数”“”
        self.fc1 = nn.Linear(feature_size + class_size, 500)
        self.fc2_mu = nn.Linear(500, latent_size)
        self.fc2_log_std = nn.Linear(500, latent_size)
        self.fc3 = nn.Linear(latent_size + class_size, 500)
        self.fc4 = nn.Linear(500, feature_size)
        self.fc5 = nn.Linear(feature_size, 500)
        self.fc6_mu = nn.Linear(500, feature_size)

    def encode(self, x, y):
        # concat features and labels
        h1 = F.relu(self.fc1(torch.cat([x, y], dim=0)))
        mu = self.fc2_mu(h1)
        log_std_t = self.fc2_log_std(h1)
        log_std = torch.sigmoid(log_std_t)
        return mu, log_std

    def decode(self, z, y):
        # concat latents and labels
        h3 = F.relu(self.fc3(torch.cat([z, y], dim=0)))
        # use sigmoid because the input image's pixel is between 0-1
        recon = torch.sigmoid(self.fc4(h3))
        return recon

    def reparametrize(self, mu, log_std):
        #std = torch.exp(log_std)
        std = log_std
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x, y):
        mu_e = []
        mu_d = []
        logstd = []
        # 第一次编码，得到均值和方差
        mu0, log_std0 = self.encode(x, y)
        mu_e.append(mu0)
        logstd.append(log_std0)
        z1 = self.reparametrize(mu0, log_std0)

        # 第二次编码
        #prediction_z1 = 0.5 * (z1 + prj_x)
        mu1, log_std1 = self.encode(z1, y)
        mu_e.append(mu1)
        logstd.append(log_std1)
        z2 = self.reparametrize(mu1, log_std1)

        # 第三次编码
        #prediction_z2 = 0.5 * (z2 + prj_x)
        mu2, log_std2 = self.encode(z2, y)
        mu_e.append(mu2)
        logstd.append(log_std2)
        z3 = self.reparametrize(mu2, log_std2)

        # 第四次编码
        #prediction_z3 = 0.5 * (z3 + prj_x)
        mu3, log_std3 = self.encode(z3, y)
        mu_e.append(mu3)
        logstd.append(log_std3)
        z4 = self.reparametrize(mu3, log_std3)

        # 第五次编码
        mu4, log_std4 = self.encode(z4, y)
        mu_e.append(mu4)
        logstd.append(log_std4)
        z5 = self.reparametrize(mu4, log_std4)

        # 第一次解码
        recon4 = self.decode(z5, y)
        mu_d4 = self.fc6_mu(self.fc5(recon4))
        mu_d.append(mu_d4)

        #第二次解码
        recon3 = self.decode(recon4, y)
        mu_d3 = self.fc6_mu(self.fc5(recon3))
        mu_d.append(mu_d3)
        #prediction_recon3 = 0.5 * (recon3 + prj_x)

        #第三次解码
        recon2 = self.decode(recon3, y)
        #prediction_recon2 = 0.5 * (recon2 + prj_x)
        mu_d2 = self.fc6_mu(self.fc5(recon2))
        mu_d.append(mu_d2)
        
        #第四次解码
        #prediction_recon2 = 0.5 * (recon2 + prj_x)
        recon1 = self.decode(recon2, y)
        mu_d1 = self.fc6_mu(self.fc5(recon1))
        mu_d.append(mu_d1)

        # 第五次解码
        #prediction_recon1 = 0.5 * (recon1 + prj_x)
        recon0 = self.decode(recon1, y)
        mu_d0 = self.fc6_mu(self.fc5(recon0))
        mu_d.append(mu_d0)
        # y_sum = torch.sum(y)
        # recon3_sum = torch.sum(recon3)
        # recon2_sum = torch.sum(recon2)
        # recon0_sum = torch.sum(recon0)
        # z3_sum = torch.sum(z3)
        # print("z3_sum", z3_sum)
        # z4_sum = torch.sum(z4)
        # print("z4_sum", z4_sum)
        # print("recon3_sum", recon3_sum)
        # print("recon2_sum", recon2_sum)
        # print("recon0_sum", recon0_sum)
        # print("y_sum", y_sum)
        # print(mu_e)
        #mu_et = torch.tensor(mu_e)
        #logstdt = torch.tensor(logstd)
        #mu_dt = torch.tensor(mu_d)
        recon = recon0
        return recon, mu_e, mu_d, logstd

    def loss_function(self, recon, x, mu_e, mu_d, log_std, level) -> torch.Tensor:
        # use "mean" may have a bad effect on gradients
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = 0
        for i in range(level-1):
            me = mu_e[i].clone().detach()
            #mu_e = torch.tensor(mu_e)
            md = mu_d[level-2-i].clone().detach()
            #mu_d = torch.tensor(mu_d)
            log_std_t = log_std[i].clone().detach()
            #log_std = torch.tensor(log_std)
            temp_kl_loss = torch.pow((me - md), 2) - log_std_t
            kl_loss = kl_loss + temp_kl_loss
        #kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = kl_loss - log_std[level-1]
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss


#model = torchkeras.kerasmodel(CVAE(40,40,40))
# print(model)
