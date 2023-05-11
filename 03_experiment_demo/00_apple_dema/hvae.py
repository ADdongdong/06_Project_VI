import torch
import torch.nn as nn
import nflows 
from tqdm import tqdm
from UHA import uha
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
            nn.Linear(hidden_size2, latent_size2*2)
        )
        
        # First Decoder 2->
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, input_size)
        )
        
        # Second Decoder
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_size2, hidden_size1),
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

    def compute_elbo(model, x, num_samples=10):
        """
        计算未校正的哈密顿动力学模型下的ELBO
        
        参数：
            model: 未校正的哈密顿动力学模型
            x: 输入数据，大小为[batch_size, input_size]
            num_samples: 采样数量
        
        返回：
            elbo: ELBO（Evidence Lower Bound）
        """
        
        batch_size, input_size = x.size()

        # 从后验分布q(z|x)中采样num_samples个样本
        z_samples = []
        for i in range(num_samples):
            z_q, _ = model.q_z(x)
            z_samples.append(z_q)

        # 将样本堆叠成张量
        z_samples = torch.stack(z_samples)

        # 计算解码器p(x|z)的对数似然,即重构误差
        x_logits = model.p_x(z_samples).view(num_samples, batch_size, -1)
        log_likelihood = F.log_softmax(x_logits, dim=-1).sum(-1).mean(0)

        # 计算后验分布q(z|x)与先验分布p(z)之间的KL散度
        z_p = model.sample_prior(num_samples)
        kl_divergence = torch.distributions.kl_divergence(
            torch.distributions.Normal(z_samples.mean(0), z_samples.std(0)),
            torch.distributions.Normal(z_p, torch.ones_like(z_p))
        ).sum(-1).mean(0)

        # 计算ELBO
        elbo = log_likelihood - kl_divergence

        return -elbo  # 返回负数，因为我们使用优化器最小化损失函数   
    
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
        recon_loss1 = nn.functional.mse_loss(x_recon1, x, reduction='sum') 
        recon_loss2 = nn.functional.mse_loss(x_recon2, x, reduction='sum')
        #计算两次的kl散度，也就是q(z)和p(z)之间的差距
        # KL(q(z|x) || p(z|x)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # 在这里我们一般认为p(z|x)为N(0,1)
        kld_loss1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
        kld_loss2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
        total_loss = recon_loss1 + recon_loss2 + kld_loss1 + kld_loss2
        
        return total_loss

