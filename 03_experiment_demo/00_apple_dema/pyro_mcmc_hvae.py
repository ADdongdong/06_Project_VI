import torch
import torch.nn as nn
from tqdm import tqdm
from UHA import UHA
import torch.nn.functional as F
import pyro
from pyro.infer.trace_elbo import Trace_ELBO
import pyro.distributions as dist


#继承Trace_ELBO，并进程重载
class CustomELBO(Trace_ELBO):
    def _get_trace(self, model, guide, *args, **kwargs):
        # Get the trace from the original implementation
        trace = super()._get_trace(model, guide, *args, **kwargs)

        # Access the sampled latent variables from the trace
        z_1 = trace.nodes["latent_1"]["value"]
        z_2 = trace.nodes["latent_2"]["value"]

        # 从解码器中获得重构数据
        loc_img_1 = model.decoder.forward(z_1)
        loc_img_2 = model.decoder.forward(z_2)

        # Compute the reconstruction loss
        x = kwargs["obs"]
        recon_loss = -dist.Bernoulli(loc_img_1).log_prob(x.reshape(-1, 784)).sum(-1)
        recon_loss += -dist.Bernoulli(loc_img_2).log_prob(x.reshape(-1, 784)).sum(-1)

        # Compute the KL divergence
        kl_divergence = 0.5 * (z_1 ** 2).sum(-1) - 0.5 * (z_2 ** 2).sum(-1)

        # Compute the ELBO
        elbo = recon_loss - kl_divergence

        # Update the trace with the ELBO value
        trace.add_node("elbo", value=elbo.sum())

        return trace



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
    
    #define the model P(x|z)p(z)
    def model(self, x):
        pyro.module()


    def forward(self, x):
        # Encode 两次编码
        mu1, logvar1 = self.encoder1(x).chunk(2, dim=-1)
        z1 = self.reparameterize(mu1, logvar1)
        mu2, logvar2 = self.encoder2(z1).chunk(2, dim=-1)
        #print('mu2', mu2)
        #print('mu2', logvar2)

        #这里对q(z|x)进行UHA优化
        uha = UHA(2, None, L_m=10, step_size=0.1)
        result = uha.sample(mu2, logvar2, 1)
        #print(result[0])
        uha_mu2 = torch.tensor(result[0][0]).requires_grad_(True)
        uha_logvar2 = torch.tensor(result[0][1]).requires_grad_(True)
    
        z2 = self.reparameterize(uha_mu2, uha_logvar2)

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

    
   