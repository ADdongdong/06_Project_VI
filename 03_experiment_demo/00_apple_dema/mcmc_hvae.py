import torch
import torch.distributions as dist
import torch.nn as nn
from tqdm import tqdm
from UHA import UHA
import torch.nn.functional as F
import numpy as np



class HVAE(nn.Module):
    def __init__(self, input_size, hidden_size1, latent_size1, hidden_size2, latent_size2):
        super(HVAE, self).__init__()
        
        #定义一个成员变量，这个变量在初始化HVAE的时候就创建
        #创建一个8行1列的列向量，每一行都是A1+zi线性组合的结果 
        self.List = torch.empty(8, 2)#, requires_grad=True)
        
        #通过numpy读取数据,这个数据在模型被创建的时候就导入
        self.a1_a8 = np.load('./00_data/mean_var_list.npy', allow_pickle=True)
        
        # First Encoder 1000 -> 2
        self.encoder1 = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, 64),
            nn.ReLU(),
            nn.Linear(64, latent_size1*2)
        )
        
        # 第二次编码，将8个因素和第一次编码的结果做和，
        # 然后分成8个编码器分别进行编码
        self.encoder2 = nn.Sequential(
            nn.Linear(latent_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, latent_size2)
        )
        
        # 对第二次编码的结果进行投影定理运算
        # 对投影定理运算的结果放入第三层编码器
        self.encoder3 = nn.Sequential(
            nn.Linear(2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, latent_size2*2)
        )
        
        # First Decoder 2->
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, input_size)
        )
        
        # Second Decoder
        self.decoder2 = nn.Sequential(
            nn.Linear(1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, latent_size1),
        )

        self.decoder3 = nn.Sequential(
            nn.Linear(latent_size1, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, input_size)
        )
        
        
    #重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    

    #将A1-A8这8个先验因素加入到网络中，和第一次编码的结果做线性和
    def AddA1_A8_encoder(self, index, z1):
        a_mu = torch.tensor(self.a1_a8[index][0], requires_grad=True)
        a_logvar = torch.tensor(self.a1_a8[index][1], requires_grad=True)
        A1 = self.reparameterize(a_mu, a_logvar)
        # 经过正则化流学习出来的先验和第一次编码学习出来的先验做线性和
        aizi = A1 + z1
        #将计算的结果送入encoder2
        mu2, logvar2 = self.encoder2(aizi).chunk(2, dim=-1)
        return mu2, logvar2


    #定义第二次编码的函数
    def Encoder2(self, z1):
        """
            这个函数将A1-A8个8个先验因素加入到的编码器中
            并且，以8个因素和第一次编码的结果第二次编码的输入
            第二次编码对于8个因素有对应的8个编码器
            第二次编码结果以list形式输出，每个元素是对应因素的均值和方差
            参数: 
                第一次编码的结果,mu1,logvar1,重参数化以后的结果
            返回值：
                个因素的均值和方差
                以均值数组和方差数组的形式返回
        """
        #第二次编码和第一次解码都是8个因素对应的8个编码器和8个解码器，所以，这个过程放在for循环中
        encoder2_mu = []
        encoder2_logvar = []
        for i in range(8):
            #对8个因素进行编码，这里编码的时候，会添加进去使用正则化流计算出来的先验
            mu2, logvar2 = self.AddA1_A8_encoder(i, z1)
            encoder2_mu.append(mu2)
            encoder2_logvar.append(logvar2)
        return encoder2_mu, encoder2_logvar


    #定义第三次编码函数
    def Encoder3(self, mu_list, logvar_list):
        """
        第三次编码接受的参数是8个因素经过投影定理处理过后因素。
        参数：
            包含8个因素经过重参数以后z1-z8的list
        返回：
            返回8个因素经过第三次编码以后的均值和方差，以待UHA对mu和logvar进行优化
        """
        encoder3_mu = []
        encoder3_logvar = []
        for i in range(8):
            # 进项重参数化
            zi =  self.reparameterize(mu_list[i], logvar_list[i])
            #print(zi.size())
            mu ,logvar = self.encoder3(zi).chunk(2, dim=-1)
            encoder3_logvar.append(logvar)
            encoder3_mu.append(mu)
        return encoder3_mu, encoder3_logvar


    #使用uha对mu和logvar进行优化
    def UHA_Optim(self, mu_list, logvar_list):
        """
        这个函数对第三次编码后的8个因素的均值和方差进行UHA优化
        参数：
            经过第三次编码后的8个因素的均值和方差组成的list
        返回值：
            经过UHA优化后的8个因素的均值和方差，
            并且，使用重参数化技巧，将其组织，方便第一层解码器进行解码
            因为是8个编码器，所以，也会生成8个对应的z
        """
        z3_list = []
        for i in range(8):
            uha = UHA(2, None, L_m=10, step_size=0.1)
            result = uha.sample(mu_list[i], logvar_list[i], 1)
            #print(result[0])
            uha_mu2 = torch.tensor(result[0][0]).requires_grad_(True)
            uha_logvar2 = torch.tensor(result[0][1]).requires_grad_(True)

            #对优化过后的数据进行重参数化，以便放入解码器进行解码
            z3 = self.reparameterize(uha_mu2, uha_logvar2)
            z3_list.append(z3) 
            #print("对第" + str(i) + "个因素进行uha优化完毕")

        return z3_list


    #定义投影定理计算函数
    def Projection(self, encoder2_mu_list, encoder2_logvar_list):
        """
        使用投影定理，对第二次编码，8个编码器得到的8个因素进行投影定理计算
        参数：
            第一个参数为8个因素编码得到的均值列表
            第二个参数为8个因素编码得到的方差列表
        返回：
            返回8个因素经过投影定理计算以后得出的新结果，
            这个结果将会放入第三次编码器中。所以最后使用重参数对8个因素进行处理。
            最后的结果放在一个list中。
            第三次编码还是8个编码器
        """
        project_mu_list = []
        project_logvar_list = []
        for i in range(8):
            target_factor_mu = encoder2_mu_list[i]
            target_factor_logvar = encoder2_logvar_list[i]
            #收集剩余的额7个因素的均值方差
            temp_list_mu = []
            temp_list_logvar = []
            for j in range(8):
                if j != i:
                    temp_list_mu.append(encoder2_mu_list[j])
                    temp_list_logvar.append(encoder2_logvar_list[j])

            #计算投影向量，即，计算这7个因素的平均值
            projection_mu = torch.mean(torch.cat(temp_list_mu, dim=0), dim=0)
            projection_logvar = torch.mean(torch.cat(temp_list_logvar, dim=0), dim=0)

            #计算投影系数
            projection_coefficients_mu = projection_mu / torch.norm(target_factor_mu)
            projection_coefficients_logvar = projection_logvar / torch.norm(target_factor_logvar)
            project_mu_list.append(projection_coefficients_mu)
            project_logvar_list.append(projection_coefficients_logvar)

        return project_mu_list, project_logvar_list


    #
    def loss_function(self,):
        """
        计算整个层次变分编码器的误差函数，包括重构误差和KL散度
        参数：
            每层编码器的输入和输出
        返回值：
            总和计算出来的loss值
        """
        # 计算loss函数
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
        pass


    def forward(self, x):
        # Encode 3次编码
        #第一次编码
        mu1, logvar1 = self.encoder1(x).chunk(2, dim=-1)
        z1 = self.reparameterize(mu1, logvar1)

        #第二次编码
        encoder2_mu, encoder2_logvar = self.Encoder2(z1)
        print("encoder_mu", len(encoder2_mu))
        print("encoder_logvar", len(encoder2_logvar))
        print("mu2.size:", encoder2_mu[0].size())
        #投影定理计算第三次编码的输入
        project_mu, project_logvar = self.Projection(encoder2_mu, encoder2_logvar)

        print("project_mu:", project_mu)
        print("project_logvar:", project_logvar)

        #第三次编码
        mu3_list, logvar3_list = self.Encoder3(project_mu, project_logvar)
        print("len(mu3_list)", len(mu3_list)) 
        print("mu3.tensor.size", mu3_list[0].size())

        #这里对第3次编码的结果并结果投影定理计算的结果进行UHA优化
        z3_list = self.UHA_Optim(mu3_list, logvar3_list)


        # Decode 3次解码,
        #第一次解码，输入为z3是uha优化过的数据
        x_recon1 = self.Decoder1(z3_list)

            
        #进行第二次解码
        x_recon2 = self.decoder2(x_recon2)


        #进行第三次解码
        x_recon3 = self.decoder3(x_recon3) 
        
        #计算loss这个变分自编码器的Loss函数
        total_loss = self.loss_function()

        return total_loss

    
   