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
        
        # 定义均方差对象
        self.Loss_MSE = torch.nn.MSELoss()

        # 定义投影定理系数计算要用的线性模型
        self.linear_regression = torch.nn.Linear(8,1)

        #定义随机梯度下降优化器
        self.optim_SGD = torch.optim.SGD(self.linear_regression.parameters(), lr=0.01)

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
            nn.Linear(latent_size1*2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 16)
        )
        
        # 对第二次编码的结果进行投影定理运算
        # 对投影定理运算的结果放入第三层编码器
        self.encoder3 = nn.Sequential(
            nn.Linear(24, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, latent_size2*2)
        )
        
        # First Decoder 2->
        self.decoder1 = nn.Sequential(
            nn.Linear(12, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 8)
        )
        
        # Second Decoder
        self.decoder2 = nn.Sequential(
            nn.Linear(16, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 8),
        )

        self.decoder3 = nn.Sequential(
            nn.Linear(16, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, 1000)
        )
        
        
    #重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        #生成噪声
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    

    #将A1-A8这8个先验因素加入到网络中,CVAE
    def catA1_A8_encoder(self, index, z1):
        #将这8个因素，分别与第一次编码的结果做cat
        #作为第二次编码的输入，即为条件变分自编码器
        a_mu = torch.tensor(self.a1_a8[index][0], requires_grad=True)
        a_logvar = torch.tensor(self.a1_a8[index][1], requires_grad=True)
        A1 = self.reparameterize(a_mu, a_logvar) 
        A1 = A1.expand([1, 8])

        # 将8个因素的与待输入编码器的数据做cat操作，即为条件变分自编码
        aizi = torch.cat([A1, z1], dim = 1)

        #将计算的结果送入encoder2
        mu2, logvar2 = self.encoder2(aizi).chunk(2, dim=-1)
        return mu2, logvar2


    # 定义第二次编码的函数CVAE
    def Encoder2(self, z1):
        """
        为条件变分自编码，要将8个因素作为label与数据进行拼接,
        然后整体放入编码器中，这样就是CVAE
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
            mu2, logvar2 = self.catA1_A8_encoder(i, z1)
            encoder2_mu.append(mu2)
            encoder2_logvar.append(logvar2)
        return encoder2_mu, encoder2_logvar


    #定义第三次编码函数
    def Encoder3(self, projection_list, mu_list, logvar_list):
        """
            第三次编码,也是条件编码，将encoder2学习出来的zi_2放入编码器。
            将使用投影地理算出来的元素作为条件放入编码器中。
            将正则化流学习出来的先验A1-A8也放入其中
        参数：
            projection_list 经过投影定理算出的a1`-a8`
            mu_list 第二次编码得到的8个因素的mu列表
            logvar_list 第二次编码得到的8个因素的logvar列表
        返回：
            返回8个因素经过第三次编码以后的均值和方差，以待UHA对mu和logvar进行优化
        """
        encoder3_mu = []
        encoder3_logvar = []
        for i in range(8):
            # 进项重参数化
            zi =  self.reparameterize(mu_list[i], logvar_list[i])
        
            #将A1和A8加入
            a_mu = torch.tensor(self.a1_a8[i][0], requires_grad=True)
            a_logvar = torch.tensor(self.a1_a8[i][1], requires_grad=True)
            Ai = self.reparameterize(a_mu, a_logvar) 
            Ai = Ai.expand([1, 8])


            # 将zi和投影定理计算出来的结果做cat操作，作为条件
            aizi = torch.cat([zi, projection_list[i], Ai], dim=1)
            
            # 放入第三层编码器编码
            mu ,logvar = self.encoder3(aizi).chunk(2, dim=-1)
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
        '''
            投影定理，第二次编码有8个编码器
            这8个编码器出来8个zi, 我们假设这8个zi都是8维的向量
            然后拿出来z1,用z2-z8去线性表示z1
            z1= a2*z2 + a2*z3 + ... + a2*z8
            最后得到的系数就是我们要求的结果。
            这里的a2-a8可以用随机梯度下降将系数学习出来
        '''
        ai_list = []
        for i, j in zip(encoder2_mu_list, encoder2_logvar_list):
            zi_2 = self.reparameterize(i, j)
            ai_list.append(zi_2)
        
        #print("len(ai_list)", len(ai_list))

        #定义一个0向量
        
        zero_vector = torch.zeros([1, 8], requires_grad=True)
        #print("zero_vector", zero_vector)
        #ai_list 中保存的就是a1-a8的8个向量
        #定义project_list,ai投影在a1-a8上的参数向量保存在其中
        projection_list = []
        for i in range(8):
            #定义输入特征和目标值
            target_vector = ai_list[i][0]
            
            #print("target_vector", target_vector)
            #定义7个基向量
            intput_vector = ai_list[:i] + [zero_vector]+ ai_list[i+1:]
            #将数据转化为线性模型可以接受的张量
            intput_vector = torch.cat(intput_vector).reshape(len(intput_vector), -1)

            
            num_epches = 1000
            for epoch in range(num_epches):
                #向前传播
                prediction = self.linear_regression(torch.tensor(intput_vector))

                #计算损失
                loss = self.Loss_MSE(prediction.squeeze(), target_vector)

                #反向传播和优化
                self.optim_SGD.zero_grad() #梯度清零
                # 保存中间的梯度
                loss.backward(retain_graph=True) #计算梯度
                self.optim_SGD.step() #更新参数
            
            #获取线性回归模型的权重
            weight = self.linear_regression.weight.data
            # print("weight.size", weight.size())
            # print("weight", weight)

            #将每个因素投影的结果添加到结果列表中
            projection_list.append(weight)
            print("第A" + str(i+1) + "个因素的投影计算完毕...")

        return  projection_list


    #定义Decoder函数
    def Decoder(self, z_list, decoder) -> list:
        """
            decoder函数用来解码
            参数：
                z_list:当次送入解码器的输入内容
                decoder:本次所要用到的解码器
            返回值：
                返回当前次编码器8个因素编码的结果所组成的列表
        """
        # 定义第一次加码的结果
        decoder_rec_list = []
        # 对于条件变分自编码器，解码的时候也要将条件加进去
        for index in range(8):
            a_mu = torch.tensor(self.a1_a8[index][0], requires_grad=True)
            a_logvar = torch.tensor(self.a1_a8[index][1], requires_grad=True)
            Ai = self.reparameterize(a_mu, a_logvar)
            Ai = Ai.expand([1, 8])
            #添加条件以后Decoder1的输入
            aizi = torch.cat([z_list[index] , Ai], dim=1)
            #print(f"aizi.siez{aizi.size()}")
            #进行Decoder1解码
            di = decoder(aizi)

            #将因素i第一次解码的结果添加到decoder1_rec_list中
            decoder_rec_list.append(di)

        return decoder_rec_list


        

    #定义loss函数
    def loss_function(self, rec_list, input_list, mu_logvar):
        """
        计算整个层次变分编码器的误差函数，包括重构误差和KL散度
        参数：
            每层编码器的输入和输出
        返回值：
            总和计算出来的loss值
        """
        total_loss = torch.tensor(0)

        #计算重构误差1
        recon_loss_x = torch.tensor(0)
        for i in range(8):
            #重构误差1：计算输入数据和租后一次解码输出之间的重构误差
            recon_loss1 = nn.functional.mse_loss(rec_list[2][i], input_list[0], reduction='sum') 
            recon_loss_x = recon_loss_x + recon_loss1
        print("重构误差1计算成功")
        
        #计算重构误差2
        recon_loss_z1 = torch.tensor(0)
        for i in range(8):
            #重构误差2：计算第二次编码的输入数据和第一次解码的输出之间的鸿沟误差
            recon_loss2 = nn.functional.mse_loss(rec_list[1][i], input_list[1], reduction='sum')
            recon_loss_z1 = recon_loss_z1 + recon_loss2
        print("重构误差2计算成功")

        #计算重构误差3
        recon_loss_z2 = torch.tensor(0) 
        for i in range(8):
            #通过mu_list和logvar_list进行重参数化
            z2 = self.reparameterize(mu_logvar[1][0][i], mu_logvar[1][1][i])
            recon_loss3 = nn.functional.mse_loss(rec_list[0][i], z2, reduction='sum')
            recon_loss_z2 = recon_loss_z2 + recon_loss3
        print("重构误差3计算成功")


        #计算3次的kl散度，也就是q(z)和p(z)之间的差距
        # KL(q(z|x) || p(z|x)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # 在这里我们一般认为p(z|x)为N(0,1)
        # kld_loss1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
        # kld_loss2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
        # kld_loss3 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
        
        total_loss_tmp = recon_loss_x + recon_loss_z1 + recon_loss_z2 

        total_loss = total_loss_tmp + total_loss

        return total_loss 


    def forward(self, x):
        # Encode 3次编码
        #第一次编码
        mu1, logvar1 = self.encoder1(x).chunk(2, dim=-1)
        print("第一次编码结束...")
        z1 = self.reparameterize(mu1, logvar1)

        #第二次编码
        encoder2_mu, encoder2_logvar = self.Encoder2(z1)
        print("第二次编码结束...")


        #投影定理计算第三次编码的输入
        project_list  = self.Projection(encoder2_mu, encoder2_logvar)


        #第三次编码,条件变分自编码
        mu3_list, logvar3_list = self.Encoder3(project_list, encoder2_mu, encoder2_logvar)
        print("第三次编码结束...")
        print("len(mu3_list)", len(mu3_list)) 
        print("mu3.tensor.size", mu3_list[0].size())

        #这里对第3次编码的结果并结果投影定理计算的结果进行UHA优化
        z3_list = self.UHA_Optim(mu3_list, logvar3_list)
        print(f"len(z3_list){len(z3_list)}")
        print(f"size(z3_list){z3_list[0].size()}")


        # Decode 3次解码,
        #第一次解码，输入为z3是uha优化过的数据
        x_recon1 = self.Decoder(z3_list, self.decoder1)
        print("第一次解码完成")
            
        #进行第二次解码
        x_recon2 = self.Decoder(x_recon1, self.decoder2)
        print("第二次解码完成")

        #进行第三次解码
        x_recon3 = self.Decoder(x_recon2, self.decoder3) 
        print("第三次解码完成")
        
        #打包要计算loss所要用到的数据
        recon_list = [x_recon1, x_recon2, x_recon3]
        input_list = [x, z1]
        mu_logvar = [[mu1, logvar1], [encoder2_mu, encoder2_logvar], [mu3_list, logvar3_list]]
        #计算loss这个变分自编码器的Loss函数
        total_loss = self.loss_function(recon_list, input_list, mu_logvar)

        return total_loss

    
   