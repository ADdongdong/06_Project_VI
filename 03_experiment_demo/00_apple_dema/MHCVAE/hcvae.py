import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import utils
from torch.utils.data import DataLoader


class HCVAE(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""

    def __init__(self, feature_size, class_size, latent_size):
        super(HCVAE, self).__init__()

        # 定义均方差对象
        self.Loss_MSE = torch.nn.MSELoss()

        # 定义网络
        self.fc2_mu = nn.Linear(200, latent_size)
        self.fc2_log_std = nn.Linear(200, latent_size)
        # 编码
        self.encoder_fc1 = nn.Linear(feature_size + class_size, 200)
        self.encoder_fc2 = nn.Linear(latent_size + class_size, 200)
        self.encoder_fc3 = nn.Linear(latent_size + class_size, 200)

        # 解码
        self.decoder_fc1 = nn.Linear(latent_size + class_size, 200)
        self.decoder_fc2 = nn.Linear(latent_size + class_size, 200)
        self.decoder_fc3 = nn.Linear(latent_size + class_size, 200)
        self.decoder_mu = nn.Linear(200, feature_size)
        self.decoder_log_std = nn.Linear(200, feature_size)

    def encode(self, func, x, y):
        # concat features and labels
        h1 = F.relu(func(torch.cat([x, y], dim=1)))
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, func, z, y):
        # concat latents and labels
        h3 = F.relu(func(torch.cat([z, y], dim=1)))
        # 这里decoder也是先decoder出来均值和方差，因为，后面计算loss函数要用
        # 在decoder后再使用reparametrize重采样出来一个z放入下一层解码
        de_mu = self.fc2_mu(h3)
        de_log_std = self.fc2_log_std(h3)
        return de_mu, de_log_std

    def final_decode(self, z, y):
        h3 = F.relu(self.decoder_fc3(torch.cat([z, y], dim=1)))
        mu = self.decoder_mu(h3)
        log_std = self.decoder_log_std(h3)
        return mu, log_std

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x, y):
        # 第一次条件编码
        mu_1, log_std_1 = self.encode(self.encoder_fc1, x, y)
        z_1 = self.reparametrize(mu_1, log_std_1)

        # 第二次条件编码
        mu_2, log_std_2 = self.encode(self.encoder_fc2, z_1, y)
        z2 = self.reparametrize(mu_2, log_std_2)

        # 第三次条件编码
        mu_3, log_std_3 = self.encode(self.encoder_fc3, z2, y)
        z3 = self.reparametrize(mu_3, log_std_3)

        # 第一次条件解码
        de_mu1, de_log1 = self.decode(self.decoder_fc1, z3, y)
        de_z1 = self.reparametrize(de_mu1, de_log1)

        # 第二次条件解码
        de_mu2, de_log2 = self.decode(self.decoder_fc2, de_z1, y)
        de_z2 = self.reparametrize(de_mu2, de_log2)

        # 第三次条件解码
        de_mu3, de_log3 = self.final_decode(de_z2, y)
        de_z3 = self.reparametrize(de_mu3, de_log3)

        # 根据计算loss函数所用到的内容
        # 将编码得到的方差打包成数组
        en_log = [log_std_1, log_std_2, log_std_3]
        # 将编码和解码得到的均值打包成方差
        en_mu = [mu_1, mu_2, mu_3]
        de_mu = [de_mu1, de_mu2, de_mu3]

        loss = self.loss_function(en_log, en_mu, de_mu)
        return loss

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


# 训练模型
def test1():
    losses = []  # 将训练中loss的变化情况保存在losses列表中
    # 随机生成数据
    x = torch.randn(1000, 1)
    x = x.reshape((1, 1000))
    # 生成条件
    y = torch.randn(10, 1)
    y = y.reshape((1, 10))
    # 创建模型、优化器和损失函数
    model = HCVAE(feature_size=1000, class_size=10, latent_size=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 开始训练
    num_epochs = 5000
    for epoch in range(num_epochs):
        # 前向传播
        loss = model.forward(x, y)
        # losses.append(loss.detach().numpy())
        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss={loss:.2f}")

    # 绘制losses变化曲线图，并将其保存
    # 绘制曲线图
    x = np.arange(len(losses))
    plt.plot(x, losses)

    # 添加标题和轴标签
    plt.title('Loss over time')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    # 将数据保存在excel表格中

    return model


def test2():
    epochs = 100
    batch_size = 100

    recon = None
    img = None

    utils.make_dir("./img/cvae")
    utils.make_dir("./model_weights/cvae")

    train_data = torchvision.datasets.MNIST(
        root='./mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    data_loader = DataLoader(train_data, batch_size=100, shuffle=True)

    cvae = HCVAE(feature_size=784, class_size=10, latent_size=10)

    optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)

    for epoch in range(100):
        train_loss = 0
        i = 0
        for batch_id, data in enumerate(data_loader):
            img, label = data
            inputs = img.reshape(img.shape[0], -1)
            y = utils.to_one_hot(label.reshape(-1, 1), num_class=10)
            #recon, mu, log_std = cvae(inputs, y)
            #loss = cvae.loss_function(recon, inputs, mu, log_std)
            loss = cvae.forward(inputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            i += 1

            if batch_id % 100 == 0:
                print("Epoch[{}/{}], Batch[{}/{}], batch_loss:{:.6f}".format(
                    epoch+1, epochs, batch_id+1, len(data_loader), loss.item()))

        print("======>epoch:{},\t epoch_average_batch_loss:{:.6f}============".format(
            epoch+1, train_loss/i), "\n")

        # save imgs
        if epoch % 10 == 0:
            imgs = utils.to_img(recon.detach())
            path = "./img/cvae/epoch{}.png".format(epoch+1)
            torchvision.utils.save_image(imgs, path, nrow=10)
            print("save:", path, "\n")

    torchvision.utils.save_image(img, "./img/cvae/raw.png", nrow=10)
    print("save raw image:./img/cvae/raw/png", "\n")

    # save val model
    utils.save_model(cvae, "./model_weights/cvae/cvae_weights.pth")


if __name__ == '__main__':
    test2()
