import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt

def model(data):
    # 模型参数
    num_clusters = 3 #这个高斯混合模型由三个高斯混合而成
    dim = 2 #数据的维度是二维

    # 超参数
    alpha = torch.ones(num_clusters) / num_clusters
    mu = torch.randn(num_clusters, dim)
    sigma = torch.eye(dim).expand(num_clusters, dim, dim) #二维数据的sigma是一个2乘2的矩阵
    #使用expand扩展eye(dim)矩阵，

    # 隐变量
    z = pyro.sample('z', dist.Categorical(alpha), obs=None)

    # 观测变量
    with pyro.plate('data', len(data)):
        pyro.sample('obs', dist.MultivariateNormal(mu[z], sigma[z]), obs=data)
        #MultivariateNormal用于生成一个多元正态分布


def guide(data):
    # 模型参数
    num_clusters = 3
    dim = 2

    # 变分参数
    alpha_q = pyro.param('alpha_q', torch.ones(num_clusters) / num_clusters,
                         constraint=dist.constraints.simplex)
    #dist.constraints.simplex:简单形式约束条件是指，向量中的所有元素都必须大于等于0，且所有元素之和必须等于1
    mu_q = pyro.param('mu_q', torch.randn(num_clusters, dim))
    sigma_q = pyro.param('sigma_q', torch.eye(dim).expand(num_clusters, dim, dim),
                         constraint=dist.constraints.positive_definite)

    # 隐变量
    pyro.sample('z', dist.Categorical(alpha_q))

losses = []
def train(data):
    # 模型参数
    num_clusters = 3

    # 损失函数
    svi = pyro.infer.SVI(model=model,
                         guide=guide,
                         optim=pyro.optim.Adam({'lr': 0.01}),
                         loss=pyro.infer.Trace_ELBO())

    # 训练模型
    for i in range(1000):
        loss = svi.step(data)
        losses.append(loss)
        if i % 100 == 0:
            print("Epoch ", i, " Loss ", loss)


# 数据集
#生成一个大小为(300, 2)的张量data。
# 其中，前100行表示一个均值为(5, 0)，方差为1的二维高斯分布；
# 接下来的100行表示一个均值为(-5, 0)，方差为1的二维高斯分布；
# 最后的100行表示一个均值为(0, 5)，方差为1的二维高斯分布。
data = torch.cat([torch.randn(100, 2) + torch.tensor([5, 0]),
                  torch.randn(100, 2) + torch.tensor([-5, 0]),
                  torch.randn(100, 2) + torch.tensor([0, 5])])

x = data[:, 0]
y = data[:, 1]

plt.scatter(x, y)
plt.show()


# 训练模型
train(data)

plt.figure(figsize=(6,5))
plt.plot(losses)
plt.xlabel("SVI step")
plt.ylabel("ELBO loss")
plt.show()