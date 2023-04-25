import pyro
import numpy
import torch
import pyro.distributions as dist
import pyro.optim as optim
import pyro.infer as infer

from pyro.infer import SVI, Trace_ELBO, ReweightedWakeSleep
from pyro.optim import Adam
from pyro.distributions import constraints


# 定义模型
def model(data):
    # 定义先验分布
    mix = pyro.sample("mix", dist.Dirichlet(torch.tensor([1.0, 1.0])))
    locs = pyro.sample("locs", dist.Normal(torch.tensor([0.0, 0.0]), torch.tensor([10.0, 10.0])).to_event(1))
    scales = pyro.sample("scales", dist.InverseGamma(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])).to_event(1))

    # 生成观测值
    with pyro.plate("data", len(data)):
        #从混合分布中采样(每一个种类，代表一个高斯分布）
        z = pyro.sample("z", dist.Categorical(mix))
        #从对应的高斯分布中采样
        pyro.sample("obs", dist.Normal(locs[z], scales[z]), obs=data)


# 定义变分分布 定义变分推断函数，用于学习高斯混合模型的潜在变量
def guide(data):
    # 定义变分分布
    mix_alpha = pyro.param("mix_alpha", torch.tensor([1.0, 1.0]), constraint=constraints.positive)
    locs_mu = pyro.param("locs_mu", torch.tensor([0.0, 0.0]))
    locs_sigma = pyro.param("locs_sigma", torch.tensor([1.0, 1.0]), constraint=constraints.positive)
    scales_alpha = pyro.param("scales_alpha", torch.tensor([1.0, 1.0]), constraint=constraints.positive)
    scales_beta = pyro.param("scales_beta", torch.tensor([1.0, 1.0]), constraint=constraints.positive)

    mix = pyro.sample("mix", dist.Dirichlet(mix_alpha))
    locs = pyro.sample("locs", dist.Normal(locs_mu, locs_sigma).to_event(1))
    scales = pyro.sample("scales", dist.InverseGamma(scales_alpha, scales_beta).to_event(1))


# 准备数据
data = torch.cat([torch.randn(100) + 2.5, torch.randn(100) - 2.5])

# 定义 SVI 对象
#svi = SVI(model, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())
svi = SVI(model, guide, Adam({"lr": 0.01}), loss= Trace_ELBO())

# 训练模型
num_iterations = 1000
for i in range(num_iterations):
    loss = svi.step(data)
    if i % 100 == 0:
        print("iteration {}: loss = {}".format(i, loss / len(data)))

# 进行推断
params = {
    "mix": pyro.param("mix_alpha").detach().numpy(),
    "locs": pyro.param("locs_mu").detach().numpy(),
    "scales": pyro.param("scales_alpha").detach().numpy() / pyro.param("scales_beta").detach().numpy()
}

print(params)

# 对新的观测值进行采样
with torch.no_grad():
    z = dist.Categorical(torch.tensor(params["mix"])).sample(torch.Size([100]))
    samples = dist.Normal(torch.tensor(params["locs"])[z], torch.tensor(params["scales"])[z]).sample()

print(samples)
