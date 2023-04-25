import torch
from nflows import transforms, distributions, flows
import matplotlib.pyplot as plt
import numpy as np



# Generate some random data
x = torch.randn(20)
print(x.shape)
data = x.reshape((len(x), 1))
print(data.shape)
print(data.dtype)

# Define the base distribution
base_distribution = distributions.StandardNormal((1,))

# Define the flow
num_transforms = 4
transforms_ = []
for _ in range(num_transforms):
    transforms_.append(transforms.MaskedAffineAutoregressiveTransform(
        features=1,
        hidden_features=4,
        context_features=None,
        # activation='ReLU',
        # activation: 激活函数，可以是任何PyTorch中支持的函数，默认是torch.nn.ReLU。
    ))
transform = transforms.CompositeTransform(transforms_)

# Define the distribution to learn
# 定义一个流模型
distribution = flows.Flow(transform, base_distribution)

# 可视化
data_nonopt = distribution.sample(10000)
print("【未经过优化】均值：", data_nonopt.mean())
print("【未经过优化】方差：", data_nonopt.var())
data_nonopt = data_nonopt.view(-1)
x = np.arange(10000)
y = data_nonopt.detach().numpy()
fig = plt.figure(figsize=(8, 4))
hist, bins = np.histogram(y, bins= 50)
freq = hist / len(y)
plt.bar(bins[:-1], freq, width=  bins[1] - bins[0])
plt.show()



#通过训练这个流模型来学习这个分布
optimizer = torch.optim.Adam(distribution.parameters(), lr=1e-3)
for i in range(1000):
    optimizer.zero_grad()
    loss = -distribution.log_prob(inputs=data).mean()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(i)

# Sample from the learned distribution
samples = distribution.sample(5000)
print("均值：", samples.mean())
print("方差：", samples.var())
samples = samples.view(-1)
x = np.arange(5000)
y = samples.detach().numpy()
#计算频率
hist, bins = np.histogram(y, bins = 100)
freq =  hist / len(y)
fig = plt.figure(figsize=(8,4))
plt.bar(bins[:-1], freq, width=bins[1]-bins[0])
plt.show()

fig2 = plt.figure(figsize=(8, 4))
plt.scatter(y, x, label="Flow_distribution", s=75, c='b', alpha=0.5, marker='o', linewidths=0)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()