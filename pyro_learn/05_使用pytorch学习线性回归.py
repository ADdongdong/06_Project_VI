import os
from functools import partial
from torch import nn
import torch
import numpy as np
import pandas as pd
from pyro.nn import PyroModule
import matplotlib.pyplot as plt

'''
例子为地形崎岖指数衡量国家的地形异质性与人均GDP的关系
rugged：量化地形崎岖指数
cont_africa：给定的国家是否在非洲
rgdppc_2000： 2000年实际人均国内生产总值
'''
#读取数据
DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

#pytorch实现线性回归
'''
PyroModule是pyro库中的一个子类，它是nn.Module的一个混合子类，
PyroModule有三种不同的创建方式：
1.创建一个新的子类
2.使用PyroModule修饰现在有的nn.Module子类
3.使用PyroModule.from_module()方法从现有的nn.Module实例中创建一个
  PyroModule实例
  
PyroModule[nn.Linear]是一个由PyroModule创建的nn.Linear子类
'''
assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

'''
loss:均方误差
optimizer:Adam
'''
df["cont_africa_x_rugged"] = df["cont_africa"] * df["rugged"]
data = torch.tensor(df[["cont_africa", "rugged", "cont_africa_x_rugged", "rgdppc_2000"]].values,
                        dtype=torch.float)
x_data, y_data = data[:, :-1], data[:, -1]

#定义回归模型
#创建一个线性层，这个线性层有三个输入，一个输出[3,1]
#y = wx + b
linear_reg_model = PyroModule[nn.Linear](3,1)

#定义loss和optimize
loss_fn = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.05)
num_iterations = 1500

losses = []
def train():
    #运行线性模型
    y_pred = linear_reg_model(x_data).squeeze(-1)
    #计算均方误差
    loss = loss_fn(y_pred, y_data)
    losses.append(loss)
    #初始化梯度信息
    optim.zero_grad()
    #梯队反向传播
    loss.backward()
    #进行下一步运算
    optim.step()
    return loss

#运行
for j in range(num_iterations):
    loss = train()
    if (j + 1) % 50 == 0:
        print("[iteration %04d] loss :%.4f" % (j + 1, loss.item()))

#检查最终训练好的参数
print("learned parameters:")
for name, param in linear_reg_model.named_parameters():
    print(name, param.data.numpy())

#绘制回归拟合
fit = df.copy()
#用训练好的模型，输入x_data计算y_pred,并保存在fit["mean"]这个数组中
fit["mean"] = linear_reg_model(x_data).detach().cpu().numpy()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = fit[fit["cont_africa"]==1]
non_african_nations = fit[fit["cont_africa"]==0]
fig.suptitle("Regression Fit", fontsize = 16)
ax[0].plot(non_african_nations["rugged"], non_african_nations["rgdppc_2000"], "o")
ax[0].plot(non_african_nations["rugged"], non_african_nations["mean"], linewidth=2)
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")
ax[1].plot(african_nations["rugged"], african_nations["rgdppc_2000"], "o")
ax[1].plot(african_nations["rugged"], african_nations["mean"], linewidth=2)
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations")

plt.legend()
plt.show()



