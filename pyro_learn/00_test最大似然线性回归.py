import logging
import os

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyro.optim
import pyro.infer
import pyro
import pyro.infer.autoguide


smoke_test = ('CI' in os.environ)
#检查pyro的版本是不是1.8.4,如果不是，则报错
assert pyro.__version__.startswith('1.8.4')

#检查
pyro.enable_validation(True)
pyro.set_rng_seed(1)
logging.basicConfig(format='%(message)s', level=logging.INFO)

# Set matplotlib settings
#%matplotlib inline
plt.style.use('default')

data = pd.read_csv("rugged_data.csv", encoding="ISO-8859-1")

#cont_africa 特定国家是否在非洲
#rugged 量化地形崎岖程度指数
#rgdppc_2000 2000年实际人均国内生产总值
df = data[["cont_africa", "rugged", "rgdppc_2000"]]

#isfinite函数用来判断数组中元素是否为有限数组
df = df[np.isfinite(df.rgdppc_2000)]
#log 计算rgdppc_2000这一列中每个元素的自然对数
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

#将dp中所有元素提取出来，并转换成tensor，且数据类型为float
train = torch.tensor(df.values, dtype=torch.float)
is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]

'''
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
#筛选非洲国家和非非洲国家，使用mask
african_nations = df[df["cont_africa"] == 1]
non_african_nations = df[df["cont_africa"] == 0]
sns.scatterplot(x=non_african_nations["rugged"],
                y=non_african_nations["rgdppc_2000"],
                ax=ax[0])
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")
sns.scatterplot(x=african_nations["rugged"],
                y=african_nations["rgdppc_2000"],
                ax=ax[1])
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations")
plt.show()
'''

#mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness

import pyro.distributions as dist
import pyro.distributions.constraints as constraints

def simple_model_1(is_cont_africa, ruggedness, log_gdp=None):

    a = pyro.param("a", lambda: torch.randn(()))#偏移量
    b_a = pyro.param("bA", lambda: torch.randn(()))
    b_r = pyro.param("bR", lambda: torch.randn(()))
    b_ar = pyro.param("bAR", lambda: torch.randn(()))
    sigma = pyro.param("sigma", lambda: torch.ones(()), constraint=constraints.positive)
    '''
    pyro.param()函数：定义模型的参数
        1.参数1 参数的名称，数据类型为字符串
        2.参数2 参数的初始值
            2.1 这里代码中用来lambda来定义了一个匿名函数，这个匿名函数的函数体就是torch.randn(),
                用torch.randn()来生成参数
        3.constraint参数 是用来约束参数的，比如取值为constraints.positive这个参数取值必须为正数
    '''
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness


    '''
    pyro.plate()函数：声明某些批次的维度是独立的
        1.参数name:字符串，表示该维度的名称
        2.size：整数或者元组，表示该维度的大小
    '''
    with pyro.plate("data", len(ruggedness)):
        '''
               pyro.sample()函数：返回定义好的一个随机变量：
                  1.随机变量名称：obs
                  2.随机变量服从的分布：均值为mean,方差为sigma的标准正态分布
                  3.obs参数：可选参数，用来指定观测值。
                       3.1 如果指定的观测值，该函数会返回一个确定的值
                       3.2 如果没有指定观测值，那么就会返回一个这个随机变量服从分布采样到的值
                       3.3 如果指定了obs参数，该函数将被视为观测结点，也就是已知数据的一部分，
                           其分布不会随着模型的推断过程所改变
                  4.infer参数：指定推断算法的，可以指定pyro.infer.SVI，pyro.infer.Trace_ELBO，pyro.infer.MCMC等
               '''
        # 这个sample函数指定了obs所，会一直返回log_gdp,log_gdp的默认值是None
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)

def simple_model_2(is_cont_africa, ruggedness, log_gdp=None):
    a = pyro.sample("a", dist.Normal(0., 10.))
    b_a = pyro.sample("bA", dist.Normal(0., 1.))
    b_r = pyro.sample("bR", dist.Normal(0., 1.))
    b_ar = pyro.sample("bAR", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))

    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness

    with pyro.plate("data", len(ruggedness)):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)

'''
pyro.render_model()函数：用于将概率图模型可视化，可以将概率图转换为各种图片模式png、jpg等保存
    1 参数model 要渲染的模型(model这个关键字可以不写，因为默认第一个参数就是要渲染的模型，传入的是函数名）
    2 参数filename 这个概率图模型图片保存在哪里
    3 参数model_args:传递给模型的参数(也就是模型函数的参数)
    4 model_kwargs 传递给模型关键字的参数
    5 render_param 是否显示传递给模型的参数
    6 render_distributions 是否显示随机变量的信息，服从的分布
    7 rankdir 图形方向
'''
pyro.render_model(model=simple_model_2, filename= "img/simple_model_2.png", model_args=(is_cont_africa, ruggedness, log_gdp),  render_params=True, render_distributions=True)

'''
#定义变分分布，且，q(z)基于平均场
def custom_guide(is_cont_africa, ruggedness, log_gdp=None):
    #a_loc,a_scale,sigma_loc,weights_loc,weights_scale指定了模型的先验分布即q(z)
    #且，随机变量都被划分成了M个组，且组与组之间相互独立（平均场理论）
    a_loc = pyro.param('a_loc', lambda: torch.tensor(0.))
    a_scale = pyro.param('a_scale', lambda: torch.tensor(1.),
                         constraint=constraints.positive)
    sigma_loc = pyro.param('sigma_loc', lambda: torch.tensor(1.),
                           constraint=constraints.positive)
    weights_loc = pyro.param('weights_loc', lambda: torch.randn(3))
    weights_scale = pyro.param('weights_scale', lambda: torch.ones(3),
                               constraint=constraints.positive)
    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
    b_a = pyro.sample("bA", dist.Normal(weights_loc[0], weights_scale[0]))
    b_r = pyro.sample("bR", dist.Normal(weights_loc[1], weights_scale[1]))
    b_ar = pyro.sample("bAR", dist.Normal(weights_loc[2], weights_scale[2]))
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
    return {"a": a, "b_a": b_a, "bR": b_r, "bAR": b_ar, "sigma": sigma}
'''

#绘制隐变量的概率图
#pyro.render_model(custom_guide, filename="img/guide_model.png", model_args=(is_cont_africa,ruggedness, log_gdp), render_params=True)

pyro.clear_param_store()

auto_guide = pyro.infer.autoguide.AutoNormal(model=simple_model_2)
#定义优化算法,learning_rate定义为0.02
adam = pyro.optim.Adam({"lr":0.02})
#定义ELBO
elbo = pyro.infer.Trace_ELBO()
#指定模型为随机梯度变分推断SVI
svi = pyro.infer.SVI(simple_model_2, auto_guide, adam, elbo)

#定义loss函数数组，用来绘制图像
losses = []
#losses_ = []
for step in range(1000):
    loss = svi.step(is_cont_africa, ruggedness, log_gdp)
    #loss_ = svi.evaluate_loss(is_cont_africa, ruggedness, log_gdp)
    #losses_.append(loss_)
    losses.append(-loss)
    if step%100 == 0:
        logging.info("ELBO loss: {}".format(loss))

plt.figure(figsize=(8,5))
plt.plot(losses)
#plt.plot(losses_)
plt.xlabel("SVI step")
plt.ylabel("ELBO loss")
#plt.show()

#查看经过优化过后的模型参数
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())

#推断
#custom_guide()函数，给出在观察变量is_cont_africa,ruggedness下
#未观察变量log_gdp可能的取值，并且，生成800个这样的样本
#custom_guide定义时，最后一个参数是log_gdp,所以就是在dim=-1的变量上采样
#这样就算出了，给定is_cont_africa和ruggedness时候，log_gdp的值
with pyro.plate("samples", 800, dim=-1):
    samples = auto_guide(is_cont_africa, ruggedness)

#查看is_cont_africa和ruggedness的权重影响
gamma_within_africa = samples["bR"] + samples["bAR"]
gamma_outside_africa = samples["bR"]

fig = plt.figure(figsize=(10, 6))
sns.histplot(gamma_within_africa.detach().cpu().numpy(), kde=True, stat="density", label="African nations")
sns.histplot(gamma_outside_africa.detach().cpu().numpy(), kde=True, stat="density", label="Non-African nations", color="orange")
fig.suptitle("Density of Slope : log(GDP) vs. Terrain Ruggedness")
plt.xlabel("Slope of regression line")
plt.legend()
plt.show()

#模型的评估
#使用Predictive类，从后验预测分布中生成和可视化800个样本
predictive = pyro.infer.Predictive(simple_model_2, guide = auto_guide, num_samples=800)
svi_sample = predictive(is_cont_africa, ruggedness, log_gdp=None)
svi_gdp = svi_sample["obs"]

#可视化
predictions = pd.DataFrame({
    "cont_africa": is_cont_africa,
    "rugged": ruggedness,
    "y_mean": svi_gdp.mean(0).detach().cpu().numpy(),
    "y_perc_5": svi_gdp.kthvalue(int(len(svi_gdp) * 0.05), dim=0)[0].detach().cpu().numpy(),
    "y_perc_95": svi_gdp.kthvalue(int(len(svi_gdp) * 0.95), dim=0)[0].detach().cpu().numpy(),
    "true_gdp": log_gdp,
})
african_nations = predictions[predictions["cont_africa"] == 1].sort_values(by=["rugged"])
non_african_nations = predictions[predictions["cont_africa"] == 0].sort_values(by=["rugged"])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
fig.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)

ax[0].plot(non_african_nations["rugged"], non_african_nations["y_mean"])
ax[0].fill_between(non_african_nations["rugged"], non_african_nations["y_perc_5"], non_african_nations["y_perc_95"], alpha=0.5)
ax[0].plot(non_african_nations["rugged"], non_african_nations["true_gdp"], "o")
ax[0].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="Non African Nations")

ax[1].plot(african_nations["rugged"], african_nations["y_mean"])
ax[1].fill_between(african_nations["rugged"], african_nations["y_perc_5"], african_nations["y_perc_95"], alpha=0.5)
ax[1].plot(african_nations["rugged"], african_nations["true_gdp"], "o")
ax[1].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="African Nations")
plt.show()