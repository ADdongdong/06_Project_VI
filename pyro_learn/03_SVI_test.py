import pyro
import pyro.distributions as dist
import torch
import pyro.distributions.constraints as constraints
from pyro.optim import Adam
import math



#创建一些观测数据
data = []
for _ in range(6):
    data.append(torch.tensor(1.0))
for _ in range(4):
    data.append(torch.tensor(0.0))

#定义模型，里面有观测变量和隐变量
def model(data):
    #定义控制先验的超参数
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    #定义随机变量f, f服从先验Bata分布,这里f没有设置obs参数，laten_fairness是隐变量
    f = pyro.sample("laten_fairness", dist.Beta(alpha0, beta0))
    #生成一些样本，从f中
    for i in range(len(data)):
        # 观测数据从伯努利分布中生成，观测数据为指定的data[i]
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f) ,obs=data[i])

def guide(data):
    # register the two variational parameters with Pyro.
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
                         constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0),
                        constraint=constraints.positive)
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    # 随机变量latent_fairness服从beta分布
    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))


pyro.render_model(model = model, filename="img/SVI_test.png", model_args=(data,),render_params=True)

#定义优化器
adam_params = {"lr":0.0005, "betas":(0.90, 0.999)}
optimizer = pyro.optim.Adam(adam_params)

#设置随机变分推断算法
svi = pyro.infer.SVI(model, guide, optimizer, loss = pyro.infer.Trace_ELBO())

for step in range(2000):
    svi.step(data)
    if step %100 ==0:
        print('-', end='')

# grab the learned variational parameters
# 获得学习到的变分参数
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()

# here we use some facts about the Beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * math.sqrt(factor)

print("\nBased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))
