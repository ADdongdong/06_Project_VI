import os
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

'''
render_model()函数
主要是用来可视化概率图的
'''

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.4')

#定义一个例子
def model(data):
    m = pyro.sample("m", dist.Normal(0, 1))
    sd = pyro.sample("sd", dist.LogNormal(m, 1))
    with pyro.plate("N", len(data)):
        pyro.sample("obs", dist.Normal(m, sd), obs=data)

data = torch.ones(10)
pyro.render_model(model, filename="img/sample_1.jpg", model_args=(data,))