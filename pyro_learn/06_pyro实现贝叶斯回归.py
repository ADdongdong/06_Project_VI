import pyro
from pyro.nn import PyroSample
from pyro.nn import PyroModule
from torch import nn
import pyro.distributions as dist

class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        #定义线性层，输入维度为in_features, 输出维度为out_features
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight =  PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y = None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean