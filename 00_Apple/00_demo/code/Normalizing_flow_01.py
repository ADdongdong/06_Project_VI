# 导入nflows库和numpy库
import nflows as nf
import nflows.distributions
import nflows.transforms
import nflows.flows
import numpy as np

'''
MaskedAffineAutoregressiveTransform()函数的参数：

features: 输入变量的维数，必须是一个正整数。
hidden_features: 隐藏层神经网络的特征数，必须是一个正整数。
num_blocks: 隐藏层神经网络的块数，必须是一个正整数。
use_residual_blocks: 是否使用残差连接，必须是一个布尔值。
random_mask: 是否使用随机掩码，必须是一个布尔值。
activation: 激活函数，可以是任何PyTorch中支持的函数，默认是torch.nn.ReLU。
dropout_probability: 隐藏层神经网络中使用dropout的概率，必须是一个在0到1之间的浮点数，默认是0。
use_batch_norm: 是否使用批量归一化，必须是一个布尔值，默认是False。
'''

# 定义一个基础分布，标准正态分布
base_dist = nf.distributions.StandardNormal(shape=[1])

# 定义一个变换函数，实值自动编码器变换（RealNVP）
# compositeTransform函数，用来构造一个可逆变换，这个可逆变换可以是一个复合的变换，可以由多个可逆变换组合在一起
transform = nf.transforms.CompositeTransform([
    nf.transforms.MaskedAffineAutoregressiveTransform(features=1, hidden_features=10),
    nf.transforms.RandomPermutation(features=1)
])

# 将基础分布和变换函数组合成一个流（Flow）对象
flow = nf.flows.Flow(transform, base_dist)

#导入处理好的数据
data = np.load('normalization_list.npy', allow_pickle=True)

for i in data:
    #将list转换成array
    x = np.array(i, dtype=np.float32)#注意，这里要声明是单精度浮点形，如果写float就是默认双精度浮点和double一样
    #将数据修改成21行1列的数据，方便计算(21,1)
    x_ = x.reshape((len(x), 1))

    #使用流对象的log_prob方法来计算输入数组的对数概率密度
    log_prob = flow.log_prob(x_).mean()
    print(log_prob)#输出对数概率密度

    # 使用流对象的sample方法来从流中采样新的数组
    samples = flow.sample(8)
    print(samples)
    #print(flow.log_prob(x_).mean())

