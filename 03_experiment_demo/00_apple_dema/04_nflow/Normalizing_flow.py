# 导入nflows库和numpy库
import torch
import nflows as nf
import nflows.distributions as distributions
import nflows.transforms as transforms
import nflows.flows as flows
import numpy as np
import os
from matrix_trans import normalizing_data

print(os.getcwd())
# 数据处理
data = np.load('./data/resutl_matrix.npy', allow_pickle=True)
final_list = normalizing_data(data)

# 对未经过聚类和挑选最优矩阵的数据进行正则化流处理
# Define the base distribution
base_distribution = distributions.StandardNormal((1,))

# Define the flow
num_transforms = 4
transforms_ = []
for _ in range(num_transforms):
    transforms_.append(transforms.MaskedAffineAutoregressiveTransform(
        features=1,
        hidden_features=2,
        context_features=None,
        # activation='ReLU',
        # activation: 激活函数，可以是任何PyTorch中支持的函数，默认是torch.nn.ReLU。
    ))
transform = transforms.CompositeTransform(transforms_)

# Define the distribution to learn
# 定义一个流模型
distribution = flows.Flow(transform, base_distribution)

# 通过训练这个流模型来学习这个分布
distribution_list = []
optimizer = torch.optim.Adam(distribution.parameters(), lr=1e-3)
for j in range(8):
    # 每次取出一个特征的数据进行学习
    data_ = torch.tensor(final_list[j])
    # print(data_.shape)
    data_ = data_.reshape((len(data_), 1)).float()
    # print(data_.shape)
    # print(data_.dtype)
    for i in range(1000):
        optimizer.zero_grad()
        loss = -distribution.log_prob(inputs=data_).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(i)
    # 将每次训练的结果保存下来
    model_name = "./normalizingFlowModel/A" + str(j+1) + ".pt"
    torch.save(distribution.state_dict(), model_name)
