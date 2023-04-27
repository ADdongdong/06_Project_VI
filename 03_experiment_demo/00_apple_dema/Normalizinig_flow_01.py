# 导入nflows库和numpy库
import torch
import nflows as nf
import nflows.distributions as distributions
import nflows.transforms as transforms
import nflows.flows as flows
import numpy as np


#数据处理
data = np.load('./00_data/normalization_list.npy', allow_pickle=True)

#定义模型 对经过聚类和挑选最优矩阵的数据进行正则化流处理
def define_model():
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

    # 定义一个流模型
    distribution = flows.Flow(transform, base_distribution)
    return distribution



def train_model(distribution):
    distribution_list = []
    optimizer = torch.optim.Adam(distribution.parameters(), lr=1e-3)
    for j in range(8):
        #每次取出一个特征的数据进行学习
        data_ = torch.tensor(data=[j])
        print(data_.shape)
        data_ = data_.reshape((len(data_), 1)).float()
        print(data_.shape)
        print(data_.dtype)
        for i in range(1000):
            optimizer.zero_grad()
            loss = -distribution.log_prob(inputs=data_).mean()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(i)
        #将每次训练的结果保存下来
        model_name = "./normalizingFlowModel/OptimA" + str(j+1) +".pt"
        torch.save(distribution.state_dict(), model_name)
        #从Ai对应的模型中进行采样1000个数据并加入到sample_list文件中
        sapmle_ = distribution.sample(1000)
        sample_data.append(sapmle_.detach().numpy())


sample_data = []
#通过训练这个流模型来学习这个分布
distribution = define_model()
train_model(distribution)
#将采样出来的数据保存起来``
sample_data = np.array(sample_data)
np.save('./00_data/sample_data.npy', sample_data)

