# 导入nflows库和numpy库
import torch
import nflows as nf
import nflows.distributions as distributions
import nflows.transforms as transforms
import nflows.flows as flows
import numpy as np
import os

print(os.getcwd())
#数据处理
data = np.load('./00_data/arr_unique_count.npy', allow_pickle=True)
#print(data[0])
#将data中的所有一维数据都转换成8乘8的二维数据

List = []
for i in data:
    matrix = i.reshape((8,8))
    List.append(matrix)
#print(List[0])

#对于每一个矩阵，计算其集几何平均值：将8个元素连续乘，然后对结果开8次方
result = []
for matrix in List:
    x = []#这个数组用来记录每一个矩阵的8个几何平均值
    for i in matrix:#一个matrix里面有8行，就会计算出来8个元素
        num = 1
        for j in range(8):
            num = i[j]*num
        #对num开n次方
        num = num**(1/8)
        x.append(num)
    result.append(x)


#对每一行元素进行归一化处理
normalized_result = []
for i in result:
    i = np.array(i)
    normalized_v = i / np.sqrt(np.sum(i**2))
    normalized_result.append(normalized_v)



#将数据按照列进行规划排列
normalized_result = np.array(normalized_result)
final_list = []
#将这个矩阵，按照列保存
for i in range(8):
    final_list.append(normalized_result[:,i])

#print(final_list)
'''
对未经过聚类和挑选最优矩阵的数据进行正则化流处理
'''
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

#通过训练这个流模型来学习这个分布
distribution_list = []
optimizer = torch.optim.Adam(distribution.parameters(), lr=1e-3)
for j in range(8):
    #每次取出一个特征的数据进行学习
    data_ = torch.tensor(final_list[j])
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
    model_name = "./normalizingFlowModel/A" + str(j+1) +".pt"
    torch.save(distribution.state_dict(), model_name)

