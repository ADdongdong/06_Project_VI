# 导入nflows库和numpy库
import torch
import nflows as nf
import nflows.distributions as distributions
import nflows.transforms as transforms
import nflows.flows as flows
import numpy as np
from tqdm import tqdm
from matrix_trans import normalizing_data


# 数据处理
# data = np.load('./00_data/normalization_list.npy', allow_pickle=True)
# 数据处理
data_ = np.load('./data/resutl_matrix.npy', allow_pickle=True)
final_list = normalizing_data(data_)

# 定义模型 对经过聚类和挑选最优矩阵的数据进行正则化流处理


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


# 定义训练模型的函数
def train_model(distribution, data):
    sample_data = []
    optimizer = torch.optim.Adam(distribution.parameters(), lr=1e-4)
    for j in range(8):
        # 每次取出一个特征的数据进行学习
        # 将array数据转化为pytorch.tensor数据,这样才可以进行随机梯度下降
        data_ = torch.tensor(data[j])
        data_ = data_.reshape((len(data_), 1)).float()  # (1, len) -> (len, 1)
        #pbar = tqdm(range(500))
        for i in range(500):
            optimizer.zero_grad()
            loss = -distribution.log_prob(inputs=data_).mean()
            loss.backward()
            optimizer.step()

            #description = f"Loss={loss:.2f}"
            # pbar.set_description(description)
            if i % 10 == 0:
                print(str(i) + " Loss:", loss.item())

        # 将每次训练的结果保存下来
        model_name = "./normalizingFlowModel/OptimA" + str(j+1) + ".pt"
        torch.save(distribution.state_dict(), model_name)
        # 从Ai对应的模型中进行采样1000个数据并加入到sample_list文件中
        sapmle_ = distribution.sample(10000)
        sample_data.append(sapmle_.detach().numpy())

    return sample_data


# 定义计算均值和方差的函数
def mena_var(sample_data):
    result = []
    for i in sample_data:
        mean = np.mean(i)
        var = np.var(i)
        result.append([mean, var])
    return result


# 定义数组，保存Ai采样出来的数据
# 通过训练这个流模型来学习这个分布
distribution = define_model()
sample_data = train_model(distribution, final_list)
# 计算均值和方差
mean_var_list = mena_var(sample_data)
print(mean_var_list)

# 将采样出来的数据保存起来
sample_data = np.array(sample_data)
# 将计算的均值和方差也保存起来
mean_var_list = np.array(mean_var_list)
np.save("./data/mean_var_list.npy", mean_var_list)
np.save('./data/sample_data.npy', sample_data)
