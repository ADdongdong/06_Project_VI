# 导入nflows库和numpy库
import nflows as nf
import nflows.distributions
import nflows.transforms
import nflows.flows
import numpy as np

#数据处理
data = np.load('arr_unique_count.npy', allow_pickle=True)
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

normalized_result = np.array(normalized_result)
#print(normalized_result)
final_list = []
#print(normalized_result[:, [i]])
#将这个矩阵，按照列保存
for i in range(8):
    final_list.append(normalized_result[:,i])

#print(final_list)
print(len(final_list))
print(len(final_list[0]))




'''
对未经过聚类和挑选最优矩阵的数据进行正则化流处理
'''

# 定义一个基础分布，比如标准正态分布
base_dist = nf.distributions.StandardNormal(shape=[1])

# 定义一个变换函数，比如实值自动编码器变换（RealNVP）
transform = nf.transforms.CompositeTransform([
    nf.transforms.MaskedAffineAutoregressiveTransform(features=1, hidden_features=10),
    nf.transforms.RandomPermutation(features=1)
])

# 将基础分布和变换函数组合成一个流（Flow）对象
flow = nf.flows.Flow(transform, base_dist)

#导入处理好的数据

for i in normalized_result:
    #将list转换成array
    x = np.array(i, dtype=np.float32)#注意，这里要声明是单精度浮点形，如果写float就是默认双精度浮点和double一样
    #将数据修改成21行1列的数据，方便计算(21,1)
    x_ = x.reshape((len(x), 1))

    #使用流对象的log_prob方法来计算输入数组的对数概率密度
    log_prob = flow.log_prob(x_)
    #print(log_prob)#输出对数概率密度

    # 使用流对象的sample方法来从流中采样新的数组
    samples = flow.sample(8)
    #print(samples)






