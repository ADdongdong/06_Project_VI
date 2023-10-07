import numpy as np
from ScoreWeight import ScoreWeight


# 假设您的数据包含n_samples个元素，每个元素是一个m x n的二维矩阵，同时有一个n维的一维向量
n_samples = 0  # 用于示例的样本数量
m, n = 3, 4  # 二维矩阵的大小，用于示例，n个标准，m个方案

# 生成示例数据
experts_data = []
for _ in range(n_samples):
    scores = np.random.rand(m, n)  # 随机生成打分矩阵
    weights = np.random.rand(n)  # 随机生成权重向量
    expert = ScoreWeight(scores, weights)  # 定义scoreweight结构体变量
    experts_data.append(expert)


# 计算专家之间的兼容度,公式4
def calculate_compativility_degree(experts_data):
    '''
    参数:
    1. expert_decision_matrix:
        专家决策矩阵，这个矩阵是根据初始打分矩阵进行归一化后的矩阵
        比如，对于一个决策，一共有3个评判标准，一共有2个方案，
        那么专家矩阵就是一个2*3的矩阵，即专家会对每个方案，
        按照这3个标准去对每个方案打分。
    2. criteria_weight_matrix:
        标准权重打分。还是上面的3个标准，不同专家对不同标准的重要程度认识不同。
        所以，每个专家对应这样一个一维向量，有3个标准这个向量中就有3个元素
        并且，这3个标准的和是1。
    '''
    # 定义矩阵保存两两专家之间的兼容度
    # 这里expert_number就是专家的数量
    expert_number = len(experts_data)

    compatibility_matrix = np.zeros((expert_number, expert_number))
    for i in range(expert_number):
        for j in range(expert_number):
            # 这里m是共有m个备选方案，后面以超参数给出
            # 根据公式4计算
            C_i_j = (0.5 *
                     np.sum(experts_data[i].scores / m - experts_data[j].scores) +
                     0.5*np.sum(experts_data[i].weights - experts_data[j].weights)) / n
            compatibility_matrix[i][j] = C_i_j
    return compatibility_matrix


# 根据公式(10)计算子组之间的兼容度
def calculate_subgroup_compatibility_degree(subgroup_list):
    '''
        参数：
            参数是subgroup_list 表示自组经过求均值的scoreweight类型列表
        返回值：
            返回值就是The subgroup conflict index (SCI) between Gk and Gl
    '''
    # 计算子组的个数
    G_number = len(subgroup_list)
    # 定义一个二维矩阵保存自组之间的兼容度
    subgroup_conflict_index = np.zeros((G_number, G_number))
    for i in range(G_number):
        for j in range(G_number):
            C_i_j = (0.5*np.sum(subgroup_list[i].scores - subgroup_list[j].scores)/m +
                     0.5*np.sum(subgroup_list[i].weights - subgroup_list[j].weights))/n
        subgroup_conflict_index[i][j] = C_i_j

    return subgroup_conflict_index


def calculate_subgroup_weights(SCIkl):
    # 根据公式(14)计算子组权重
    # 这里需要根据实际公式填充具体计算
    pass
    # return subgroup_weights
