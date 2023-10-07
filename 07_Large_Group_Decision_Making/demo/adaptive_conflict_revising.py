import numpy as np
from ScoreWeight import ScoreWeight
from function import *

'''
这个文件是论文中的算法2
函数的作用是减少子组之间的冲突
'''
# 初始化参数
T = 100  # 最大迭代次数
ψ = 0.1  # 冲突阈值
K = 5  # 子组数量, 这里的子组数量就是簇的数量

# 假设您的数据包含n_samples个元素，每个元素是一个m x n的二维矩阵，同时有一个n维的一维向量
m, n = 3, 4  # 二维矩阵的大小，用于示例，n个标准，m个方案

# 生成示例数据
experts_data = []
for _ in range(K):
    scores = np.random.randint(1, 8, size=(m, n), dtype=int)  # 随机生成打分矩阵
    weights = np.random.rand(n)  # 随机生成权重向量
    expert = ScoreWeight(scores, weights)  # 定义scoreweight结构体变量
    expert.normalization_matrix()  # 对打分矩阵进行归一化处理
    experts_data.append(expert)

# 定义修正冲突的函数


def adaptive_conflict_revising(experts_data, ψ, K):
    '''
        参数：
            experts_data: 包含了所有子组的数组，每个自组值一个ScoreWeight结构体，是经过fcm以后得子组
            ψ：冲突阈值
            K: 自组的个数，也就是fcm之后簇的数量

    '''
    Gt = [(i, j) for i in range(K) for j in range(K)]  # 初始子组对
    # 这里t是迭代次数
    t = 0
    # 开始迭代
    while t < T and Gt:
        # 计算子组冲突指数
        sci_matrix = calculate_subgroup_compatibility_degree(experts_data)
        SCIt = np.max(sci_matrix)

        # 检查冲突终止条件
        if np.all(SCIt <= ψ):
            break

        # 选择最不兼容的子组
        y, z = np.unravel_index(np.argmax(sci_matrix), sci_matrix.shape)

        # 冲突解决 15 - 18
        if (y, z) in Gt and (z, y) in Gt:
            # a) 两个子组都接受反馈建议
            alpha_t = 0.5  # 根据算法具体公式确定 alpha_t
            for k in range(K):
                if k == y:
                    experts_data[k].weights = (
                        experts_data[y].weights**alpha_t * experts_data[z].weights**(1-alpha_t))
                    experts_data[k].scores = (
                        experts_data[y].scores**alpha_t * experts_data[z].scores**(1-alpha_t))
                elif k == z:
                    experts_data[k].weights = (
                        experts_data[z].weights**alpha_t * experts_data[y].weights**(1-alpha_t))
                    experts_data[k].scores = (
                        experts_data[z].scores**alpha_t * experts_data[y].scores**(1-alpha_t))

        elif (y, z) in Gt:
            # b) Gy接受，Gz拒绝
            alpha_t = 0.5  # 根据算法具体公式确定 alpha_t
            for k in range(K):
                if k == y:
                    experts_data[k].weights = (
                        experts_data[y].weights**alpha_t * experts_data[z].weights**(1-alpha_t))
                    experts_data[k].scores = (
                        experts_data[y].scores**alpha_t * experts_data[z].scores**(1-alpha_t))

        elif (z, y) in Gt:
            # c) Gz接受，Gy拒绝
            alpha_t = 0.5  # 根据算法具体公式确定 alpha_t
            for k in range(K):
                if k == z:
                    experts_data[k].weights = (
                        experts_data[z].weights**alpha_t * experts_data[y].weights**(1-alpha_t))
                    experts_data[k] = (experts_data[z]**alpha_t *
                                       experts_data[y]**(1-alpha_t))

        # 更新迭代计数和子组集合
        t += 1
        Gt = [(i, j) for i in range(K) for j in range(K) if SCIt[i, j] > ψ]

    return experts_data


experts_data = adaptive_conflict_revising(experts_data, ψ, K)
print(experts_data)

for i, expert_data in enumerate(experts_data):
    print(expert_data.scores)
    print(expert_data.weights)
# 最终结果
# ω ∗ Gk = ωG
# D ∗ Gk = DG
