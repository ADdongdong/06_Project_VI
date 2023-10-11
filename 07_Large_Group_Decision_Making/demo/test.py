from adaptiveConflictRevising import adaptiveConfilctRevising
from adaptiveConflictRevising import ScoreWeight
import numpy as np


# 假设您的数据包含n_samples个元素，每个元素是一个m x n的二维矩阵，同时有一个n维的一维向量
n_samples = 50  # 用于示例的样本数量
m, n = 3, 4  # 二维矩阵的大小，用于示例，n个标准，m个方案

# 生成示例数据
experts_data = []
for _ in range(n_samples):
    scores = np.random.rand(m, n)  # 随机生成打分矩阵
    weights = np.random.rand(n)  # 随机生成权重向量
    expert = ScoreWeight(scores, weights)  # 定义scoreweight结构体变量
    # expert = ScoreWeight(scores)  # 定义scoreweight结构体变量
    experts_data.append(expert)

obj = adaptiveConfilctRevising(experts_data, 6, 1000, 0.001)
#result1 = obj.fcm()
result = obj.adaptive_conflict_resolution_model()
