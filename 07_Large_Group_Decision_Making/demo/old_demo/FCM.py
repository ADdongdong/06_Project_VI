import numpy as np
import skfuzzy as fuzz
from ScoreWeight import ScoreWeight

'''
这个文件定义了fcm模糊聚类函数
'''

# 假设您的数据包含n_samples个元素，每个元素是一个m x n的二维矩阵，同时有一个n维的一维向量
n_samples = 20  # 用于示例的样本数量
m, n = 3, 4  # 二维矩阵的大小，用于示例，n个标准，m个方案

# 生成示例数据
experts_data = []
for _ in range(n_samples):
    scores = np.random.randint(1, 8, size=(m, n), dtype=int)  # 随机生成打分矩阵
    weights = np.random.rand(n)  # 随机生成权重向量
    expert = ScoreWeight(scores, weights)  # 定义scoreweight结构体变量
    expert.normalization_matrix()  # 对打分矩阵进行归一化处理
    experts_data.append(expert)


# 定义fcm函数
def fcm(experts_data, n_clusters):
    '''
        参数：
            experts_data 专家打分结构体
            n_cluster 聚类簇的数量
    '''
    # 将打分矩阵展平,只选择打分矩阵进行排序
    flattened_scores = [expert.scores.flatten() for expert in experts_data]
    data_matrix = np.vstack(flattened_scores).T

    # 进行模糊C均值聚类
    # 指定簇的数量，这里已经作为参数给出了
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data_matrix, n_clusters, 2, error=0.005, maxiter=1000)

    # 找到每个专家所属的簇
    cluster_indices = np.argmax(u, axis=0)

    # 对每个簇中的专家数据进行处理
    clustered_experts_data = [[] for _ in range(n_clusters)]

    for i, cluster_index in enumerate(cluster_indices):
        clustered_experts_data[cluster_index].append(experts_data[i])

    # 计算每个簇中的打分矩阵和权重向量的平均值
    cluster_means = []

    for cluster_data in clustered_experts_data:
        cluster_scores = [expert.scores for expert in cluster_data]
        cluster_weights = [expert.weights for expert in cluster_data]

        mean_scores = np.mean(cluster_scores, axis=0)
        mean_weights = np.mean(cluster_weights, axis=0)

        cluster_means.append((mean_scores, mean_weights))

    # cluster_means 中包含每个簇的平均打分矩阵和平均权重向量
    # 同时，返回一个新的数组，这个数组中每个元素都包含了每个簇的scores和标准vector的均值
    clusters = []
    for i, (mean_scores, mean_weights) in enumerate(cluster_means):
        if not np.isnan(mean_scores).any() or not np.isnan(mean_weights).any():
            print(f"Cluster {i + 1} - Mean Scores:")
            print(mean_scores)
            print(f"Cluster {i + 1} - Mean Weights:")
            print(mean_weights)
            cluster = ScoreWeight(mean_scores, mean_weights)
            clusters.append(cluster)

    return clusters


clusters = fcm(experts_data, 3)
print(clusters)
print(len(clusters))
print(clusters[0].scores)
