import numpy as np
from FCM import fcm
from ScoreWeight import ScoreWeight
from adaptive_conflict_revising import adaptive_conflict_revising
from function import calculate_compatibility_degree


def algorithm_3(ωq, Dq, ψ, T, Q):
    # Step 1: 计算专家之间的兼容度(结果保存在一个二维矩阵中)
    # 先生成一个矩阵，(Q, Q)这里Q是专家的个数

    # 计算专家之间的兼容度
    compatibility_matrix = calculate_compatibility_degree(eq, es)

    t = 0
    Gt = [(k1, k2) for k1 in range(K) for k2 in range(K)]

    while t < T and Gt:
        # Step 2: 专家聚类
        subgroups = fcm()
        # 使用(9)确定最佳子组数量
        # 这里需要根据实际情况实现最佳子组数量的确定

        # Step 3: 计算子组之间的冲突度
        subgroup_compatibility_matrix = np.zeros((K, K))
        for k1 in range(K):
            for k2 in range(K):
                subgroup_compatibility_matrix[k1, k2] = calculate_subgroup_compatibility_degree(
                    subgroups[k1], subgroups[k2])

        max_SCIkl = np.max(subgroup_compatibility_matrix)

        # Step 4: 减小子组之间的冲突
        if max_SCIkl >= ψ:
            t += 1
            continue
        else:
            ωGk, DGk = adaptive_conflict_revising(experts_data, ψ, T)

        # Step 5: 计算子组权重
        subgroup_weights = calculate_subgroup_weights(
            subgroup_compatibility_matrix)

    # Step 6: 输出最终权重向量和决策矩阵
    # 这里需要根据实际公式和数据结构输出结果

    # Step 7: 排序备选方案
    # 这里需要根据实际情况实现备选方案的排序

    return t, ωGk, DGk

# 调用算法3并传入相应参数
# t∗, ωG∗, DG∗ = algorithm_3(ωq, Dq, ψ, T)
