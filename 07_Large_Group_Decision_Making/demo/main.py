import numpy as np
from FCM import fcm
from ScoreWeight import ScoreWeight
from adaptive_conflict_revising import adaptive_conflict_revising
from function import *


def algorithm_3(experts_data, n_clusters, ψ, T, Q):
    '''
        参数：
            experts_data:
                这里experts_data是输入的数据列表
                每个数据是一个结构体，包括scores矩阵和weight向量

            n_clusters:这里是fcm聚类的簇的数量

    '''

    # Step 1: 计算专家之间的兼容度(结果保存在一个二维矩阵中)
    # 先生成一个矩阵，(Q, Q)这里Q是专家的个数

    # 计算专家之间的兼容度,这里不用计算兼容度，因为，这个是对模糊积累的修改，这里用不上
    # compatibility_matrix = calculate_compatibility_degree(eq, es)

    # step1:对专家进行聚类
    subgroups = fcm(experts_data, n_clusters)

    # step2: 通过公式10计算自组之间的兼容度
    //
    sci_matrix = calculate_subgroup_compatibility_degree(subgroups)
    max_SCIkl = np.max(sci_matrix)

    # Step 4: 减小子组之间的冲突
    finally_matrix = adaptive_conflict_revising(subgroups, ψ, T)

    # Step 6: 输出最终权重向量和决策矩阵
    # 这里需要根据实际公式和数据结构输出结果

    # Step 7: 排序备选方案
    # 这里需要根据实际情况实现备选方案的排序

    # return t, ωGk, DGk

# 调用算法3并传入相应参数
# t∗, ωG∗, DG∗ = algorithm_3(ωq, Dq, ψ, T)
