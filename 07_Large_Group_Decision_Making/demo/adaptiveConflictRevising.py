import numpy as np
import skfuzzy as fuzz


# 专家输入的数据是这样的：
# 第一部分：一个打分矩阵，这个矩阵是对m个方案的n个标准进行的打分
# 第二部分：一个vector,对于n个标准，每个维度专家会给出一个这个标准对应的权重，n个标准权重组成的vector
# 这里定义一个结构体来保存每个专家对应打分矩阵和标准vector
class ScoreWeight:
    def __init__(self, score, weights=[]) -> None:
        # 这里如果没有weight权重打分的话，则将weight设置为[]
        self.scores = score  # 打分矩阵, 直接初始化为normalization后的矩阵
        self.weights = weights
        self.benefit_criterion_matrix = score  # 定义benefit criterion matrix
        self.cost_criterion_matrix = score  # 定义cost_criterion_matrix

    def normalization_matrix(self):
        # 找到每一行的最小值和最大值
        min_values = np.min(self.scores, axis=1, keepdims=True)  # 沿着第三个维度找最小值
        max_values = np.max(self.scores, axis=1, keepdims=True)  # 沿着第三个维度找最大值

        # 对每一行进行操作,获取到利益标准矩阵
        benefit_criterion = (self.scores - min_values) / \
            (max_values - min_values)

        # 计算获取到cost_criterion矩阵, 这里不返回cost_criterion矩阵
        cost_criterion = (max_values - self.scores) / \
            (max_values - min_values)

        # 将打分矩阵修改为normalization的矩阵
        self.scores = benefit_criterion
        self.benefit_criterion_matrix = benefit_criterion
        self.cost_criterion_matrix = cost_criterion


# 定义adaptiveConfilctRevising类，在运行的时候，将数据整理好
# 通过这个类中的成员函数可以完成所有的工作
class adaptiveConfilctRevising:
    def __init__(self, expert_data: list, cluster_num: int, T: int, ψ: float = 0.1) -> None:
        '''
            构造函数参数：
            expert_data:所有专家打分数据组成的列表，列表中每个元素都是ScoreWeight 
            cluster_num:FCM模糊C聚类期期望得到聚类的簇数
            T: 是在处理冲突过程中的迭代次数
            ψ：处理冲突时候的冲突阈值，一般设置为0.1

        '''
        self.expert_data = expert_data
        self.cluster_num = cluster_num
        self.T = T
        self.ψ = ψ

    # 根据公式(10)计算子组之间的兼容度

    def calculate_subgroup_compatibility_degree(self, subgroup_data):
        '''
            参数：
                参数是self.expert_data 表示自组经过求均值的scoreweight类型列表
            返回值：
                返回值就是The subgroup conflict index (SCI) between Gk and Gl
        '''
        # 计算子组的个数
        G_number = len(subgroup_data)
        # 获取标准数（行数）m和备选决策项数（列数）n
        m, n = subgroup_data[0].scores.shape
        print(f"(m, n)= ({m}, {n})")
        # 定义一个二维矩阵保存自组之间的兼容度
        subgroup_conflict_index = np.zeros((G_number, G_number))
        for i in range(G_number):
            for j in range(G_number):
                # C_i_j = (0.5*np.sum(subgroup_data[i].scores - subgroup_data[j].scores)/m +
                #          0.5*np.sum(subgroup_data[i].weights - subgroup_data[j].weights))/n
                if subgroup_data[i].weights == []:
                    C_i_j = 0.5 * \
                        np.sum(
                            np.abs(subgroup_data[i].scores - subgroup_data[j].scores))/m
                else:
                    C_i_j = (0.5*np.sum(np.abs(subgroup_data[i].scores - subgroup_data[j].scores))/m +
                             0.5*np.sum(np.abs(subgroup_data[i].weights - subgroup_data[j].weights)))/n
            subgroup_conflict_index[i][j] = C_i_j

        return subgroup_conflict_index

    # 定义解决自组冲突的函数，这个函数运行完，我们认为已经通过算法解决冲突问题
    def adaptive_conflict_revising(self, experts_data):
        '''
            参数：
                experts_data: 包含了所有子组的数组，每个自组值一个ScoreWeight结构体，是经过fcm以后得子组
                ψ：冲突阈值
                K: 自组的个数，也就是fcm之后簇的数量

        '''
        # 之前打印一下子组的数据
        # for i, scoreWeight in enumerate(experts_data):
        #     print(f"group{i + 1}:")
        #     print(scoreWeight.scores)

        Gt = [(i, j) for i in range(len(experts_data))
              for j in range(len(experts_data))]  # 初始子组对
        # print(Gt)
        # 这里t是迭代次数
        t = 0
        # 开始迭代
        while t < self.T and Gt:
            # 计算子组冲突指数
            sci_matrix = np.array(
                self.calculate_subgroup_compatibility_degree(experts_data))
            # print(f"sci_matrix:")
            # print(sci_matrix)
            # 找到冲突最大的那个值，那个值的二维坐标就是那一对冲突最大的自组
            initial_value = -np.inf
            SCIt = np.max(sci_matrix, initial=initial_value)

            # 检查冲突终止条件
            if np.all(SCIt <= self.ψ):
                break

            # 选择最不兼容的子组
            y, z = np.unravel_index(np.argmax(sci_matrix), sci_matrix.shape)

            # 冲突解决 15 - 18
            if (y, z) in Gt and (z, y) in Gt:
                # print("开始解决冲突")
                # a) 两个子组都接受反馈建议
                alpha_t = 0.5  # 根据算法具体公式确定 alpha_t
                for k in range(len(experts_data)):
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
                for k in range(len(experts_data)):
                    if k == y:
                        experts_data[k].weights = (
                            experts_data[y].weights**alpha_t * experts_data[z].weights**(1-alpha_t))
                        experts_data[k].scores = (
                            experts_data[y].scores**alpha_t * experts_data[z].scores**(1-alpha_t))

            elif (z, y) in Gt:
                # c) Gz接受，Gy拒绝
                alpha_t = 0.5  # 根据算法具体公式确定 alpha_t
                for k in range(len(experts_data)):
                    if k == z:
                        experts_data[k].weights = (
                            experts_data[z].weights**alpha_t * experts_data[y].weights**(1-alpha_t))
                        experts_data[k] = (experts_data[z]**alpha_t *
                                           experts_data[y]**(1-alpha_t))

            # 更新迭代计数和子组集合
            t += 1
            Gt = [(i, j) for i in range(len(experts_data))
                  for j in range(len(experts_data)) if sci_matrix[i, j] > self.ψ]
            print(f"剩余不兼容自组对:{Gt}")
        return experts_data

    # 定义fcm函数
    def fcm(self):
        '''
            对成员函数self.expert_data进行聚类
            参数：
                n_cluster 聚类簇的数量
        '''
        # 将打分矩阵展平,只选择打分矩阵进行排序
        flattened_scores = [expert.scores.flatten()
                            for expert in self.expert_data]
        data_matrix = np.vstack(flattened_scores).T

        # 进行模糊C均值聚类
        # 指定簇的数量，这里已经作为参数给出了
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data_matrix, self.cluster_num, 2, error=0.005, maxiter=1000)

        # 找到每个专家所属的簇
        cluster_indices = np.argmax(u, axis=0)

        # 对每个簇中的专家数据进行处理
        clustered_experts_data = [[] for _ in range(self.cluster_num)]

        for i, cluster_index in enumerate(cluster_indices):
            clustered_experts_data[cluster_index].append(self.expert_data[i])

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
                # print(f"Cluster {i + 1} - Mean Scores:")
                # print(mean_scores)
                # print(f"Cluster {i + 1} - Mean Weights:")
                # print(mean_weights)
                cluster = ScoreWeight(mean_scores, mean_weights)
                clusters.append(cluster)
        return clusters

    def adaptive_conflict_resolution_model(self):
        # step1:对专家进行聚类
        subgroups = self.fcm()
        # print(len(subgroups))
        for i, scoreWeight in enumerate(subgroups):
            print(f"group{i + 1}:")
            print(scoreWeight.scores)

        # Step 2: 减小子组之间的冲突
        finally_matrix = self.adaptive_conflict_revising(
            subgroups)

        print(len(finally_matrix))
        for i, scoreWeight in enumerate(finally_matrix):
            print(f"group{i + 1}:")
            print(scoreWeight.scores)
