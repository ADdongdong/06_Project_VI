import numpy as np

# 专家输入的数据是这样的：
# 第一部分：一个打分矩阵，这个矩阵是对m个方案的n个标准进行的打分
# 第二部分：一个vector,对于n个标准，每个维度专家会给出一个这个标准对应的权重，n个标准权重组成的vector
# 这里定义一个结构体来保存每个专家对应打分矩阵和标准vector


class ScoreWeight:
    def __init__(self, score, weights) -> None:
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
