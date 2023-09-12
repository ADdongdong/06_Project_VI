'''
这个文件，定义函数，将打分矩阵进行归一化计算
'''

import numpy as np
import pandas as pd
from openpyxl import Workbook

# 将数据的处理定义成一个类
# 这个类接收到的数据是一个numpy.array
# 这个array中，有多个8*8的二维array
# 所以，初始化对象的时候，可以传入一个(n, 8, 8)shape的数据


class matrix_data():
    def __init__(self, data) -> None:
        self.data = data

    # private:输入一个8*8的array,对这个8*8的array进行处理
    def __data_trans(self, data):
        # 对矩阵的每一个数字进行如下计算：
        # i = power(sqre(2), i-2)
        for j in range(8):
            for k in range(8):
                data[j][k] = np.power(np.sqrt(2), data[j][k] - 2)
        # 将矩阵的中间元素转化1
        np.fill_diagonal(data, 1.0)
        # print(data)

        # 补齐矩阵的上三角
        for i in range(8):
            for j in range(i + 1, 8):
                data[i][j] = 1/data[j][i]
        return data

    # 定义函数，处理一个数组的数据
    # 同时会将处理好的数据保存为.npy文件，方便下次使用
    def get_result_matrix_npy(self, filename: str) -> np.array:
        result = []
        # print(list[0])
        # 因为list中有元素为0，所以，这里给所有的元素+1
        for i in self.data:
            i = i.astype(float)
            i += 1.0  # 个矩阵的所有元素+1 打分的范围从1-8
            data = self.__data_trans(i)
            result.append(data)
            # break
        result = np.array(result)
        np.save(filename, result)
        return result

    # 将多个8*8的array保存在xlsx表格中
    def save_excel(self, data,  file_name: str) -> None:
        # 创建一个工作簿
        wb = Workbook()
        # 创建一个工作表
        ws = wb.active
        # 初始化一个行数变量，用于跟踪插入行的位置
        row_num = 1
        # 逐个写入每个张量到工作表中，并在每个张量之后插入一个空行
        for tensor in data:
            # 将 8x8 的张量转换为 DataFrame
            df = pd.DataFrame(tensor)

            # 写入 DataFrame 数据到工作表中
            for r_idx, row in enumerate(df.values):
                for c_idx, value in enumerate(row):
                    ws.cell(row=row_num, column=c_idx + 1, value=value)
                row_num += 1

            # 在张量之后插入一个空行
            row_num += 1
        # 保存 Excel 文件
        wb.save(file_name)

    # 将经过上面处理的数据再经过归一化，处理为可以放入正则化流中学习的数据
    def to_normalizing_flow_data(self, data) -> list:
        # 对于每一个矩阵，计算其集几何平均值：将8个元素连续乘，然后对结果开8次方
        result = []
        for matrix in data:
            x = []  # 这个数组用来记录每一个矩阵的8个几何平均值
            for i in matrix:  # 一个matrix里面有8行，就会计算出来8个元素
                num = 1
                for j in range(8):
                    num = i[j]*num
                # 对num开n次方
                num = num**(1/8)
                x.append(num)
            result.append(x)

        # 对每一行元素进行归一化处理
        normalized_result = []
        for i in result:
            i = np.array(i)
            normalized_v = i / np.sqrt(np.sum(i**2))
            normalized_result.append(normalized_v)

        # 将数据按照列进行规划排列
        normalized_result = np.array(normalized_result)
        final_list = []
        # 将这个矩阵，按照列保存
        for i in range(8):
            final_list.append(normalized_result[:, i])
        return final_list
