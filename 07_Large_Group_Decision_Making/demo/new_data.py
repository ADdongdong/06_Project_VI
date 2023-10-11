import pandas as pd
import numpy as np
from adaptiveConflictRevising import *


def read_excel_and_create_tensor(file_path):
    # 读取Excel文件的第四行到第11行的第E列到第L列
    data = pd.read_excel(file_path, header=None,
                         skiprows=3, usecols="E:L", nrows=8)
    return data.to_numpy()


#result = []
expert_data = []
for i in range(1, 52):
    # 定义文件的名字，1.xlsx - 51.xlsx
    file_path = "./New data1/" + str(i) + ".xlsx"
    tensor = read_excel_and_create_tensor(file_path)

    # 创建一个张量数组来保存这个文件
    tensor_array = np.array(tensor)
    # print(tensor)
    expert_data.append(ScoreWeight(tensor_array))
    # result.append(tensor_array)

# print(f"输入到读取到文件的数量{len(result)}")
# print(result)

# 使用adaptiveConflict来处理冲突

obj = adaptiveConfilctRevising(expert_data, 15, 1000)
#result1 = obj.fcm()
result = obj.adaptive_conflict_resolution_model()
