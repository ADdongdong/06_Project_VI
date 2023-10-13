import pandas as pd
import numpy as np
from openpyxl import Workbook


def read_excel_and_create_tensor(file_path):
    # 读取Excel文件的第四行到第11行的第E列到第L列
    data = pd.read_excel(file_path, header=None,
                         skiprows=3, usecols="E:L", nrows=8)
    return data.to_numpy()


result = []
for i in range(1, 52):
    # 定义文件的名字，1.xlsx - 51.xlsx
    file_path = "./New data1/" + str(i) + ".xlsx"
    tensor = read_excel_and_create_tensor(file_path)

    # 创建一个张量数组来保存这个文件
    tensor_array = np.array(tensor)
    result.append(tensor_array)


# 将51个张量保存成npy格式，方便下次使用
my_array = np.array(result)
# 保存到文件
np.save("new_data.npy", my_array)


# 创建一个工作簿
wb = Workbook()

# 创建一个工作表
ws = wb.active

# 初始化一个行数变量，用于跟踪插入行的位置
row_num = 1

# 逐个写入每个张量到工作表中，并在每个张量之后插入一个空行
for tensor in result:
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
wb.save('output.xlsx')


print(f"输入到读取到文件的数量{len(result)}")
print(result[0])
print(result[1])
print(result[2])
