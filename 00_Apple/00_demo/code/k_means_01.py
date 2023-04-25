import numpy as np
import pandas as pd

#使用pandas读取xlsx文件
frame = pd.read_excel("../data/replace_raw_data.xlsx")

#使用to_numpy()将dataframe转换成array
data = frame.to_numpy()

#print(data)
#将所有数据转换成三维数组，每一个元素都是一个二维数组矩阵
count = 0
List = []
element = []

for i in data:
    if (count % 9) != 0:
        element.append(i)
    elif element:
    #排除第一个二维数组是空的情况
        List.append(element)
        element = []
    count += 1


#查看前三个数
def scanTir():
    for i in range(3):
        print(List[i])


#对矩阵的每一个数字进行计算
#公式如下 i = power(sqrt(2), i-2)
print(len(List))
for i in List:
    for j in range(8):
        for k in range(8):
            if i[j][k] != 9:
                i[j][k] =  np.power(np.sqrt(2), i[j][k] - 2)


#将对角线转换为1
#创建一个对角阵
eye = 8*np.identity(8)
#print(eye)
for i in range(len(List)):
    List[i] = List[i] - eye


#补齐每一矩阵的上三角矩阵
for i in List:
    for j in range(8):
        for k in range(8):
            if i[j][k] == 9.0:
                i[j][k] =  1/(i[k][j])


print(type(List))
#print(List)
#将处理好的数据保存
#List是三维数据，不能写入3维数据，先转换成二维数据
#每间隔8行，空一行
# temp_list = []
# count = 0
# for i in List:
#     for j in i:
#         if (count % 8 == 0) and (temp_list != []):
#             temp_list.append([])
#         temp_list.append(j)
#         count += 1
#
#
# #将np转成pandas
# s = pd.DataFrame(temp_list)
# #将pandas保存在excel文件
# s.to_excel("../data/temp.xlsx")


#首先将List转换为array
array_list = np.array(List)
print(array_list)
# 将三维数组转换为DataFrame对象,这个对象具有86个元素，每个元素是一个序列，这个序列中有64个元素
df = pd.DataFrame(array_list.reshape(86, 64))
# 创建一个Excel写入器
writer = pd.ExcelWriter('output.xlsx')
# 将DataFrame对象写入Excel文件中
df.to_excel(writer, sheet_name='Sheet1', index=False)
# 保存Excel文件
writer.save()

#将df由dataframe转换成array
data_original = np.array(df)
np.save('data_oraiginal.npy', data_original)
