import numpy as np


data = np.load('arr_unique_count.npy', allow_pickle=True)
#print(data[0])
#将data中的所有一维数据都转换成8乘8的二维数据

List = []
for i in data:
    matrix = i.reshape((8,8))
    List.append(matrix)

#print(List[0])

#对于每一个矩阵，计算其集几何平均值：将8个元素连续乘，然后对结果开8次方
result = []
for matrix in List:
    x = []#这个数组用来记录每一个矩阵的8个几何平均值
    for i in matrix:#一个matrix里面有8行，就会计算出来8个元素
        num = 1
        for j in range(8):
            num = i[j]*num
        #对num开n次方
        num = num**(1/8)
        x.append(num)
    result.append(x)

#对每一行元素进行归一化处理
normalized_result = []
for i in result:
    i = np.array(i)
    normalized_v = i / np.sqrt(np.sum(i**2))
    normalized_result.append(normalized_v)

#检测是否归一化了(归一化是缩放到0-1之间）并不一定求和就是1

