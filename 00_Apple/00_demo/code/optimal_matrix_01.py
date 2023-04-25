import operator

import get_optimal_matrix as gom
import numpy as np

result = np.load('result.npy', allow_pickle=True)

#使用reshape对result中的所有一维数组转换成8乘8的二维数组
for cluster in result:#这里的每一个cluster都代表一个簇
    for i in range(len(cluster)):#range(len(cluster))是每个cluster中元素的个数
        cluster[i] = cluster[i].reshape((8,8))
#现在result中存放了21个cluster，每个cluster中是多个8乘8的二维数组，


#构造区间区间，这里的每一个区间的数据类型仍然是list
#将result中，簇对应的元素放入同一个区间
#使用numpy中的stack函数
#将每个簇中元素的个数保存在cluster_len中
#将矩阵对应元素整合在一起，构成区间
combin_result = []
for cluster in result:
    if len(cluster) == 1:
        List = []
        cluster_1 = []
        for i in range(8):
            List = [[x] for x in cluster[0][i]]
            cluster_1.append(List)
        combin_result.append(cluster_1)
    elif len(cluster) == 2:
        cluster_combin = np.stack((cluster[0], cluster[1]), axis=2)
        combin_result.append(cluster_combin)
    elif len(cluster) == 3:
        cluster_combin = np.stack((cluster[0], cluster[1], cluster[2]),axis=2)
        combin_result.append(cluster_combin)
    elif len(cluster) == 4:
        cluster_combin = np.stack((cluster[0], cluster[1], cluster[2], cluster[3]), axis=2)
        combin_result.append(cluster_combin)


print(len(combin_result[1]))

#数据类型转换，将数据中区间由list转换为optimal_matrix可用的struct_matrix
#定义一个数组，用来保存区间转换成struct_matrix的数据
interval_matrix_result = []

#这里的每一个interval_matrix就是一个簇组成的区间矩阵
for interval_matrix in combin_result:
    interval_matrix_len = len(interval_matrix[0][0])
    # 定义数值区间矩阵，将combin_result中每一个区间转换成一个struct_matrix
    numerical_interval_matrix = np.empty((8, 8), dtype=gom.struct_matrix)
    for i in range(8):
        for j in range(8):
            #定义一个区间结构体
            struct = gom.struct_matrix()
            #给struct的count赋值
            struct.count = interval_matrix_len
            struct.list = interval_matrix[i][j]
            #print(struct.list)
            numerical_interval_matrix[i][j] = struct
    #每转换好一个矩阵，就将这个转换过后的矩阵加入到interval_matrix_result中，这个数组最终的元素个数，就是簇数
    interval_matrix_result.append(numerical_interval_matrix)


better_matrix_csv = "../data/better_matrix_csv.csv"
norm_csv = "../data/norm_csv.csv"
#调用get_optimal_matrix函数来求解最优矩阵
all_normalization = gom.get_optimal_matrix(8, interval_matrix_result, 7,better_matrix_csv,7, norm_csv)

#print(all_normalization)#得出的结果应该是已经平均值归一化过后的数据
sum = 0
for i in range(8):
    sum += all_normalization[2][i]
#print(sum)

#将结果all_normalization保存在npy文件中
all_normalization = np.array(all_normalization)
np.save('all_normalization.npy', all_normalization)

#将all_normalization中的数据，按列组织，也就是说，A1对应一个一维数组，这个数组中有21个元素
normalizationl_list = []

for j in range(8):
    List = []
    for i in all_normalization:
        List.append(i[j])
    normalizationl_list.append(List)


#将A1-A8按列组织有的数据保存在normalization_list.npy中
normalizationl_list = np.array(normalizationl_list)
np.save('normalization_list.npy', normalizationl_list)