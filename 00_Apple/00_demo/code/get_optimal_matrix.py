import numpy as np
import pandas as pd
import math
import csv
import os

#定义区间矩阵的结构体(存储区间矩阵的长度与具体数据)
class struct_matrix(object):
    #这个结构体就是保存区间的，这样可以降低数据的维数，所以在处理数据的时候，
    #需要宝8,8,n的三维数组保存成8,8的列表，列表中每个元素是一个struct_matrix
    def __init__(self):
        self.count = 0#区间有几个元素
        self.list = []#区间列表

#删除数值矩阵中已经得到的数据
def del_data(list_real_index, numerical_interval_matrix):
    list_delete_index = []
    for i in range(len(list_real_index)):
        list_delete_index.append(list_real_index[i])

    # 先对得到的序列副本进行从大到小排序
    list_delete_index.sort()
    list_delete_index.reverse()

    # 将其要删除的数据删除
    for i in range(len(list_real_index)):
        numerical_interval_matrix = np.delete(numerical_interval_matrix,
                                                  list_delete_index[i],
                                                  axis=1)  # 删除行
        numerical_interval_matrix = np.delete(numerical_interval_matrix,
                                                  list_delete_index[i],
                                                  axis=0)  # 删除列
    return  numerical_interval_matrix

#将矩阵或列表写入csv文件
def write_csv(path,datas,flag):
    file_csv = open(path, 'a+',newline='')
    writer = csv.writer(file_csv)
    if flag == 1:         #flag为1时，写入的是矩阵
        for data in datas:
            writer.writerow(data)
        file_csv.close()
    else:
        writer.writerow(datas)
        file_csv.close()

#将原始的矩阵转化为数值矩阵
def get_numerical_matrix(numerical_interval_matrix,original_interval_matrix, max_index):
    for i in range(len(numerical_interval_matrix)):
        for j in range(len(numerical_interval_matrix[i])):
            value_of_numerical_interval_matrix = []
            for k in range(1,len(original_interval_matrix[i][j]) - 1):
                if original_interval_matrix[i][j][k] == ",":
                    continue
                elif original_interval_matrix[i][j][k] == " ":
                    continue
                else:
                    value_of_numerical_interval_matrix.append(
                    math.pow(math.sqrt(2), int(original_interval_matrix[i][j][k]) - max_index/2))
            numerical_interval_matrix[i][j].count = len(value_of_numerical_interval_matrix)
            numerical_interval_matrix[i][j].list = value_of_numerical_interval_matrix
    return  numerical_interval_matrix

'''
step1:定义一个随机选择初始矩阵的函数
参数：numerical_interval_matrix 数字区间矩阵
'''
# 初始化改成取上三角矩阵
def random_choose_original_matrix(numerical_interval_matrix):
    [long, weight] = np.shape(numerical_interval_matrix)
    random_original_metrix = np.empty((long, weight))
    for i in range(len(random_original_metrix)):
        #for j in range(i+1, len)
        for j in range(i+1, len(random_original_metrix)):
            random_original_metrix[i][j] = numerical_interval_matrix[i][j].list[
                numerical_interval_matrix[i][j].count // 2]
    #用选出来的上半三角矩阵计算下半三角矩阵
    for i in range(len(random_original_metrix)):
        for j in range(0, i):
            random_original_metrix[i][j] = 1/random_original_metrix[j][i]
    #将对角线元素设置为1
    for i in range(len(random_original_metrix)):
        for j in range(len(random_original_metrix)):
            if i == j:
                random_original_metrix[i][j] = 1
    return random_original_metrix

'''
step2.1： Ak* 对矩阵进行归一化(不用修改，可用)
这里应该是对挑选出来的初始化数值矩阵进行归一化。后面的get_optimal_matrix函数中会用到。
'''
def normalization(matrix):
    norm_multiply = []      # 求8个乘积
    norm_geometric_mean = []      # 求几何平均值
    normlization_result = []         # 归一化
    [long,wide] = np.shape(matrix)      # 几何平均值
    #将每行数据取出，计算其乘积
    for i in range(long):
        muls = np.prod(matrix[i])
        norm_multiply.append(muls)
    #将每行数据的乘积开n次方
    for i in range(len(norm_multiply)):
        norm_geometric_mean.append(math.pow(norm_multiply[i], 1 / len(norm_multiply)))
    sum_norm_geometric_mean = sum(norm_geometric_mean)
    #归一化处理
    for i in range(len(norm_geometric_mean)):
        normlization_result.append(norm_geometric_mean[i] / sum_norm_geometric_mean)
    return normlization_result

#归一化构成的矩阵
def aistar(list_norm):
    list = []
    for i in range(len(list_norm)):
        for j in range(len(list_norm)):
            list.append(list_norm[i]/list_norm[j])
    matrix = np.array(list)
    matrix = matrix.reshape(len(list_norm), len(list_norm))
    return matrix

'''
step2.2 对ci矩阵进行平均值的求解(不用修改，可用)
'''
def get_ci_matrix(original_matrix, original_matrix_star):
    ci_matrix = np.zeros((len(original_matrix),len(original_matrix)))
    for i in range(len(original_matrix)):
        for j in range(len(original_matrix[i])):
            ci_matrix[i][j] = original_matrix[i][j] / original_matrix_star[i][j]
    sum_list = ci_matrix.sum(1)
    average = sum(sum_list) / len(sum_list)
    return average

#重新求解一个矩阵
def get_newinitial_matrix(numerical_matrix, numerical_interval_matrix):
    numerical_matrix1 = np.zeros((len(numerical_matrix),len(numerical_matrix)))
    for i in range(len(numerical_matrix)):
        for j in range(i, len(numerical_matrix[i])):
            list_comparison = []
            for m in range(len(numerical_interval_matrix[i][j].list)):
                list_comparison.append(abs(math.log(
                    numerical_interval_matrix[i][j].list[m],math.e) -
                            math.log(numerical_matrix[i][j],math.e)))
            numerical_matrix1[i][j] = numerical_interval_matrix[i][j].list[list_comparison.index(min(list_comparison))]

    #计算下半三角
    for i in range(len(numerical_matrix)):
        for j in range(i):
            numerical_matrix1[i][j] = 1 / numerical_matrix1[j][i]

    return numerical_matrix1

def get_optimal_matrix(data_degree, numerical_interval_matrix, max_index, better_matrix_csv, rounding, norm_csv):
    ci_probolity = [1.36, 1.26, 1.12, 0.89, 0.52]  # 从区间矩阵选择最优矩阵的计算系数

    '''
    #data_total, data_degree, max_index,
    #给表格加头部
    list_original_index = []
    for i in range(data_degree):
        list_original_index.append(i)

    # 读取原始(符号矩阵)的区间矩阵
    # 改成读取数字区间矩阵（由符号矩阵处理过后的数字矩阵）
    if not os.path.exists(path_interval_matrix):
        os.makedirs(path_interval_matrix)
    csv_file = pd.read_csv(path_interval_matrix)
    csv_file = csv_file.values  # 将csv文件中的区间矩阵取出

    
    for k in range(data_tot  al):
        # 读出原始矩阵(不需要)
        original_matrix = csv_file[
            np.arange((data_degree + 1) * k + 1, (data_degree + 1) * k + (data_degree + 1))]

        # 定义数值区间矩阵（不需要）
        numerical_interval_matrix = np.empty((data_degree, data_degree),
                                             dtype=struct_matrix)  # 改7乘7
        for i in range(len(numerical_interval_matrix)):
            for j in range(len(numerical_interval_matrix[i])):
                numerical_interval_matrix[i][j] = struct_matrix()

        # 将读取的区间矩阵转化为数值区间矩阵（不需要）
        numerical_interval_matrix = get_numerical_matrix(numerical_interval_matrix,
                                                         original_matrix, max_index)

        # 删除已经选择PSR最高的数据（不需要）
        #numerical_interval_matrix = del_data(list_real_index, numerical_interval_matrix)
        '''

    #最终结果保存在这个列表中
    all_normalization = []
    #这里直接开始遍历，就遍历已经处理好的，numerical_interval_matrix,这个矩阵中，包含了21个矩阵，也就是21个簇
    for numerical_matrix in numerical_interval_matrix:
        # 随机选择初始的矩阵
        initial_matrix = random_choose_original_matrix(
            numerical_matrix)

        new_ci = -1
        old_ci = -2
        # 选择一致性最优的矩阵
        while (old_ci != new_ci):
            old_ci = new_ci
            initial_matrix_star = aistar(normalization(initial_matrix))
            matrix_ci_value = get_ci_matrix(initial_matrix,
                                            initial_matrix_star)
            if data_degree - rounding > 3:
                new_ci = (matrix_ci_value - data_degree - rounding) / (
                        (data_degree - 1) * ci_probolity[rounding])
            else:
                new_ci = (matrix_ci_value - data_degree)
            initial_matrix = get_newinitial_matrix(
                initial_matrix_star, numerical_matrix)

        return_original_matrix = np.zeros((data_degree - rounding, data_degree - rounding))
        for i in range(data_degree - rounding):
            for j in range(data_degree - rounding):
                return_original_matrix[i][j] = round(max_index / 2 + math.log(initial_matrix[i][j], math.sqrt(2)),
                                                     1)

        write_csv(better_matrix_csv, [str(rounding + 1) + "time"], 0)
        #write_csv(better_matrix_csv, ["matrix" + str(numerical_matrix + 1)], 0)#修改了k修改成了numerical_matrix
        write_csv(better_matrix_csv, return_original_matrix, 1)
        write_csv(better_matrix_csv, [[]], 1)

        all_normalization.append(
            normalization(initial_matrix))
    write_csv(norm_csv, [str(rounding + 1) + "time"], 0)
    write_csv(norm_csv, all_normalization, 1)
    write_csv(norm_csv, [[]], 1)

    return all_normalization
