import numpy as np
import pandas as pd
import math
import csv
import os

#分组
def get_group(group,all_list_normalization,data_degree):

    # 将列表中的数据进行分组 step 1：找到所有列表中的
    for i in range(len(all_list_normalization)):
        list_group = []
        max_value = max(all_list_normalization[i])
        for j in range(len(all_list_normalization[i])):
            if max_value == all_list_normalization[i][j]:
                list_group.append(j)
        for j in range(len(list_group)):
            group[list_group[j]].append(all_list_normalization[i])

#计算行除列的值
def get_probility(matrix_average_group,data_degree):
    [long, wide] = np.shape(matrix_average_group)
    probility_Ai = []
    list_probility_Ai_index = []
    for i in range(data_degree):
        probility_Ai.append(-1)
        list_probility_Ai_index.append(i)
    list_group_novalue_index = []
    #先将行为0的下标记录在list_avetemp中
    for i in range(wide):
        sum_rows = 0.0
        for j in range(wide):
            sum_rows += matrix_average_group[i][j]
        if sum_rows==0:
            list_group_novalue_index.append(i)
    #将list_avetemp中的值拿出来，将其对应的下标中的值的最终结果置为0
    for i in range(len(list_group_novalue_index)):
        probility_Ai_index = list_group_novalue_index[i]
        probility_Ai[probility_Ai_index] = 0.0
    #然后删除其中分组没有数据的那一行一列
    list_group_novalue_index.reverse()
    for i in range(len(list_group_novalue_index)):
        matrix_average_group = np.delete(matrix_average_group, list_group_novalue_index[i], axis=1)  # 删除行
        matrix_average_group = np.delete(matrix_average_group, list_group_novalue_index[i], axis=0)  # 删除列
        #删除概率为零对应的下标
        list_probility_Ai_index.remove(list_group_novalue_index[i])
    [long, wide] = np.shape(matrix_average_group)
    for i in range(wide):
        sum = 0.0
        for j in range (wide):
            sum += matrix_average_group[i][j] / matrix_average_group[j][i]
        #找出对应的真实下标
        real_index = list_probility_Ai_index[i]
        probility_Ai[real_index] = sum
    return probility_Ai

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

def get_order(data_degree, rounding, all_list_normalization, group_csv, probility_csv,list_original_index, list_real_index):
    # 根据归一化结果分组
    # step 1: 定义一个字典， 将其归一化列表中的值分到字典中
    group = {}
    for i in range(data_degree - rounding):
        group.setdefault(i, [])

    get_group(group, all_list_normalization, data_degree - rounding)

    # 将分组结果写入文件中
    write_csv(group_csv, [str(rounding + 1) + "time"], 0)
    for i in range(data_degree - rounding):
        if i in group.keys():
            write_csv(group_csv, ["group" + str(i + 1)], 0)
            write_csv(group_csv, group[i], 1)
        else:
            write_csv(group_csv, ["group" + str(i + 1)], 0)
            write_csv(group_csv, [[]], 1)

    # 求每组比率
    group_rate = []
    total_rate = 0
    for i in range(len(group)):
        total_rate = total_rate + len(group[i])

    for i in range(data_degree - rounding):
        if i in group.keys():
            group_rate.append(len(group[i]) / total_rate)
        else:
            group_rate.append(0)

    # 对组内数据进行聚合
    probolity_matrix = np.zeros((data_degree - rounding, data_degree - rounding))
    # 按列进行平均
    for i in range(data_degree - rounding):
        if len(group[i]) == 1:
            probolity_matrix[i] = group[i][0]
        elif len(group[i]) > 1:
            probolity_matrix[i] = np.array(group[i]).mean(axis=0)
        else:
            continue

    write_csv(probility_csv, [str(rounding + 1) + "time"], 0)
    write_csv(probility_csv, probolity_matrix, 1)
    write_csv(probility_csv, [[]], 1)
    write_csv(probility_csv, ["group_rate:"], 0)
    write_csv(probility_csv, group_rate, 0)

    probility_result = get_probility(probolity_matrix, data_degree - rounding)

    for i in range(len(probility_result)):
        probility_result[i] = probility_result[i] * group_rate[i]

    probility_max_index = probility_result.index(max(probility_result))  # 得到概率值最大的下标
    # probility_value.append(max(probility_result))  # 保存最大概率值
    confuse_index = list_original_index[probility_max_index]  # 得到真正的下标
    list_real_index.append(confuse_index)  # 保存真正的下标
    list_original_index.remove(confuse_index)
    return list_original_index, list_real_index




