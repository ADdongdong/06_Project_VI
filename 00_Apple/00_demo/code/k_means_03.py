from k_means_02 import distEuclid,initCentroid,KMeans
from k_means_01 import List
import numpy as np
#from sklearn.cluster import KMeans
import pandas as pd

'''
思路：
1.展开成一维列表，方便聚类算法进行聚类
2.将元素进行去重，相同元素只保留4个
4.对元素进行聚类
5.输出聚类结果
'''

#1.将三维列表展开成二维列表
arr_list = np.array(List)
new_list = arr_list.reshape(arr_list.shape[0], -1)

#查看dtype
print(new_list.dtype)
#将new_list的类型转换成float类型，便于进行去重
new_list_float = new_list.astype('float')
print(new_list_float.dtype)

#2.对展开的数据进行去重
arr_unique_counts, counts = np.unique(new_list_float, axis=0, return_counts=True)
np.save('arr_unique_count.npy',arr_unique_counts)

print("去重过后数据中元素的个数是", len(arr_unique_counts))


#返回值的第一列元素就是对应的每个数据被分到的类

#定义二分类的函数，每次二分类都返回两个array
#每次输入的是要分类的numpy,array数组
def b_KMeans(_data):
    ##使用sklearn的二分聚类尝试
    cp, cluster = KMeans(_data, 2)

    cluster_list = [row[0] for row in cluster]
    cluster_list = np.array(cluster_list)
    #print(cluster_list)
    # 将分类信息添加到原始二维数组的最后一列
    data_cluster = np.hstack((_data, cluster_list.reshape(-1, 1)))

    # 将两类数据保存在两个变量中
    _data_cluster_0 = []
    _data_cluster_1 = []

    # 对依照添加的分类信息cluster_list进行分类，
    # 二分类有两类0.和1.类
    for i in data_cluster:
        if i[64] == 0.0:
            _data_cluster_0.append(list(i))
        elif i[64] == 1.0:
            _data_cluster_1.append(list(i))

    # 删除二维数组的最后一列
    _data_cluster_0 = np.array(_data_cluster_0)
    _data_cluster_1 = np.array(_data_cluster_1)

    if _data_cluster_0.size != 0:
        _data_cluster_0 = _data_cluster_0[:, :-1]
    if _data_cluster_1.size != 0:
        _data_cluster_1 = _data_cluster_1[:, :-1]

    return _data_cluster_0, _data_cluster_1

#定义一个递归算法，如果遇到len<=4就把这个自列表添加到result中，如果大于就继续二分类
#递归三要素，递归终止条件，递归返回值，递归函数参数

result = []
result_len = 0
def B_kmeans(_data):
    global result_len#声明为全局变量
    stack = [_data]

    #只要栈不为空，就继续循环
    #栈里面始终维持的是len>4的簇
    #如果len<=4就让这个簇里面的元素进入result
    while len(stack) != 0:
        #pop默认删除列表最后一个元素，并将其返回
        #print("当前栈中未被二分的簇的个数", len(stack))#
        top_data = stack.pop()
        #处理栈顶结点
        data_1, data_2 = b_KMeans(top_data)
        #print("当前二分第1个簇中元素个数",len(data_1))
        #print("当前二分第2个簇中元素个数",len(data_2))
        #如果当前簇中元素小于等于4，都加入到result中，包括0也加入，否则，就加入stack
        if len(data_1) <= 4:
            result.append(list(data_1))
        else:
            stack.append(data_1)

        if len(data_2) <= 4:
            result.append(list(data_2))
        else:
            stack.append(data_2)

B_kmeans(arr_unique_counts)
cluster_0, cluster_1 = b_KMeans(arr_unique_counts)
print(len(cluster_0))
print(len(cluster_1))


#聚类结束，结果保存在了result数组中
result = np.array(result)

#print(num)
#将计算出来的结果保存在result.npy文件中， 下次就不用重新计算了
np.save('result.npy', result)