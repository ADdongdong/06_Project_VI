import pandas as pd
from k_means_01 import List
import numpy as np


##计算欧式距离
def distEuclid(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

## 初始化簇中心点 一开始随机从样本中选择k个 当做各类簇的中心
def initCentroid(data, k=2):
    num, dim = data.shape
    centpoint = np.zeros((k, dim))
    l = [x for x in range(num)]
    #print(l)
    np.random.shuffle(l)
    for i in range(k):
        index = l[i]
        #print(index)
        centpoint[i] = data[index]
    return centpoint


## 进行KMeans分类
def KMeans(data, k):
    ##样本个数
    num = np.shape(data)[0]
    ##记录各样本 簇信息 0:属于哪个簇 1:距离该簇中心点距离
    cluster = np.zeros((num, 2))
    cluster[:, 0] = -1

    ##记录是否有样本改变簇分类
    change = True
    ##初始化各簇中心点
    cp = initCentroid(data, k)

    while change:
        change = False

        ##遍历每一个样本
        for i in range(num):
            minDist = 9999.9
            minIndex = -1

            ##计算该样本距离每一个簇中心点的距离 找到距离最近的中心点
            for j in range(k):
                dis = distEuclid(cp[j], data[i])
                if dis < minDist:
                    minDist = dis
                    minIndex = j

            ##如果找到的簇中心点非当前簇 则改变该样本的簇分类
            if cluster[i, 0] != minIndex:
                change = True
                cluster[i, :] = minIndex, minDist

        ## 根据样本重新分类  计算新的簇中心点
        for j in range(k):
            pointincluster = data[[x for x in range(num) if cluster[x, 0] == j]]
            cp[j] = np.mean(pointincluster, axis=0)

    #print("finish!")
    return cp, cluster
