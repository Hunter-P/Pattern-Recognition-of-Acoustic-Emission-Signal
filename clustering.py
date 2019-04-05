# -*- coding:utf-8 -*-


# 利用聚类算法进行无监督学习，将信号分类
# 高斯混合模型 | 模糊聚类 | kmeans

import pandas as pd
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import json
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random

# ------------------------------------------数据准备-------------------------------------------------

pd.set_option('display.width', 1000)


"""
features:
HH:MM:SS.mmmuuun
通道
上升时间
计数
能量
持续时间
幅值
平均频率
RMS
ASL
峰值频率
反算频率
初始频率
信号强度
绝对能量
中心频率
峰频
时间
"""
# ------------------------------------------高斯混合模型-----------------------------------------------


class GMM_P(object):
    def __init__(self, X):
        self.X = X.iloc[:, 2:].drop(['信号强度', '初始频率', '绝对能量', '中心频率', '峰频'], axis=1)
        self.X = StandardScaler().fit_transform(self.X)  # 标准化，方差为1，平均值为0， 应该用标准化
        # self.X = MinMaxScaler(feature_range=(0, 1)).fit_transform(self.X)  # 归一化，归一化是标准化的一种
        self.labels = 0

    def fit_predict(self, n_list):
        index_sc, index_ch = [], []
        for n in n_list:
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=7)
            gmm.fit(self.X)
            labels = gmm.predict(self.X)
            # print('labels:', type(labels), labels)
            m = self.evaluation_index_silhouette_coefficient(labels)
            k = self.evaluation_index_calinski_harabaz(labels)
            print("聚类中心数目：", n, " 轮廓系数:%.3f" % m, "| Calinski-Harabasz系数:%.3f" % k)
            index_sc.append(m)
            index_ch.append(k)
        return index_sc, index_ch

    def prediction(self, n):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=7)
        gmm.fit(self.X)
        labels = gmm.predict(self.X)
        return labels

    def evaluation_index_silhouette_coefficient(self, labels, distance='euclidean'):
        """
        聚类评估指标：轮廓系数，评分结果在[-1, +1]之间，评分结果越高，聚类结果越好
        :param labels:
        :param distance:
        :return:
        """
        return metrics.silhouette_score(self.X, labels, distance)

    def evaluation_index_calinski_harabaz(self, labels):
        """
        聚类评估指标：Calinski-Harabasz, 分数值越大则聚类效果越好
        :param labels:
        :return:
        """
        return metrics.calinski_harabaz_score(self.X, labels)


# -------------------------------------------FCM模型和kmeans++模型--------------------------------------------------


class FCM_P(object):

    # FLOAT_MAX = 1e10  # 设置一个较大的值作为初始化的最小的距离

    def __init__(self, dataset, cluster_center_number):
        """
        :param dataset: 数据集
        :param cluster_center_number: 聚类中心数目
        """
        self.dataset = dataset
        self.cluster_center_number = cluster_center_number

    # 随机化生成[-1,1]之间的中心点
    def get_random_center_point(self):
        center_point = np.array([random.uniform(-1, 1) for i in range(self.dataset.shape[1])])
        return center_point

    # 随机选择一个样本点作为初始中心
    def random_point_as_center_point(self):
        n = self.dataset.shape[0]
        return self.dataset[random.randint(0, n), ]

    # 计算一个样本点与已经初始化的各个聚类中心之间的距离，并选择其中最短的距离输出
    def nearest_distance(self, point, center_points):
        FLOAT_MAX = 1e10  # 设置一个较大的值作为初始化的最小的距离
        min_dist = FLOAT_MAX
        m = center_points.shape[0]
        for i in range(m):
            d = self.distance(center_points[i, ], point)
            if min_dist > d:
                min_dist = d
        return min_dist

    def get_initial_center_points(self):
        """
        生成所有初始聚类中心点，
        :return: 输出中心点矩阵，np.array
        """
        center_points = self.random_point_as_center_point()  # 获取第一个初始化中心点
        for i in range(self.cluster_center_number):
            nearset_distances = []
            for sample in self.dataset:
                min_dist = self.nearest_distance(sample, center_points)
                nearset_distances.append(min_dist)
            index = nearset_distances.index(max(nearset_distances))
            center_point = self.dataset[index, ]
            center_points = np.vstack((center_points, center_point))

        return center_points

    def distance(self, point_A, point_B):
        """
        # 计算两点之间的距离
        :param point_A:
        :param point_B:
        :return:
        """
        return sum((point_A-point_B)*(point_A-point_B).T)

    def k_means_plus_plus(self, epsilon=0.001):
        """
        k-means++算法
        :return: 样本分类列表point_classify，list；center_pointers 聚类中心坐标
        """
        n = self.dataset.shape[0]  # 样本的个数
        center_points = self.get_initial_center_points()  # 初始化中心点
        point_classify = [0]*n  # 样本点的分类
        change = True  # 判断是否需要重新计算聚类中心
        while change is True:
            for i in range(n):  # 对样本点进行分类
                distance_list = [self.distance(point, self.dataset[i, ]) for point in center_points]
                point_classify[i] = distance_list.index(min(distance_list))

            C = []
            for i in range(self.cluster_center_number):
                c_list = [point for point, classify in zip(self.dataset, point_classify) if classify == i]
                C.append(sum(c_list)/len(c_list))
            C = np.array(C)
            # center_points = np.array(center_points)
            flag = 0
            for i in range(len(C)):
                if abs(sum(C[i]-center_points[i])) <= epsilon:
                    flag += 1
                else:
                    center_points[i] = C[i]
            if flag == len(C):
                change = False

            # if (C == center_points).all() is not True:
            #     # 更新中心点
            #     center_points = C
            # else:
            #     change = False

        return point_classify, center_points

    def FCM(self, m=2, epsilon=0.000000001):
        """
        FCM算法，m的最佳取值范围为[1.5，2.5]
        :param m:m 权值
        :param epsilon:终止条件
        :return: 聚类中心矩阵C，np.array;  U:隶属度矩阵
        """
        U = self.initialise_U()  # 隶属度矩阵
        # print("U: ", U)
        J = 0  # 目标函数
        change = True  # 判断是否需要重新计算聚类中心
        C = 0
        # 计算聚类中心c
        while change is True:
            C = []  # C为聚类中心矩阵，np.array
            for i in range(self.cluster_center_number):
                c = sum([U[j, i]**m*self.dataset[j, ] for j in range(self.dataset.shape[0])])/sum(U[:, i]**m)
                C.append(c)
            C = np.array(C)
            # print('聚类中心矩阵C: ', "\n", C)
            # 计算目标函数J
            for i in range(self.cluster_center_number):
                j = sum([U[j, i]**m*(self.distance(C[i, ], self.dataset[j, ])**2) for j in
                         range(self.dataset.shape[0])])
                J += j
            # print('J: ', J)
            # 更新U矩阵
            new_U = np.array([[0]*self.cluster_center_number]*self.dataset.shape[0], dtype='float')
            for i in range(self.cluster_center_number):
                for j in range(self.dataset.shape[0]):
                    new_U[j, i] = 1/sum([(self.distance(self.dataset[j], C[i])/self.distance(self.dataset[j], C[l]))
                                         ** (2/(m-1)) for l in range(self.cluster_center_number)])
            # print('new U: ', new_U)
            count = 0
            for i in range(self.dataset.shape[0]):
                for j in range(self.cluster_center_number):
                    if abs(new_U[i, j]-U[i, j]) <= epsilon:
                        count += 1
            if float(count) >= self.dataset.shape[0]*self.cluster_center_number*0.6:
                change = False

            U = new_U
        # print('c: ', C)
        return C, U

    def initialise_U(self):
        """
        生成初始化隶属度矩阵U，每行加起来是1，输出为np.array结构
        """
        m = self.dataset.shape[0]
        n = self.cluster_center_number
        U = []
        for i in range(m):
            u = [random.randint(1, 10) for j in range(n)]
            u = np.array(u) / sum(u)
            U.append(u)

        return np.array(U)

    def FCM_classify(self, C):
        """
        FCM算法的聚类结果
        :param C:聚类中心矩阵
        :return: 分类结果,list
        """
        point_classify = [0] * self.dataset.shape[0]  # 样本点的分类
        for i in range(self.dataset.shape[0]):
            mid = [self.distance(self.dataset[i, ], point) for point in C]
            point_classify[i] = mid.index(min(mid))

        if sum(point_classify) < len(point_classify)/2:  # 1比0多，交换0,1
            for i in range(len(point_classify)):
                if point_classify[i] == 0:
                    point_classify[i] = 1
                else:
                    point_classify[i] = 0

        return point_classify

    def validity_index(self, C, U):
        """
        计算聚类有效性指标
        :param C:聚类中心矩阵
        :param U:隶属度矩阵
        :return: v
        """
        n = self.dataset.shape[0]
        k = self.cluster_center_number
        # 计算模糊分测度,gamma
        fenzi = 0.
        for i in range(n):
            fenzi += sum([(U[i, j]-1/k)**2*(self.distance(self.dataset[i, ], C[j, ])) for j in range(k)])
        fenmu = min([self.distance(C[0], C[i]) for i in range(1, k)])*n
        gamma = fenzi/fenmu
        # 计算有效性指标v
        v = gamma*(-1./n)*sum([sum([U[i, j]*math.log(U[i, j]) for i in range(n) for j in range(k)])])
        return v

    def evaluation_index_silhouette_coefficient(self, labels, distance='euclidean'):
        """
        聚类评估指标：轮廓系数，评分结果在[-1, +1]之间，评分结果越高，聚类结果越好
        :param labels:
        :param distance:
        :return:
        """
        return metrics.silhouette_score(self.dataset, labels, distance)

    def evaluation_index_calinski_harabaz(self, labels):
        """
        聚类评估指标：Calinski-Harabasz, 分数值越大则聚类效果越好
        :param labels:
        :return:
        """
        return metrics.calinski_harabaz_score(self.dataset, labels)


# index1 = data_1_1.loc[data_1_1['能量'] > 100, :].index
# index2 = data_1_2.loc[data_1_2['能量'] > 130, :].index
# index3 = data_1_3.loc[data_1_3['能量'] > 100, :].index
# index4 = data_1_4.loc[data_1_4['能量'] > 63, :].index
# data_1_lvbo = data_1.loc[sorted(list(index1)+list(index2)+list(index3)+list(index4)), :]
# data_1_lvbo = data_1_lvbo.loc[data_1_lvbo['通道'] == 1, :]
# data_1_lvbo = data_1_lvbo.iloc[:, 2:-1].drop(['信号强度'], axis=1)
#
# for i in range(2, 7):
#     model = FCM_P(StandardScaler().fit_transform(data_1_lvbo), i)
#     C, U = model.FCM()
#     labels = model.FCM_classify(C)
    # labels, center_points = model.k_means_plus_plus()
#     m = model.evaluation_index_silhouette_coefficient(labels)
#     k = model.evaluation_index_calinski_harabaz(labels)
#     print("聚类中心数目:", i, " 轮廓系数:%.3f" % m, "| Calinski-Harabasz系数:%.3f" % k)


# ------------------------------------------------KMeans------------------------------------------------------

class KMeans_P(object):
    def __init__(self, X):
        self.X = X.iloc[:, 2:].drop(['信号强度', '初始频率', '绝对能量', '中心频率', '峰频'], axis=1)
        self.X = StandardScaler().fit_transform(self.X)  # 标准化，方差为1，平均值为0， 应该用标准化
        # self.X = MinMaxScaler(feature_range=(0, 1)).fit_transform(self.X)  # 归一化
        self.labels = 0

    def fit_predict(self, n_list):
        index_sc, index_ch = [], []
        for n in n_list:
            kmeans = KMeans(n_clusters=n, random_state=7).fit(self.X)
            labels = kmeans.labels_
            m = self.evaluation_index_silhouette_coefficient(labels)
            k = self.evaluation_index_calinski_harabaz(labels)
            print("聚类中心数目：", n, " 轮廓系数:%.3f" % m, "| Calinski-Harabasz系数:%.3f" % k)
            index_sc.append(m)
            index_ch.append(k)
        return index_sc, index_ch

    def prediction(self, n):
        kmeans = KMeans(n_clusters=n, random_state=7).fit(self.X)
        return kmeans.labels_

    def evaluation_index_silhouette_coefficient(self, labels, distance='euclidean'):
        """
        聚类评估指标：轮廓系数，评分结果在[-1, +1]之间，评分结果越高，聚类结果越好
        :param labels:
        :param distance:
        :return:
        """
        return metrics.silhouette_score(self.X, labels, distance)

    def evaluation_index_calinski_harabaz(self, labels):
        """
        聚类评估指标：Calinski-Harabasz, 分数值越大则聚类效果越好
        :param labels:
        :return:
        """
        return metrics.calinski_harabaz_score(self.X, labels)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return [int(i) for i in obj]
        return json.JSONEncoder.default(self, obj)


# 将聚类结果保存到csv文件中
def to_json(X, file):
    """
    :param X: data_1_lvbo  一个试件的信号参数数据
    :param file: json文件名
    :return:
    """
    d_labels = {}

    for i in range(1, 5):
        print('----------通道%d-----------------------------' % i)
        data = X.loc[X['通道'] == i, :]
        labels = GMM_P(data).prediction(2)
        if sum(labels) > len(labels) / 2:
            for j in range(len(labels)):
                if labels[j] == 1:
                    labels[j] = 0
                else:
                    labels[j] = 1
        print('gmm:', sum(labels), len(labels))
        if i == 1:
            d_labels["GMM1"] = labels
        elif i == 2:
            d_labels["GMM2"] = labels
        elif i == 3:
            d_labels["GMM3"] = labels
        else:
            d_labels["GMM4"] = labels

        labels = KMeans_P(data).prediction(2)
        if sum(labels) > len(labels)/2:
            for j in range(len(labels)):
                if labels[j] == 1:
                    labels[j] = 0
                else:
                    labels[j] = 1
        print('kmeans:', sum(labels), len(labels))
        if i == 1:
            d_labels["kmeans1"] = labels
        elif i == 2:
            d_labels["kmeans2"] = labels
        elif i == 3:
            d_labels["kmeans3"] = labels
        else:
            d_labels["kmeans4"] = labels

        fcm_X = data.iloc[:, 2:].drop(['信号强度', '初始频率', '绝对能量', '中心频率', '峰频'], axis=1)
        fcm = FCM_P(StandardScaler().fit_transform(fcm_X), 2)
        # fcm = FCM_P(MinMaxScaler(feature_range=(0, 1)).fit_transform(fcm_X), 2)
        C, U = fcm.FCM()
        labels = fcm.FCM_classify(C)
        if sum(labels) > len(labels)//2:
            for j in range(len(labels)):
                if labels[j] == 1:
                    labels[j] = 0
                else:
                    labels[j] = 1
        print('fcm:', sum(labels), len(labels))
        if i == 1:
            d_labels["FCM1"] = labels
        elif i == 2:
            d_labels["FCM2"] = labels
        elif i == 3:
            d_labels["FCM3"] = labels
        else:
            d_labels["FCM4"] = labels

    with open(file, 'w', encoding='utf-8') as fo:
        json.dump(d_labels, fo, cls=MyEncoder)


def fo():
    m1_file = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\t滤波后-7200.csv"
    m2_file = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件2-0324\正式加载\t滤波后-7200.csv"
    m3_file = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件3-0324\正式加载\t滤波后-7200.csv"

    with open(m1_file, 'r') as fl:
        m1_data = pd.read_csv(fl)
    with open(m2_file, 'r') as fl:
        m2_data = pd.read_csv(fl)
    with open(m3_file, 'r') as fl:
        m3_data = pd.read_csv(fl)

    print('---------试件1------------------------------')
    file1 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\t_standardscale_labels.json"
    to_json(m1_data, file1)
    print('---------试件2------------------------------')
    file2 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件2-0324\正式加载\t_standardscale_labels.json"
    to_json(m2_data, file2)
    print('---------试件3------------------------------')
    file3 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件3-0324\正式加载\t_standardscale_labels.json"
    to_json(m3_data, file3)

fo()
