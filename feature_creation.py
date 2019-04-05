# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import csv


class FeatureCreation:
    def __init__(self, wava_data):
        """
        :param wava_data: np.array
        """
        self.wave_data = wava_data

    def average_x(self):
        """
        均值
        :return:
        """
        avg = []
        for wave in self.wave_data:
            avg.append(np.average(wave))
        return avg

    def p_x(self):
        """
        峰值
        :return:
        """
        m = []
        for wave in self.wave_data:
            m.append(max(abs(wave)))
        return m

    def rms_x(self):
        """
        均方根
        :return:
        """
        rms = []
        for wave in self.wave_data:
            rms.append(sum(wave*wave)/len(wave)**0.5)
        return rms

    def std_x(self):
        """
        标准差
        :return:
        """
        std = []
        for wave in self.wave_data:
            std.append(np.std(wave))
        return std

    def sk_x(self):
        """
        偏度指标
        :return:
        """
        sk = []
        avg = np.array(self.average_x()).reshape((-1, 1))
        std = np.array(self.average_x()).reshape((-1, 1))
        for i, j in zip((self.wave_data-avg)**3, (self.wave_data.shape[1]-1)*(std**3)):
            sk.append(sum(i)/j)
        return np.array(sk)

    def kv_x(self):
        """
        峭度指标
        :return:
        """
        kv = []
        avg = np.array(self.average_x()).reshape((-1, 1))
        std = np.array(self.average_x()).reshape((-1, 1))
        for i, j in zip((self.wave_data - avg) ** 4, (self.wave_data.shape[1] - 1) * (std ** 4)):
            kv.append(sum(i) / j)
        return np.array(kv)

    def sf_x(self):
        """
        波形指标
        :return:
        """
        avg = np.array(self.average_x()).reshape((-1, 1))
        rms = np.array(self.rms_x()).reshape((-1, 1))
        return rms/avg

    def cf_x(self):
        """
        峰值指标
        :return:
        """
        p = np.array(self.p_x()).reshape((-1, 1))
        rms = np.array(self.rms_x()).reshape((-1, 1))
        return p / rms


class FeatureEvaluation:
    def __init__(self, features, labels):
        self.feature_rows = features.shape[0]
        self.feature_num = features.shape[1]
        self.feature = features
        self.feature['labels'] = labels
        self.feature0 = self.feature.loc[self.feature['labels'] == 0, :].iloc[:, :-1]
        self.feature1 = self.feature.loc[self.feature['labels'] == 1, :].iloc[:, :-1]
        self.feature2 = self.feature.loc[self.feature['labels'] == 2, :].iloc[:, :-1]

    def calculate_index(self):
        index_array = []
        for i in range(self.feature_num):
            print(i)
            ind = self.comprehensive_evaluate_index(i)
            print(ind)
            index_array.append(ind)
        return pd.DataFrame(index_array, self.feature.columns[0:-1])

    def out_distance(self, feature_k):
        """
        计算第k个特征对于所有类的类间距离的平均值
        :param feature_k: 第k个特征
        :return:
        """
        d0 = self.feature0.iloc[:, feature_k].mean()
        d1 = self.feature1.iloc[:, feature_k].mean()
        d2 = self.feature2.iloc[:, feature_k].mean()
        return (((d0-d1)**2+(d0-d2)**2+(d1-d2)**2)*2/6)**0.5

    def in_distance(self, feature_k):
        """
        计算第k个特征对于所有类的类内距离的平均值
        :param feature_k: 第k个特征
        :return:
        """
        f_0 = self.feature0.iloc[:, feature_k]
        f_1 = self.feature1.iloc[:, feature_k]
        f_2 = self.feature2.iloc[:, feature_k]
        s_0, s_1, s_2 = 0, 0, 0
        for i in range(len(f_0)):
            for j in range(len(f_0)):
                s_0 += (f_0.iloc[i]-f_0.iloc[j])**2

        for i in range(len(f_1)):
            for j in range(len(f_1)):
                s_1 += (f_1.iloc[i]-f_1.iloc[j])**2

        for i in range(len(f_2)):
            for j in range(len(f_2)):
                s_2 += (f_2.iloc[i]-f_2.iloc[j])**2
        s_0 = (s_0 / (self.feature_rows * (self.feature_rows - 1))) ** 0.5
        s_1 = (s_1 / (self.feature_rows * (self.feature_rows - 1))) ** 0.5
        s_2 = (s_2 / (self.feature_rows * (self.feature_rows - 1))) ** 0.5
        return (s_0+s_1+s_2)/3

    def comprehensive_evaluate_index(self, k):
        return self.out_distance(k)/self.in_distance(k)







