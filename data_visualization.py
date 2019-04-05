# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib
from pylab import mpl
import csv
import json
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_transformation import four_tensor_data, one_tensor_data


pd.set_option('display.width', 1000)

class Draw:

    def __init__(self):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        matplotlib.rcParams['xtick.direction'] = 'in'  # 刻度线朝内
        matplotlib.rcParams['ytick.direction'] = 'in'  # 刻度线朝内
        matplotlib.rcParams['savefig.dpi'] = 1000  # 分辨率

    def draw_deformation_of_load(self, deformation_array, load_array):
        """
        绘制荷载-位移曲线
        :param deformation_array:pandas.DataFrame
        :param load_array:
        :return:
        """
        fig = plt.figure(figsize=(8, 7))
        ax1 = fig.add_subplot(111)
        xminorlocator = MultipleLocator(0.5)
        ax1.xaxis.set_minor_locator(xminorlocator)  # 设置副刻度
        yminorlocator = MultipleLocator(25)
        ax1.yaxis.set_minor_locator(yminorlocator)  # 设置副刻度

        ax1.spines['right'].set_visible(False)  # 去掉边框
        ax1.spines['top'].set_visible(False)    # 去掉边框
        ax1.plot(deformation_array.iloc[:, 0], load_array, '>-', c='r', ms=8, linewidth=1, label='左接缝')
        ax1.plot(deformation_array.iloc[:, 1], load_array, 'o-', c='k', ms=8, linewidth=1, label='跨中')
        ax1.plot(deformation_array.iloc[:, 2], load_array, 's-', c='b', ms=8, linewidth=1, label='右接缝')
        ax1.set_xlabel('位移/mm', fontsize=23)
        ax1.set_ylabel('荷载/kN', fontsize=23)
        ax1.set_xlim(0, 11)
        ax1.set_ylim(0, 300)
        ax1.tick_params(axis='both', length=7, direction='out')
        ax1.tick_params(axis='both', length=4, direction='out', which='minor')
        plt.xticks(fontsize=22, family='Times New Roman')
        plt.yticks(fontsize=22, family='Times New Roman')
        plt.legend(loc='lower right', fontsize=20, frameon=False)
        plt.show()


    def draw_load_sub_feature_with_time(self, load_t, load, feature_t, feature, feature_name):
        """
        绘制双坐标图， 分别是时间——荷载（折线）， 时间——声发射参数（散点）
        :param load_t: 荷载时间
        :param load:  荷载
        :param feature_t:  信号参数时间
        :param feature:   信号参数
        :param feature_name: 声发射参数名称 + '/单位'
        :return:
        """
        fig = plt.figure(figsize=(7, 6))
        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(load_t, load, c='r', ms=7, linewidth=3)
        ax1.set_ylabel('荷载/kN', fontsize=20)
        ax1.set_xlabel('时间/s', fontsize=20)
        ax1.set_xlim(0, )
        ax1.set_ylim(0, )
        plt.xticks(fontsize=20, family='Times New Roman')
        plt.yticks(fontsize=20, family='Times New Roman')

        ax2 = ax1.twinx()
        # l2 = ax2.plot(feature_t, feature, c='k', ms=7, linewidth=3)
        l2 = ax2.scatter(feature_t, feature, marker='.', c='k', s=25)
        ax2.set_ylabel(feature_name, fontsize=20)
        ax2.set_ylim(0, )
        ax2.tick_params(axis='y', size=7)
        plt.xticks(fontsize=20, family='Times New Roman')
        plt.yticks(fontsize=20, family='Times New Roman')

        s = ""
        for i in feature_name:
            if i != '/':
                s += i
            else:
                break
        plt.legend(handles=[l1, l2], labels=['荷载', s],
                   loc='upper left', fontsize=20, frameon=False)  # 增加图例

        # plt.savefig('G:\\声发射试验\\pjx-节段胶拼压弯AE试验\\图表\M1\\1\\' + s + '.png', dpi=900)
        plt.show()

    def draw_load_sub_count_feature_with_time(self, load_t, load, feature_t, features, feature_name):
        """
        绘制双坐标图， 分别是时间——荷载（折线）， 时间——累积参数值（散点）
        :param load_t: 荷载时间
        :param load:  荷载
        :param feature_t:  四个通道信号参数时间 [t1, t2, t3, t4]
        :param feature:   四个通道信号参数  [f1, f2, f3, f4]
        :param feature_name: 声发射参数名称 + '/单位'
        :return:
        """
        fig = plt.figure(figsize=(8, 7))
        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(load_t, load, c='r', ms=7, linewidth=3)
        ax1.set_ylabel('荷载/kN', fontsize=25)
        ax1.set_xlabel('时间/s', fontsize=25)
        ax1.set_xlim(0, )
        ax1.set_ylim(0, )

        plt.xticks(fontsize=25, family='Times New Roman')
        plt.yticks(fontsize=25, family='Times New Roman')

        t1, t2, t3, t4 = feature_t
        f1, f2, f3, f4 = features
        print(len(t1))
        # f1 = np.array(range(1, len(t1) + 1))/1000
        # f2 = np.array(range(1, len(t2) + 1))/1000
        # f3 = np.array(range(1, len(t3) + 1))/1000
        # f4 = np.array(range(1, len(t4) + 1))/1000
        # print(f1)

        ax2 = ax1.twinx()
        l2, = ax2.plot(t1, f1.cumsum(), c='k', label='1#传感器', linewidth=3)
        l3, = ax2.plot(t2, f2.cumsum(), c='b', label='2#传感器', linewidth=3)
        l4, = ax2.plot(t3, f3.cumsum(), c='m', label='3#传感器', linewidth=3)
        l5, = ax2.plot(t4, f4.cumsum(), c='c', label='4#传感器', linewidth=3)
        ax1.plot([1980, 1980], [0, 250], c='k', ms=7, linewidth=3, linestyle='--')
        ax1.plot([530, 530], [0, 150], c='k', ms=7, linewidth=3, linestyle='--')

        # l2, = ax2.plot(t1, f1, c='k', label='1#传感器', linewidth=3)
        # l3, = ax2.plot(t2, f2, c='b', label='2#传感器', linewidth=3)
        # l4, = ax2.plot(t3, f3, c='m', label='3#传感器', linewidth=3)
        # l5, = ax2.plot(t4, f4, c='c', label='4#传感器', linewidth=3)

        ax2.set_ylabel('累积能量/10^6aJ', fontsize=25)
        ax2.set_ylim(0, )
        ax2.tick_params(axis='y', size=7)
        plt.xticks(fontsize=25, family='Times New Roman')
        plt.yticks(fontsize=25, family='Times New Roman')
        plt.legend(handles=[l1, l2, l3, l4, l5], labels=['荷载', '1#传感器', '2#传感器', '3#传感器', '4#传感器'],
                   loc='upper left', fontsize=20, frameon=False)  # 增加图例

        # plt.savefig('G:\\声发射试验\\pjx-节段胶拼压弯AE试验\\图表\M1\\1\\' + s + '.png', dpi=900)
        plt.show()

    def draw_feature_distribution(self, feature, feature_name, measure, start):
        """
        绘制特征参数分布直方图
        :param feature:
        :param feature_name: 声发射参数名称 + '/单位'
        :param measure: 直方图的划分尺度
        :param start: 直方图的起点
        :return:
        """
        num = len(feature)
        max_v = max(feature)
        distribution_dict = {}
        start_p = start
        while start_p <= max_v:
            distribution_dict.setdefault(start_p+measure/2, 0)
            start_p += measure
        for i in feature:
            for j in distribution_dict.keys():
                if j-measure/2 <= i < j+measure/2:
                    distribution_dict[j] += 1

        fig = plt.figure(figsize=(3, 6))

        ax = fig.add_subplot(111)
        _width = measure
        print(list(distribution_dict.keys()))
        print(list(distribution_dict.values()))
        ax.bar(list(distribution_dict.keys()), np.array(list(distribution_dict.values()))/num, color='r', edgecolor='k', width=_width)
        plt.xticks(fontsize=15, family='Times New Roman')
        plt.yticks(fontsize=15, family='Times New Roman')
        ax.set_ylabel('比例', fontsize=15)
        ax.set_xlabel(feature_name, fontsize=15)
        ax.set_xlim(start, 80)
        ax.tick_params(axis='both', length=8, direction='in')
        # plt.savefig(r'C:\Users\Administrator\Desktop\图片\图片-中文\试件2通道1柱状图.png', dpi=900)
        plt.show()
        
    def draw_feature_distribution_using_line(self, feature0, feature1, feature_name, measure, start):
        """
        绘制两种模式信号特征参数分布曲线图
        :param feature0, feature1:
        :param feature_name: 声发射参数名称 + '/单位'
        :param measure: 直方图的划分尺度
        :param start: 直方图的起点
        :return:
        """
        num1 = len(feature0)
        max_v = max(feature0)
        distribution_dict0 = {}
        start_p = start
        while start_p <= max_v:
            distribution_dict0.setdefault(start_p+measure/2, 0)
            start_p += measure
        for i in feature0:
            for j in distribution_dict0.keys():
                if j-measure/2 <= i < j+measure/2:
                    distribution_dict0[j] += 1

        num2 = len(feature1)
        max_v = max(feature1)
        distribution_dict1 = {}
        start_p = start
        while start_p <= max_v:
            distribution_dict1.setdefault(start_p + measure / 2, 0)
            start_p += measure
        for i in feature1:
            for j in distribution_dict1.keys():
                if j - measure / 2 <= i < j + measure / 2:
                    distribution_dict1[j] += 1

        fig = plt.figure(figsize=(7, 3))

        ax = fig.add_subplot(111)
        _width = measure
        print(list(distribution_dict0.keys()))
        print(list(distribution_dict0.values()))
        ax.plot(list(distribution_dict0.keys()), np.array(list(distribution_dict0.values()))/num1, color='r',
                linewidth=2, label='第一种模式信号')
        ax.plot(list(distribution_dict1.keys()), np.array(list(distribution_dict1.values())) / num2, color='b',
                linewidth=2, label='第二种模式信号')
        # ax.bar(list(distribution_dict0.keys()), np.array(list(distribution_dict0.values())) / num1, color='r',
        #        edgecolor='k', width=_width, label='第一种模式信号')
        # ax.bar(list(distribution_dict1.keys()), np.array(list(distribution_dict1.values())) / num2, color='b',
        #        edgecolor='k', width=_width, label='第二种模式信号')
        # ax.bar(list(distribution_dict0.keys()), np.array(list(distribution_dict0.values())) / num1, label='第一种模式信号')
        # ax.bar(list(distribution_dict1.keys()), np.array(list(distribution_dict1.values())) / num2,  label='第二种模式信号')

        plt.xticks(fontsize=16, family='Times New Roman')
        plt.yticks(fontsize=16, family='Times New Roman')
        ax.set_ylabel('比例', fontsize=16)
        ax.set_xlabel(feature_name, fontsize=16)
        ax.set_xlim(start, )
        ax.set_ylim(0, )
        ax.tick_params(axis='both', length=8, direction='in')
        plt.legend(loc='upper left', fontsize=16, frameon=False)
        # plt.savefig(r'C:\Users\Administrator\Desktop\图片\图片-中文\试件2通道1柱状图.png', dpi=900)
        plt.show()

    def draw_feature1_by_feature2(self, feature1, feature1_name, feature2, feature2_name):
        """
        两特征参数的相关分析散点图
        :param feature1: x轴
        :param feature1_name: 声发射参数名称 + '/单位'
        :param feature2: y轴
        :param feature2_name: 声发射参数名称 + '/单位'
        :return:
        """
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.scatter(feature1, feature2, marker='.', color='k', s=17)
        ax.set_xlabel(feature1_name, fontsize=18)
        ax.set_ylabel(feature2_name, fontsize=18)
        ax.set_xlim(0, )
        ax.set_ylim(0, )
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

    def draw_count_of_two_signal(self, labels, x_time):
        """
        绘制两种信号累积个数变化图
        :param labels: 标签序列
        :param times:  时间序列
        :return:
        """
        count_0, count_1, count_01 = [], [], []
        for i in range(len(labels)):
            count_1.append(sum(labels[:i+1]))
            count_0.append(len(labels[:i+1])-sum(labels[:i+1]))
            count_01.append(len(labels[:i+1]))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        l1, = ax.plot(x_time, count_0, 'o-', c='r', ms=5, label='0', linewidth=2)
        l2, = ax.plot(x_time, count_1, '*-', c='k', ms=5, label='1', linewidth=2)
        l3, = ax.plot(x_time, count_01, '>-', c='b', ms=5, label='01', linewidth=2)

        ax.set_xlabel('时间/s', fontsize=18)
        ax.set_ylabel('累积事件数', fontsize=18)
        plt.xticks(fontsize=15, family='Times New Roman')
        plt.yticks(fontsize=15, family='Times New Roman')
        ax.set_xlim(0, )
        ax.set_ylim(0, )
        ax.tick_params(axis='y', size=7)
        plt.legend(handles=[l1, l2, l3],
                   labels=['第一种模式信号', '第二种模式信号', '信号'],
                   loc='upper left', fontsize=15, frameon=False)  # 增加图例

        # plt.savefig(r'C:\Users\Administrator\Desktop\图片\图片-中文\试件4通道1折线图.png', dpi=900)
        plt.show()

    def draw_oscillograph_of_signal(self, file):
        """
        绘制波形图
        :param file:  .csv
        :return:
        """
        with open(file, 'r') as fl:
            reader = csv.reader(fl)
            y = []
            count = 0
            for row in reader:
                count += 1
                if count > 12:
                    y.append(float(row[0]))

            plt.plot(range(len(y)), y, color='b', lw=1)
            plt.xlim(0, 1024)
            plt.xlabel('时间/$μs$', fontsize=18)
            plt.ylabel('电压/mV', fontsize=18)
            plt.xticks(fontsize=15, family='Times New Roman')
            plt.yticks(fontsize=15, family='Times New Roman')
            plt.show()

    def draw_histogram(self, x, y, xlabel, ylabel, name_list):
        """
        绘制柱状图
        :param file:
        :return:
        """
        _width = 0.2
        fig, ax = plt.subplots()
        rects1 = ax.bar(x, y, _width, color='w', edgecolor='b', hatch="x")
        ax.set_xticks(np.array(x) + _width / 2)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xticklabels(name_list)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.legend([rects1[0]], ['第一种模式信号'], frameon=False, fontsize=15)
        # plt.savefig(r'C:\Users\Administrator\Desktop\图片\图片-中文\试件2通道1柱状图.png', dpi=900)
        plt.show()

    def draw_density_of_signal(self, labels, times, window_size):
        """
        绘制两种信号的密度变化————以信号个数为窗口
        :param labels:
        :param times:
        :param window_size: 窗口大小
        :return:
        """
        density0, density1, time_list = [], [], []
        times = list(times)
        for i in range(0, len(labels)-window_size, window_size):
            d = sum(labels[i:i + window_size]) / window_size
            density1.append(d)
            density0.append(1-d)
            time_list.append(np.mean(times[i:i+window_size]))

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.scatter(time_list, density0, color='r', s=20, marker='o', label='第一种模式信号')
        ax.scatter(time_list, density1, color='k', s=20, marker='<', label='第二种模式信号')
        ax.set_xlabel("时间", fontsize=18)
        ax.set_ylabel("信号密度", fontsize=18)
        plt.xticks(fontsize=18, family='Times New Roman')
        plt.yticks(fontsize=18, family='Times New Roman')
        plt.legend(loc='upper left', fontsize=18, frameon=False)
        ax.set_ylim(0, )
        ax.set_xlim(0, )
        ax.tick_params(axis='y', size=7)
        # plt.savefig(r'C:\Users\Administrator\Desktop\图片\图片-中文\m2-30窗口密度.png', dpi=900)
        plt.show()

    def draw_density_of_signal_by_time(self, labels, times, time_size=100):
        """
        绘制两种信号的密度变化————以一段时间为窗口
        :param labels:
        :param times:
        :param time_size:
        :return:
        """
        density0, density1 = [], []
        time_list = [0] + [i for i in range(500, 2300, time_size)] + [2287]
        times = list(times)
        for i in range(1, len(time_list)):
            time_start = time_list[i-1]
            time_end = time_list[i]
            d0, d1 = 0, 0
            for j in range(len(times)):
                if time_start < times[j] <= time_end:
                    if labels[j] == 0:
                        d0 += 1
                    else:
                        d1 += 1
            density0.append(d0/(d0+d1))
            density1.append(1-d0/(d0+d1))

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.scatter(time_list[1:], density0, color='r', s=20, marker='o', label='第一种模式信号')
        ax.scatter(time_list[1:], density1, color='k', s=20, marker='<', label='第二种模式信号')
        ax.set_xlabel("时间/s", fontsize=18)
        ax.set_ylabel("信号密度", fontsize=18)
        plt.xticks(fontsize=15, family='Times New Roman')
        plt.yticks(fontsize=15, family='Times New Roman')
        plt.legend(loc='upper left', fontsize=15, frameon=False)
        ax.set_ylim(0, )
        ax.set_xlim(0, )
        ax.tick_params(axis='y', size=7)
        # plt.savefig(r'C:\Users\Administrator\Desktop\图片\图片-中文\m2-30窗口密度.png', dpi=900)
        plt.show()

    def draw_two_signals_by_one_feature(self, labels, times, feature, feature_name, unit):
        """
        绘制某一特征下两种信号的散点图
        :param labels:  信号分类标签
        :param times:   时间序列
        :param feature: 特征序列
        :param unit: 单位
        :return:
        """
        signal0_f, signal1_f = [], []
        time0_f, time1_f = [], []
        for i in range(len(labels)):
            if not labels[i]:
                signal0_f.append(feature[i])
                time0_f.append(times[i])
            else:
                signal1_f.append(feature[i])
                time1_f.append(times[i])

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.scatter(time0_f, signal0_f, color='b', s=18, marker='o', label='第一种模式信号')
        ax.scatter(time1_f, signal1_f, color='r', s=18, marker='<', label='第二种模式信号')
        ax.plot([550, 550], [0., 0.6], '--', c='k', linewidth=4)
        ax.plot([1980, 1980], [0., 0.8], '--', c='k', linewidth=4)
        ax.set_xlabel("时间/s", fontsize=20)
        ax.set_ylabel(feature_name+unit, fontsize=20)
        plt.xticks(fontsize=20, family='Times New Roman')
        plt.yticks(fontsize=20, family='Times New Roman')
        plt.legend(loc='upper left', fontsize=18, frameon=False)
        ax.set_ylim(0, )
        ax.set_xlim(0, )
        ax.tick_params(axis='both', size=7)
        # plt.savefig(r'C:\Users\Administrator\Desktop\图片\图片-中文\m2-30窗口密度.png', dpi=900)
        plt.show()

    def draw_one_signal_by_one_feature(self, feature, feature_name, unit, times):
        """
        绘制某一类信号的某个特征时程散点图
        :param feature: 特征序列
        :param feature_name: 特征名称
        :param unit: 单位
        :param times: 时间序列
        :return:
        """
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.scatter(times, feature, color='r', s=22, marker='o')
        ax.set_xlabel('时间/s', fontsize=18)
        ax.set_ylabel(feature_name+unit, fontsize=18)
        plt.xticks(fontsize=18, family='Times New Roman')
        plt.yticks(fontsize=18, family='Times New Roman')
        plt.legend(loc='upper left', fontsize=17, frameon=False)
        ax.set_ylim(0, )
        ax.set_xlim(0, )
        ax.tick_params(axis='y', size=7)
        # plt.savefig(r'C:\Users\Administrator\Desktop\图片\图片-中文\m2-30窗口密度.png', dpi=900)
        plt.show()

    def draw_two_signals_by_two_feature(self, labels, feature1, feature1_name, unit1, feature2, feature2_name, unit2):
        """
        绘制两种特征下两种信号的散点图
        :param labels: 标签
        :param feature1: 特征1
        :param feature1_name: 特征1的名称
        :param unit1: 单位1
        :param feature2: 特征2
        :param feature2_name: 特征2的名称
        :param unit2: 单位2
        :return:
        """
        signal0_f1, signal0_f2 = [], []
        signal1_f1, signal1_f2 = [], []
        print(len(labels), len(feature1), len(feature2))
        for i in range(len(labels)):
            if labels[i] == 0:
                signal0_f1.append(feature1[i])
                signal0_f2.append(feature2[i])
            else:
                signal1_f1.append(feature1[i])
                signal1_f2.append(feature2[i])

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.scatter(signal0_f1, signal0_f2, color='r', s=26, marker='o', label='第一种模式信号')
        ax.scatter(signal1_f1, signal1_f2, color='k', s=26, marker='<', label='第二种模式信号')
        ax.set_xlabel(feature1_name+unit1, fontsize=18)
        ax.set_ylabel(feature2_name+unit2, fontsize=18)
        plt.xticks(fontsize=18, family='Times New Roman')
        plt.yticks(fontsize=18, family='Times New Roman')
        plt.legend(loc='lower right', fontsize=18, frameon=False)
        ax.set_ylim(0, )
        ax.set_xlim(0, )
        ax.tick_params(axis='both', length=5, direction='in')
        # plt.savefig(r'C:\Users\Administrator\Desktop\图片\图片-中文\m2-30窗口密度.png', dpi=900)
        plt.show()

    def draw_clustering_index(self, indexs, index_name):
        """
        绘制聚类评价指标折线图
        :param indexs: 聚类评价指标
        :param index_name: 聚类评价指标名称
        :return:
        """
        center_number_list = list(range(2, len(indexs)+2))
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.plot(center_number_list, indexs, '>-', c='r', ms=6, linewidth=2)
        ax.set_xlabel("聚类中心数目", fontsize=20)
        ax.set_ylabel(index_name, fontsize=20)
        plt.xticks(fontsize=17, family='Times New Roman')
        plt.yticks(fontsize=17, family='Times New Roman')
        # ax.set_ylim()
        ax.set_xlim(2, )
        ax.tick_params(axis='y', size=7)
        plt.savefig('C:\\Users\\Administrator\\Desktop\\'+index_name+'.png', dpi=900)

    def draw_scatter_after_pca(self, dimension1, dimension2, labels):
        """
        绘制pca降维后（降到二维）的散点图，先降维再输入参数
        :param dimension1: 第一维数据
        :param dimension2: 第二维数据
        :param labels:  信号分类标签
        :return:
        """
        signal0_x, signal0_y, signal1_x, signal1_y = [], [], [], []
        for i in range(len(labels)):
            if labels[i]:
                signal1_x.append(dimension1[i])
                signal1_y.append(dimension2[i])
            else:
                signal0_x.append(dimension1[i])
                signal0_y.append(dimension2[i])

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.scatter(signal0_x, signal0_y, color='r', s=25, marker='o', label='第一种模式信号')
        ax.scatter(signal1_x, signal1_y, color='k', s=25, marker='<', label='第二种模式信号')
        ax.set_xlabel("第一维", fontsize=19)
        ax.set_ylabel("第二维", fontsize=19)
        plt.xticks(fontsize=17, family='Times New Roman')
        plt.yticks(fontsize=17, family='Times New Roman')
        plt.legend(loc='upper right', fontsize=18, frameon=False)
        # ax.set_ylim()
        # ax.set_xlim()
        ax.tick_params(axis='y', size=7)
        plt.show()

    def draw_loss_and_validation_by_iteration(self, loss_array, validation_array):
        """
        绘制损失和验证集准确率曲线
        :param loss_array:
        :param validation_array:
        :return:
        """
        iterations = [i for i in range(1, len(loss_array)+1)]
        fig = plt.figure(figsize=(8, 7))
        ax1 = fig.add_subplot(111)
        # ax1.spines['right'].set_visible(False)  # 去掉边框
        # ax1.spines['top'].set_visible(False)
        l1, = ax1.plot(iterations, loss_array, c='r', ms=7, linewidth=2.5)
        l2, = ax1.plot(iterations, validation_array, c='k', ms=7, linewidth=2.5)
        ax1.set_xlabel('迭代次数', fontsize=22)
        ax1.set_ylabel('损失', fontsize=22)
        ax1.set_xlim(0, )
        ax1.tick_params(axis='both', length=4, direction='in')
        plt.xticks(fontsize=22, family='Times New Roman')
        plt.yticks(fontsize=22, family='Times New Roman')

        plt.legend(handles=[l1, l2], labels=['Adam', 'SGD'], loc='upper right', fontsize=22, frameon=False)
        plt.show()

    def draw_accuracy_with_shape_of_bar(self, label_list, accuracy0, accuracy1):
        """
        绘制准确率的条形图
        :param x:
        :param accuracy0:
        :param accuracy1:
        :return:
        """
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111)
        x = range(len(accuracy0))
        rects1 = ax.bar(left=x, height=accuracy0, width=0.2, color='r', edgecolor='k', label="第一种模式信号")
        rects2 = ax.bar(left=[i + 0.2 for i in x], height=accuracy1, width=0.2, color='b', edgecolor='k', label="第二种模式信号")
        ax.set_xlabel("隐含层节点数量", fontsize=21)
        ax.set_ylabel("准确率", fontsize=21)
        ax.set_ylim(0.5, 1.2)
        plt.xticks(fontsize=22, family='Times New Roman')
        plt.yticks(fontsize=22, family='Times New Roman')
        plt.xticks([index + 0.2 for index in x], label_list)
        plt.legend(loc='upper right', fontsize=21, frameon=False)
        plt.show()

    def draw_accuracy_with_sample_quantity(self, accuracy_of_b1, accuracy_of_b2, accuracy_of_b3, accuracy_of_b4, s=1):
        """
        绘制随信号事件数的增加准确率的变化曲线
        :param accuracy_of_b2:
        :param accuracy_of_b3:
        :return:
        """
        fig = plt.figure(figsize=(9, 6))
        ax1 = fig.add_subplot(111)
        ax1.spines['right'].set_visible(False)  # 去掉边框
        ax1.spines['top'].set_visible(False)

        z2 = np.polyfit(np.array(list(range(1, len(accuracy_of_b2) + 1))), np.array(accuracy_of_b2), 8)
        p2 = np.poly1d(z2)
        y2 = p2(np.array(list(range(1, len(accuracy_of_b2) + 1))))
        l2, = ax1.plot(list(range(1, len(accuracy_of_b2) + 1)), y2, c='r', linewidth=2.5)
        ax1.scatter(list(range(1, len(accuracy_of_b2) + 1)), accuracy_of_b2, marker='+', c='r')

        z3 = np.polyfit(np.array(list(range(1, len(accuracy_of_b3) + 1))), np.array(accuracy_of_b3), 8)
        p3 = np.poly1d(z3)
        y3 = p3(np.array(list(range(1, len(accuracy_of_b3) + 1))))
        l3, = ax1.plot(list(range(1, len(accuracy_of_b3) + 1)), y3, c='k', linewidth=2.5)
        ax1.scatter(list(range(1, len(accuracy_of_b3) + 1)), accuracy_of_b3, marker='o', c='k')

        z1 = np.polyfit(np.array(list(range(1, len(accuracy_of_b1) + 1))), np.array(accuracy_of_b1), 8)
        p1 = np.poly1d(z1)
        y1 = p1(np.array(list(range(1, len(accuracy_of_b1) + 1))))
        l1, = ax1.plot(list(range(1, len(accuracy_of_b1) + 1)), y1, c='b', linewidth=2.5)
        ax1.scatter(list(range(1, len(accuracy_of_b1) + 1)), accuracy_of_b1, marker='>', c='b')

        z4 = np.polyfit(np.array(list(range(1, len(accuracy_of_b4) + 1))), np.array(accuracy_of_b4), 8)
        p4 = np.poly1d(z4)
        y4 = p4(np.array(list(range(1, len(accuracy_of_b4) + 1))))
        l4, = ax1.plot(list(range(1, len(accuracy_of_b4) + 1)), y4, c='g', linewidth=2.5)
        ax1.scatter(list(range(1, len(accuracy_of_b4) + 1)), accuracy_of_b4, marker='*', c='g')

        # l3, = ax1.plot(list(range(1, len(accuracy_of_b3) + 1)), accuracy_of_b3, c='k', linewidth=2.5)
        # l1, = ax1.plot(list(range(1, len(accuracy_of_b1) + 1)), accuracy_of_b1, c='b', linewidth=2.5)
        # l4, = ax1.plot(list(range(1, len(accuracy_of_b4) + 1)), accuracy_of_b4, c='g', linewidth=2.5)

        ax1.set_xlabel('信号事件个数', fontsize=21)
        ax1.set_ylabel('准确率', fontsize=21)
        ax1.set_xlim(0, len(accuracy_of_b4)/s)
        ax1.set_ylim(0.65, 1)
        ax1.tick_params(axis='both', length=6, direction='out')
        plt.xticks(fontsize=22, family='Times New Roman')
        plt.yticks(fontsize=22, family='Times New Roman')
        # plt.xticks([5, 10, 15], fontsize=25, family='Times New Roman')

        plt.legend(handles=[l1, l2, l3, l4], labels=['第一阶段', '非第一阶段', '第二阶段', '非第二阶段'], loc='lower center',
                   fontsize=18, frameon=False, ncol=2)
        plt.show()

    def draw_accuracy_with_node(self, node_array, accuracy_of_tensor1, accuracy_of_tensor2, accuracy_of_tensor3,
                                accuracy_of_tensor4):
        """
        绘制测试集准确率，线图
        :param general_accuracy: 综合准确率
        :param accuracy_of_class0:
        :param accuracy_of_class1:
        :param accuracy_of_class2:
        :return:
        """
        fig = plt.figure(figsize=(7, 6))
        ax1 = fig.add_subplot(111)
        ax1.spines['right'].set_visible(False)  # 去掉边框
        ax1.spines['top'].set_visible(False)
        l1, = ax1.plot(node_array, accuracy_of_tensor1, '>-', c='r', ms=10, linewidth=1.5)
        l2, = ax1.plot(node_array, accuracy_of_tensor2, 'o-', c='k', ms=10, linewidth=1.5)
        l3, = ax1.plot(node_array, accuracy_of_tensor3, '+-', c='b', ms=12, linewidth=1.5)
        l4, = ax1.plot(node_array, accuracy_of_tensor4, '*-', c='g', ms=12, linewidth=1.5)
        ax1.set_xlabel('隐含层节点数', fontsize=25)
        ax1.set_ylabel('准确率', fontsize=25)
        ax1.set_xlim(5, 25)
        ax1.set_ylim(0.6, 1)
        ax1.tick_params(axis='both', length=8, direction='out')
        plt.xticks([5, 10, 15, 20, 25], fontsize=25, family='Times New Roman')
        plt.yticks(fontsize=25, family='Times New Roman')

        plt.legend(handles=[l1, l2, l3, l4], labels=['第一阶段', '非第一阶段', '第二阶段', '非第二阶段'], loc='upper right',
                   fontsize=20, frameon=False)
        plt.show()

    def draw_scatter_of_three_class_after_pca(self, data, labels):
        """

        :param data:
        :param labels:
        :return:
        """

        fig = plt.figure(figsize=(7, 6))
        ax1 = fig.add_subplot(111)
        # ax1.spines['right'].set_visible(False)  # 去掉边框
        # ax1.spines['top'].set_visible(False)
        x0, x1, x2 = [], [], []
        y0, y1, y2 = [], [], []
        for label_index in range(len(labels)):
            if labels[label_index] == 0:
                x0.append(data[label_index][0])
                y0.append(data[label_index][1])

            elif labels[label_index] == 1:
                x1.append(data[label_index][0])
                y1.append(data[label_index][1])

            else:
                x2.append(data[label_index][0])
                y2.append(data[label_index][1])

        ax1.scatter(x0, y0, color='k', s=22, marker='o', label='第一类')
        # ax1.scatter(x1, y1, color='y', s=22, marker='*', label='第二类')
        # ax1.scatter(x2, y2, color='b', s=22, marker='>', label='第三类')

        ax1.set_xlabel('', fontsize=25)
        ax1.set_ylabel('', fontsize=25)
        ax1.tick_params(axis='both', length=8, direction='in')
        plt.xticks(fontsize=25, family='Times New Roman')
        plt.yticks(fontsize=25, family='Times New Roman')

        plt.legend(loc='upper right', fontsize=21, frameon=False)
        plt.show()

    def draw_average_frequency(self, frequency_array0, frequency_array1):
        """
        绘制平均频率曲线图
        :param frequency_array:
        :return:
        """
        fig = plt.figure(figsize=(9, 4))
        ax = fig.add_subplot(111)
        ax.plot(list(range(1, 201)), frequency_array0, c='b', linewidth=1.5, label='阶段一')
        ax.plot(list(range(1, 201)), frequency_array1, c='r', linewidth=1.5, label='阶段二')
        # ax.plot(list(range(1, 201)), frequency_array2, c='k', linewidth=1.5, label='阶段三')
        ax.set_xlabel('频率/kHz', fontsize=19)
        ax.set_ylabel('幅度', fontsize=19)
        plt.xticks(fontsize=19, family='Times New Roman')
        plt.yticks(fontsize=19, family='Times New Roman')
        ax.set_xlim(0, 200)
        ax.set_ylim(0, )
        ax.tick_params(axis='both', length=6, direction='in')
        plt.legend(loc='upper right', fontsize=19, frameon=False)
        plt.grid(linestyle='--', linewidth=1.5)

        plt.show()


# file1 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\M3荷载.xlsx"
# data_1 = pd.read_excel(file1, sheetname=2)
# # print(data_1)
# load = data_1.iloc[:, 0]
# defo = data_1.iloc[:, 2:5]
# Draw().draw_deformation_of_load(defo, load)

# data_file = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\t滤波后-7200.csv"
# with open(data_file, 'r') as fl:
#     data = pd.read_csv(fl)
# data = data.loc[data['通道']==4, :]
# data = data.loc[data['持续时间']<=90000, :]
# print(data.loc[data['持续时间']>=80000, :])
# data = data.loc[data['TIME OF TEST'] > 700, :]
# data = data.iloc[:, 2:]
# Draw().draw_feature_distribution(f, 'ASL/dB', 2, 0)

# data_file = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件3-0324\正式加载\t滤波后-7200.csv"
# with open(data_file, 'r') as fl:
#     data = pd.read_csv(fl)
# file_label = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件3-0324\正式加载\t_standardscale_labels.json"
# with open(file_label, 'r') as fo:
#     data_labels = json.load(fo)
# labels = data_labels['FCM1'] + data_labels['FCM2'] + data_labels['FCM3'] + data_labels['FCM4']
# data['label'] = labels
# f0 = data.loc[data['label'] == 0, :]['上升时间']/1000
# f1 = data.loc[data['label'] == 1, :]['上升时间']/1000
#
# Draw().draw_feature_distribution_using_line(f0, f1, 'ASL/dB', 1, 0)


# data_file2 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件2-0324\正式加载\t滤波后-7200.csv"
# initial_x, y = four_tensor_data(data_file2, 550, 2200)
# # initial_x, y = one_tensor_data(data_file2, 550, 2200, 4)
# x = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(initial_x))
# Draw().draw_scatter_of_three_class_after_pca(x, y)


# pd.set_option('display.width', 1000)
# a0 = [1.0]*5
# a1 = [0.99, 0.99, 0.99, 0.98, 0.98]
# x = ['5', '10', '15', '20', '22']
# Draw().draw_accuracy_with_shape_of_bar(x, a0, a1)
# nodes = [5, 7, 10, 13, 15, 17, 20, 23, 25]
# accuracy_of_tensor1 = [0.62, 0.60, 0.63, 0.63, 0.62, 0.63, 0.65, 0.61, 0.62]
# accuracy_of_tensor2 = [0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74]
# accuracy_of_tensor3 = [0.70, 0.71, 0.71, 0.70, 0.70, 0.71, 0.71, 0.70, 0.70]
# accuracy_of_tensor4 = [0.74, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.74]
# nodes = [5, 10, 15, 20, 25]
# accuracy_of_tensor1 = [0.75, 0.74, 0.73, 0.74, 0.74]
# accuracy_of_tensor2 = [0.79, 0.8, 0.8, 0.8, 0.8]
# accuracy_of_tensor3 = [0.83, 0.85, 0.84, 0.83, 0.84]
# accuracy_of_tensor4 = [0.81, 0.8, 0.8, 0.79, 0.81]
# Draw().draw_accuracy_with_node(nodes, accuracy_of_tensor1, accuracy_of_tensor2,
#                                accuracy_of_tensor3, accuracy_of_tensor4)


# b2 = [0.71, 0.81, 0.88, 0.9, 0.93, 0.97, 0.97, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 1.0]
# b3 = [0.71, 0.77, 0.85, 0.87, 0.9, 0.92, 0.94, 0.95, 0.94, 0.97, 0.97, 0.97, 0.97, 0.98, 0.96, 0.98]
# Draw().draw_accuracy_with_sample_quantity(b2, b3)


file1_label = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\standardscale_labels.json"
file1_data = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\t滤波后-7200.csv"
file2_label = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件2-0324\正式加载\standardscale_labels.json"
file2_data = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件2-0324\正式加载\t滤波后-7200.csv"
file3_label = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件3-0324\正式加载\standardscale_labels.json"
file3_data = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件3-0324\正式加载\t滤波后-7200.csv"

tongdao_index = 1
with open(file1_label, 'r') as fo:
    data1_labels = json.load(fo)
    labels_1 = data1_labels['FCM'+str(tongdao_index)]
with open(file1_data, 'r') as fl:
    data1 = pd.read_csv(fl)
data_1_1 = data1.loc[data1['通道'] == tongdao_index, :]

with open(file2_label, 'r') as fo:
    data2_labels = json.load(fo)
    labels_2 = data2_labels['FCM'+str(tongdao_index)]
with open(file2_data, 'r') as fl:
    data2 = pd.read_csv(fl)
data_2_1 = data2.loc[data2['通道'] == tongdao_index, :]

with open(file3_label, 'r') as fo:
    data3_labels = json.load(fo)
    labels_3 = data3_labels['FCM'+str(tongdao_index)]
with open(file3_data, 'r') as fl:
    data3 = pd.read_csv(fl)
data_3_1 = data3.loc[data3['通道'] == tongdao_index, :]


# 绘制两种特征下两种信号的散点图
feature1_n = '上升时间'
feature1_unit = '/10^4$μs$'
feature1 = list(data_1_1[feature1_n]/10000)+list(data_2_1[feature1_n]/10000)+list(data_3_1[feature1_n]/10000)
print(feature1)
feature2_n = '计数'
feature2_unit = '/10^3'
feature2 = list(data_1_1[feature2_n]/1000)+list(data_2_1[feature2_n]/1000)+list(data_3_1[feature2_n]/1000)

Draw().draw_two_signals_by_two_feature(labels_1+labels_2+labels_3, feature1, feature1_n, feature1_unit, feature2,
                                       feature2_n, '振铃'+feature2_unit)


# 绘制信号参数累积变化时程图
# for j in range(1, 4):
#     file1 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\M%d荷载.xlsx"%j
#     file2 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件%d-0324\正式加载\t滤波后-7200.csv"%j
#
#     data_1 = pd.read_excel(file1, sheetname=0)
#     with open(file2, 'r') as fl:
#         data_2 = pd.read_csv(fl)
#
#     load_t = data_1['时间（s）']
#     load = data_1['荷载(kN)']
#     f_t, f = [], []
#     for sensor_index in range(1, 5):
#         f_t.append(data_2.loc[data_2['通道'] == sensor_index, :]['SSSSSSSS.mmmuuun'])
#         f.append(data_2.loc[data_2['通道'] == sensor_index, :]['能量']/1000000)
#     f_n = '累积能量/10^6aJ'
#     Draw().draw_load_sub_count_feature_with_time(load_t, load, f_t, f, f_n)


# 绘制某一特征下两种信号的散点图
# file_label = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件3-0324\正式加载\t_standardscale_labels.json"
# file_data = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件3-0324\正式加载\t滤波后-7200.csv"
#
# with open(file_label, 'r') as fo:
#     data_labels = json.load(fo)
#
# with open(file_data, 'r') as fl:
#     data = pd.read_csv(fl)
#
# feature_n = '振铃计数'
# feature_unit = '/10^3'
# data = data.loc[data['通道']==1, :]
# labels = data_labels['FCM1']
# labels = data_labels['FCM1'] + data_labels['FCM2'] + data_labels['FCM3'] + data_labels['FCM4']
# X = list(data[feature_n[2:]]/10000)
# time_array = list(data['SSSSSSSS.mmmuuun'])

    # x_0, y_0, x_1, y_1 = [], [], [], []
    # for label, i in zip(labels, range(len(labels))):
    #     if not label:
    #         x_0.append(time_array[i])
    #         y_0.append(X[i])
    #     else:
    #         x_1.append(time_array[i])
    #         y_1.append(X[i])

# Draw().draw_one_signal_by_one_feature(y_0, feature_n, feature_unit, x_0)
# Draw().draw_one_signal_by_one_feature(y_1, feature_n, feature_unit, x_1)
# Draw().draw_two_signals_by_one_feature(labels, time_array, X, feature_n, feature_unit)


# 绘制平均频率曲线图
# file_label = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\t_standardscale_labels.json"
# file_data = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\fft.csv"
#
# with open(file_label, 'r') as fo:
#     data_labels = json.load(fo)
# # labels = data_labels['FCM1']
# labels = data_labels['FCM1'] + data_labels['FCM2'] + data_labels['FCM3'] + data_labels['FCM4']
# with open(file_data, 'r') as fl:
#     data = pd.read_csv(fl)
#
# f = data.iloc[:, 2:]
# f = np.array(f).T
# f = pd.DataFrame(MinMaxScaler(feature_range=(0, 1)).fit_transform(f).T)
# f['label'] = labels
# f0 = f.loc[f['label']==0, :].iloc[:, :-1]
# l0 = len(f0)
# f0 = f0.apply(lambda x: x.sum(), axis=0)
# f1 = f.loc[f['label']==1, :].iloc[:, :-1]
# l1 = len(f1)
# f1 = f1.apply(lambda x: x.sum(), axis=0)
# Draw().draw_average_frequency(f0/l0, f1/l1)


# print(data)
# data = data.loc[data['CHANNEL NUMBER']==1, :]
# print(data)
# f = data.iloc[:, 2:]
# f = np.array(f).T
# f = pd.DataFrame(MinMaxScaler(feature_range=(0, 1)). fit_transform(f).T)
#
# f['TIME OF TEST'] = list(data['TIME OF TEST'])
# print(f)
# f0 = f.loc[f['TIME OF TEST'] <= 530, :].iloc[:, :-1]
# l0 = len(f0)
# print(l0)
# f0 = f0.apply(lambda x: x.sum(), axis=0)
#
# f1 = f.loc[f['TIME OF TEST'] > 530, :]
# f1 = f1.loc[f['TIME OF TEST'] <= 1980, :].iloc[:, :-1]
# l1 = len(f1)
# print(l1)
# f1 = f1.apply(lambda x: x.sum(), axis=0)
#
#
# f2 = f.loc[f['TIME OF TEST'] > 1980, :].iloc[:, :-1]
# l2 = len(f2)
# print(l2)
# f2 = f2.apply(lambda x: x.sum(), axis=0)

# Draw().draw_average_frequency(f0/l0, f1/l1, f2/l2)


# 绘制pca降维后的散点图
# X = data_1.iloc[:, 2:-1].drop(['信号强度', '初始频率', '绝对能量', '中心频率', '峰频'], axis=1)
# X_scale = StandardScaler().fit_transform(X)
# X_dimensionality_reduction = PCA(n_components=2).fit_transform(X_scale)  # pca降维
# x1 = X_dimensionality_reduction[:, 0]
# x2 = X_dimensionality_reduction[:, 1]
# Draw().draw_scatter_after_pca(x1, x2, labels)

# 绘制聚类有效性指标折线图
# index_silhouette_coefficient, index_calinski_harabaz = [], []
# for i in range(2, 9):
#     model = FCM_P(X_scale, i)
#     C, U = model.FCM()
#     labels = model.FCM_classify(C)
#     m = model.evaluation_index_silhouette_coefficient(labels)
#     k = model.evaluation_index_calinski_harabaz(labels)
#     index_silhouette_coefficient.append(m)
#     index_calinski_harabaz.append(k)
#     print("聚类中心数目:", i, " 轮廓系数:%.3f" % m, "| Calinski-Harabasz系数:%.3f" % k)
#
# print(index_silhouette_coefficient)
# print(index_calinski_harabaz)
# Draw().draw_clustering_index(indexs=index_silhouette_coefficient, index_name='轮廓系数')
# Draw().draw_clustering_index(indexs=index_calinski_harabaz, index_name='Calinski-Harabasz指标')


# for j in range(1, 4):
#     file1 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\M%d荷载.xlsx"%j
#     file2 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件%d-0324\正式加载\滤波后-7200.csv"%j
#
#     data_1 = pd.read_excel(file1, sheetname=0)
#     with open(file2, 'r') as fl:
#         data_2 = pd.read_csv(fl)
#     # print(data_1)
#     # print(data_2)
#
#     for i in range(1, 5):
#         t1 = data_1['时间（s）']
#         load = data_1['荷载(kN)']
#         t2 = data_2.loc[data_2['通道'] == i, :]['时间']
#         f1 = data_2.loc[data_2['通道'] == i, :]['计数']/1000
#         # f2 = data_2.loc[data_2['通道'] == 1, :]['能量']
#
#         # # f1_name = '上升时间/$μs$'
#         f1_name = '振铃计数/10^3'
#         # f2_name = '能量/aJ '
#
#         # f = [i/1000000 for i in f]
#         # for i in range(1, len(f)):
#         #     f[i] = f[i-1]+f[i]
#         Draw().draw_load_sub_feature_with_time(t1, load, t2, f1, f1_name)
#         # Draw().draw_feature1_by_feature2(f1, f1_name, f2, f2_name)
#         # Draw().draw_feature_distribution(f, f_name, measure=1000, start=0)

