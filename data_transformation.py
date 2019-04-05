# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


def split_data(data_x, data_y, sample0_test_proportion, sample1_test_proportion, sample2_test_proportion, random=1):
    """
    用随机的方法，按照指定的各类测试集样本比例生成训练集和测试集
    :param data_x: 样本, np.array
    :param data_y: 标签,[]
    :param sample0_test_proportion: 0类样本测试集比例
    :param sample1_test_proportion: 1类样本测试集比例
    :param sample2_test_proportion: 2类样本测试集比例
    :return:
    """
    data0_x, data1_x, data2_x = [], [], []
    data0_y, data1_y, data2_y = [], [], []
    for x, y in zip(data_x, data_y):
        if y == 0:
            data0_x.append(x)
            data0_y.append(y)
        elif y == 1:
            data1_x.append(x)
            data1_y.append(y)
        else:
            data2_x.append(x)
            data2_y.append(y)
    if random:
        x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(data0_x, data0_y, test_size=sample0_test_proportion,
                                                                    random_state=8)
        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data1_x, data1_y, test_size=sample1_test_proportion,
                                                                    random_state=8)
        x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(data2_x, data2_y, test_size=sample2_test_proportion,
                                                                    random_state=8)

        x_train = np.concatenate((x_train_0, x_train_1, x_train_2), axis=0)
        y_train = np.concatenate((y_train_0, y_train_1, y_train_2), axis=0)

        if x_test_0:
            x_test = np.concatenate((x_test_0, x_test_1, x_test_2), axis=0)
            y_test = np.concatenate((y_test_0, y_test_1, y_test_2), axis=0)
        else:
            x_test = []
            y_test = []
    else:
        x_train = np.concatenate((data0_x, data1_x, data2_x), axis=0)
        x_test = []
        y_train = np.concatenate((data0_y, data1_y, data2_y), axis=0)
        y_test = []

    return x_train, x_test, y_train, y_test


def generate_y(times, t1, t2):
    """
    根据时间序列生成标签
    :param times:
    :return:[]
    """
    data_y = []
    for time in times:
        if time <= t1:
            data_y.append(0)
        elif time <= t2:
            data_y.append(1)
        else:
            data_y.append(2)
    return data_y


def accuracy(test_labels, predict_labels):
    line1, line2 = 0, 0
    for i in test_labels:
        if i == 0:
            line1 += 1
        elif i == 1:
            line2 += 1
    line2 += line1
    accuracy_score0 = accuracy_score(test_labels[0:line1], predict_labels[0:line1])
    accuracy_score1 = accuracy_score(test_labels[line1:line2], predict_labels[line1:line2])
    accuracy_score2 = accuracy_score(test_labels[line2:], predict_labels[line2:])
    return accuracy_score0, accuracy_score1, accuracy_score2


def weight_of_prediction(y, weights):
    """
    根据多个信号的预测结果，结合权重输出最终预测结果
    :param y: [y1,y2,...,yn]  0,1,2
    :param weights: [w1,w2,...,wm]
    :return:
    """
    if set(y) == {0} or set(y) == {1} or set(y) == {2}:
        return y[0]
    elif set(y) == {0, 1}:
        return np.argmax(np.array(weights[:2]) * np.bincount(y))
    else:
        return np.argmax(np.array(weights)*np.bincount(y))


def prediction_use_prob(probas, num_samples):
    """
    根据概率计算分类（加权）
    :param prob:
    :param num_samples: 投票的信号个数
    :return:
    """
    predict_y = []
    # w = [1.04, 1., 1.03]
    # for i in range(probas.shape[0]):
    #     probas[i] = (probas[i]*w)/sum(probas[i]*w)
    probas = probas*[1., 1., 1.]
    for i in range(0, len(probas), num_samples):
        p = list(sum(probas[i:i + num_samples, :]))
        predict_y.append(p.index(max(p)))
    return predict_y


def one_tensor_data(data_file, time1, time2, sensor_index):
    """
    :param data_file:
    :param time1:
    :param time2:
    :param sensor_index:
    :return: data_x:pd.DataFrame data_y:[]
    """
    with open(data_file, 'r') as fl:
        data = pd.read_csv(fl)
    data_1 = data.loc[data['通道'] == sensor_index, :]
    times = data_1.iloc[:, 1]
    data_x = data_1.drop(['信号强度', '初始频率', '绝对能量', '中心频率', '峰频'], axis=1).iloc[:, 2:]
    data_y = generate_y(times, time1, time2)
    return data_x, data_y


def four_tensor_data(data_file, time1, time2, type=0):
    """
    输出四个通道的数据
    :param data_file:
    :return:data_x:pd.DataFrame, data_y:[]
    """
    with open(data_file, 'r') as fl:
        data = pd.read_csv(fl)
    if not type:
        times = data.iloc[:, 1]
        data_x = data.iloc[:, 2:]
        data_y = generate_y(times, time1, time2)
        return data_x, data_y
    times = data.iloc[:, 1]
    data_x = data.drop(['信号强度', '初始频率', '绝对能量', '中心频率', '峰频'], axis=1).iloc[:, 2:]
    data_y = generate_y(times, time1, time2)
    return data_x, data_y


def four_tensor_frequency(frequency_data_file, time1, time2):
    with open(frequency_data_file, 'r') as fl:
        f_data = pd.read_csv(fl)

    times = f_data['TIME OF TEST']
    data_x = f_data.iloc[:, 2:]
    data_y = generate_y(times, time1, time2)
    return data_x, data_y


def four_tensor_data_added_frequency(data_file, frequency_data_file, time1, time2, pca_n=None):
    with open(data_file, 'r') as fl:
        data = pd.read_csv(fl)
    times = data.iloc[:, 0]
    data_x = data.drop(['信号强度', '初始频率', '绝对能量', '中心频率', '峰频'], axis=1).iloc[:, 2:]
    data_y = generate_y(times, time1, time2)

    with open(frequency_data_file, 'r') as fl:
        f_data = pd.read_csv(fl)
    f_data_x = f_data.iloc[:, 2:]
    if pca_n:
        f_data_x = pd.DataFrame(data=PCA(n_components=pca_n).fit_transform(f_data_x))
    data_x = pd.concat([data_x, f_data_x], axis=1)
    return data_x, data_y


def add_frequency(data_file, frequency_data_file, sensor_index, time1, time2):
    with open(frequency_data_file, 'r') as fl:
        f_data = pd.read_csv(fl)

    # CHANNEL NUMBER, TIME OF TEST, 0, 1, 2 ... 198, 199

    f_data_x = f_data.loc[f_data['CHANNEL NUMBER'] == sensor_index, :].iloc[:, 2:]
    times = f_data['TIME OF TEST']

    with open(data_file, 'r') as fl:
        data = pd.read_csv(fl)
    data_1 = data.loc[data['通道'] == sensor_index, :].iloc[:, 2:-1]
    data_x = data_1.drop(['信号强度', '初始频率', '绝对能量', '中心频率', '峰频'], axis=1)
    data_x = pd.concat([data_x, f_data_x], axis=1)
    data_y = generate_y(times, time1, time2)
    return data_x, data_y


# ----------------------for cnn model------------------------
def fft_generate_y(times, t1, t2):
    """
    根据时间序列生成标签
    :param times:
    :return:
    """
    data_y = []
    for time in times:
        if time <= t1:
            data_y.append([1, 0, 0])
        elif time <= t2:
            data_y.append([0, 1, 0])
        else:
            data_y.append([0, 0, 1])
    return np.array(data_y)


def fft_four_tensor_frequency(frequency_data_file, time1, time2):
    with open(frequency_data_file, 'r') as fl:
        f_data = pd.read_csv(fl)
    times = f_data['TIME OF TEST']
    data_x = f_data.iloc[:, 2:]
    data_y = fft_generate_y(times, time1, time2)
    return data_x, data_y


def fft_split_data(data_x, data_y, sample0_test_proportion, sample1_test_proportion, sample2_test_proportion):
    """
    用随机的方法，按照指定的各类测试集样本比例生成训练集和测试集
    :param data_x: 样本, np.array
    :param data_y: 标签, np.array
    :param sample0_test_proportion: 0类样本测试集比例
    :param sample1_test_proportion: 1类样本测试集比例
    :param sample2_test_proportion: 2类样本测试集比例
    :return:
    """
    data0_x, data1_x, data2_x = [], [], []
    data0_y, data1_y, data2_y = [], [], []
    for x, y in zip(data_x, data_y):
        if y[0] == 1:
            data0_x.append(x)
            data0_y.append(y)
        elif y[1] == 1:
            data1_x.append(x)
            data1_y.append(y)
        else:
            data2_x.append(x)
            data2_y.append(y)

    x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(data0_x, data0_y, test_size=sample0_test_proportion,
                                                                random_state=8)
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data1_x, data1_y, test_size=sample1_test_proportion,
                                                                random_state=8)
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(data2_x, data2_y, test_size=sample2_test_proportion,
                                                                random_state=8)

    x_train = np.concatenate((x_train_0, x_train_1, x_train_2), axis=0)
    x_test = np.concatenate((x_test_0, x_test_1, x_test_2), axis=0)
    y_train = np.concatenate((y_train_0, y_train_1, y_train_2), axis=0)
    y_test = np.concatenate((y_test_0, y_test_1, y_test_2), axis=0)

    return x_train, x_test, y_train, y_test

# ----------------------for cnn model------------------------

