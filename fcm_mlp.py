# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from smote import Smote
from data_transformation import four_tensor_data, prediction_use_prob
import json


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
    return accuracy_score0, accuracy_score1


def split_data(data_x, data_y, sample0_test_proportion, sample1_test_proportion, random=1):
    """
    用随机的方法，按照指定的各类测试集样本比例生成训练集和测试集
    :param data_x: 样本, np.array
    :param data_y: 标签,[]
    :param sample0_test_proportion: 0类样本测试集比例
    :param sample1_test_proportion: 1类样本测试集比例
    :param sample2_test_proportion: 2类样本测试集比例
    :return:
    """
    data0_x, data1_x = [], []
    data0_y, data1_y = [], []
    for x, y in zip(data_x, data_y):
        if y == 0:
            data0_x.append(x)
            data0_y.append(y)
        else:
            data1_x.append(x)
            data1_y.append(y)

    if random:
        x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(data0_x, data0_y, test_size=sample0_test_proportion,
                                                                    random_state=8)
        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data1_x, data1_y, test_size=sample1_test_proportion,
                                                                    random_state=8)

        x_train = np.concatenate((x_train_0, x_train_1), axis=0)
        y_train = np.concatenate((y_train_0, y_train_1), axis=0)

        if x_test_0:
            x_test = np.concatenate((x_test_0, x_test_1), axis=0)
            y_test = np.concatenate((y_test_0, y_test_1), axis=0)
        else:
            x_test = []
            y_test = []
    else:
        x_train = np.concatenate((data0_x, data1_x), axis=0)
        x_test = []
        y_train = np.concatenate((data0_y, data1_y), axis=0)
        y_test = []

    return x_train, x_test, y_train, y_test


pd.set_option('display.width', 1000)


def f_start_calculate():
    data_file1 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\t滤波后-7200.csv"
    label_file1 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\t_standardscale_labels.json"

    with open(label_file1, 'r') as fo:
        data_labels = json.load(fo)
    labels = data_labels['FCM1']
    labels += data_labels['FCM2']
    labels += data_labels['FCM3']
    labels += data_labels['FCM4']

    train_X, m = four_tensor_data(data_file1, 700, 2200)

    train_X, a, train_Y, b = split_data(np.array(train_X), labels, 0., 0., random=1)

    ss1 = StandardScaler().fit(train_X)
    train_X = ss1.transform(train_X)

    data_file2 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件3-0324\正式加载\t滤波后-7200.csv"
    initial_test_x, initial_test_y = four_tensor_data(data_file2, 530, 1980)
    label_file2 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件3-0324\正式加载\t_standardscale_labels.json"

    with open(label_file2, 'r') as fo:
        data_labels = json.load(fo)
    labels = data_labels['FCM1']
    labels += data_labels['FCM2']
    labels += data_labels['FCM3']
    labels += data_labels['FCM4']
    test_X, a, test_Y, b = split_data(np.array(initial_test_x), labels, 0., 0., random=0)
    test_X = ss1.transform(test_X)

    # # 创造新样本
    s0 = Smote(train_X[26925:], N=13, k=10)
    new_samples0 = s0.over_sampling()

    # 添加新样本
    train_X = np.concatenate((train_X, new_samples0), axis=0)
    train_Y = np.concatenate((train_Y, np.    array([1] * len(new_samples0))), axis = 0)

    ss2 = StandardScaler().fit(train_X)
    train_scale_X = ss2.transform(train_X)
    test_scale_X = ss2.transform(test_X)

    print("训练集不同类别信号的个数：", list(train_Y).count(0), list(train_Y).count(1),
          '信号总数：', len(train_Y))

    clf_nn = MLPClassifier(solver='adam',
                           alpha=1.5,
                           activation='relu',
                           hidden_layer_sizes=(5,),
                           max_iter=2000,
                           verbose=0,
                           learning_rate='adaptive',
                           early_stopping=True,
                           random_state=6)

    clf_nn.fit(train_scale_X, train_Y)
    predictions = clf_nn.predict(test_scale_X)
    print("测试集不同类别信号的个数：", list(test_Y).count(0), list(test_Y).count(1),
          '信号总数：', len(test_Y))
    print('     test_Y:', list(test_Y))
    print('predictions:', list(predictions))
    # print("各类准确率：", accuracy(test_Y, predictions))
    # print("综合准确率：", accuracy_score(test_Y, predictions))

    # train_prob = np.concatenate((clf_nn.predict_proba(train_scale_X), train_scale_X), axis=1)
    test_prob = clf_nn.predict_proba(test_scale_X)

    output = []  #

    # return
    for num_samples in range(1, 2):
        test_y = []
        for i in range(0, len(test_Y), num_samples):
            test_y.append(np.argmax(np.bincount(test_Y[i:i+num_samples])))
        predict_y = []
        for i in range(0, len(predictions), num_samples):
            predict_y.append(np.argmax(np.bincount(predictions[i:i+num_samples])))

        # print("测试集区块的数量：", list(test_y).count(0), list(test_y).count(1), list(test_y).count(2),
        #       '总数：', len(test_y))
        # print("   test_y:", test_y)
        # print("predict_y:", predict_y)
        # print("各类准确率：", accuracy(test_y, predict_y))
        # print("综合准确率：", accuracy_score(test_y, predict_y))
        # print("基于概率缩放后各类准确率：", accuracy(test_y, prediction_use_prob(test_prob, num_samples)))
        # print("基于概率缩放后综合准确率：", accuracy_score(test_y, prediction_use_prob(test_prob, num_samples)))

        output.append([accuracy_score(test_y, predict_y)] + list(accuracy(test_y, predict_y)) +
                      [accuracy_score(test_y, prediction_use_prob(test_prob, num_samples))] +
                      list(accuracy(test_y, prediction_use_prob(test_prob, num_samples))))

    print(pd.DataFrame(data=output, columns=["综合准确率", "p0", 'p1', "概率分类的综合准确率", 'wp0', 'wp1']))

f_start_calculate()

