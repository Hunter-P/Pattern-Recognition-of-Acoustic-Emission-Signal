# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import csv
from feature_creation import FeatureCreation
from data_transformation import split_data, four_tensor_data, four_tensor_frequency, accuracy, prediction_use_prob
from data_transformation import four_tensor_data_added_frequency
from smote import Smote
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC


def signal_connection(signal_data, signal_label, num=2):
    if num > 1:
        data_x = []
        data_y = []
        for i in range(0, len(signal_data)-num+1, num):
            data_x.append(np.concatenate((signal_data[i], signal_data[i+1])))
            data_y.append(np.argmax(np.bincount(signal_label[i:i + num])))
        return np.array(data_x), data_y
    else:
        return signal_data, signal_label


#
# def prob_connection(prob_data, signal_label, num):
#     data_x = []
#     data_y = []
#     for i in range(0, len(signal_data) - num + 1, num):
#         data_x.append(np.concatenate((signal_data[i], signal_data[i + 1])))
#         data_y.append(np.argmax(np.bincount(signal_label[i:i + num])))
#     return np.array(data_x), data_y


pd.set_option('display.width', 1000)

# frequency_data_file1 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\fft.csv"
data_file1 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\t滤波后-7200.csv"
data_X, data_Y = four_tensor_data(data_file1, 700, 2200)
data = pd.DataFrame(data_X)
# wave_file = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\wave.csv"
# with open(wave_file, 'r') as fl:
#     waves = pd.read_csv(fl)
# waves = waves.iloc[:, 2:]
# fc = FeatureCreation(np.array(waves))
# data['average_x'] = fc.average_x()
# data['p_x'] = fc.p_x()
# data['rms_x'] = fc.rms_x()
# data['std_x'] = fc.std_x()
# data['sk_x'] = fc.sk_x()
# data['kv_x'] = fc.kv_x()
# data['sf_x'] = fc.sf_x()
# data['cf_x'] = fc.cf_x()

wavelet_file = "G:\\声发射试验\\pjx-节段胶拼压弯AE试验\\AE数据\\试件1-0324\\正式加载\\wavelet.csv"
with open(wavelet_file, 'r') as fl:
    wavelet = pd.read_csv(fl)
# data = pd.concat((data, wavelet), axis=1)
data_X = np.array(wavelet)


def f_start_calculate():
    # 按照比例随机划分训练集和测试集
    train_X, test_X, train_Y, test_Y = split_data(np.array(data_X), data_Y, 0., 0., 0.)
    train_X = StandardScaler().fit_transform(train_X)

    # print(train_X)
    # 创造新样本
    s0 = Smote(train_X[:430], N=54, k=10)
    new_samples0 = s0.over_sampling()
    s2 = Smote(train_X[23954:], N=4, k=5)
    new_samples2 = s2.over_sampling()
    # 添加新样本
    train_X = np.concatenate((train_X, new_samples0, new_samples2), axis=0)
    train_Y = np.concatenate((train_Y, np.array([0] * len(new_samples0)), np.array([2] * len(new_samples2))), axis=0)

    # 测试集
    data_file2 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件2-0324\正式加载\t滤波后-7200.csv"
    # frequency_data_file2 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件2-0324\正式加载\fft.csv"
    initial_test_x, initial_test_y = four_tensor_data(data_file2, 550, 2200)
    # wave_file = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件2-0324\正式加载\wave.csv"
    # with open(wave_file, 'r') as fl:
    #     waves = pd.read_csv(fl)
    # waves = waves.iloc[:, 2:]
    # fc = FeatureCreation(np.array(waves))
    # data = pd.DataFrame(initial_test_x)
    # data['average_x'] = fc.average_x()
    # data['p_x'] = fc.p_x()
    # data['rms_x'] = fc.rms_x()
    # data['std_x'] = fc.std_x()
    # data['sk_x'] = fc.sk_x()
    # data['kv_x'] = fc.kv_x()
    # data['sf_x'] = fc.sf_x()
    # data['cf_x'] = fc.cf_x()

    wavelet_file = "G:\\声发射试验\\pjx-节段胶拼压弯AE试验\\AE数据\\试件2-0324\\正式加载\\wavelet.csv"
    with open(wavelet_file, 'r') as fl:
        wavelet = pd.read_csv(fl)
    # data = pd.concat((data, wavelet), axis=1)
    initial_test_x = np.array(wavelet)

    test_X, a, test_Y, b = split_data(np.array(initial_test_x), initial_test_y, 0., 0., 0.)
    test_X = StandardScaler().fit_transform(test_X)

    # 归一化
    # rows, columns = train_X.shape
    # print(train_X.shape, test_X.shape)
    # scale_X = StandardScaler().fit_transform(np.concatenate((train_X, test_X), axis=0))
    # train_scale_X = scale_X[:rows]
    # test_scale_X = scale_X[rows:]

    ss = StandardScaler().fit(train_X)
    train_scale_X = ss.transform(train_X)
    test_scale_X = ss.transform(test_X)

    print("训练集不同类别信号的个数：", list(train_Y).count(0), list(train_Y).count(1), list(train_Y).count(2),
          '信号总数：', len(train_Y))

    # clf_nn = RandomForestClassifier(n_estimators=500, min_samples_leaf=4, max_depth=4)

    # clf_nn = GradientBoostingClassifier(n_estimators=50, min_samples_leaf=10, learning_rate=0.1, random_state=1)

    clf_nn = MLPClassifier(solver='adam',
                           alpha=1.,
                           activation='relu',
                           hidden_layer_sizes=(600, ),
                           max_iter=2000,
                           verbose=1,
                           learning_rate='adaptive',
                           early_stopping=True,
                           random_state=6)

    clf_nn.fit(train_scale_X, train_Y)
    predictions = clf_nn.predict(test_scale_X)
    print("测试集不同类别信号的个数：", list(test_Y).count(0), list(test_Y).count(1), list(test_Y).count(2),
          '信号总数：', len(test_Y))
    print('     test_Y:', list(test_Y))
    print('predictions:', list(predictions))
    # print("各类准确率：", accuracy(test_Y, predictions))
    # print("综合准确率：", accuracy_score(test_Y, predictions))

    test_prob = clf_nn.predict_proba(test_scale_X)
    train_prob = clf_nn.predict_proba(train_scale_X)

    # clf_nn_prob = MLPClassifier(solver='adam',
    #                             alpha=1.,
    #                             activation='relu',
    #                             hidden_layer_sizes=(100),
    #                             max_iter=2000,
    #                             verbose=0,
    #                             learning_rate='adaptive',
    #                             early_stopping=True,
    #                             random_state=9)
    # clf_nn_prob = SVC(class_weight={0:1., 1:1., 2:1.5})

    # clf_nn_prob = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, random_state=1)

    # clf_nn_prob = GradientBoostingClassifier(n_estimators=20, min_samples_leaf=4, learning_rate=0.1, random_state=1)

    # clf_nn_prob = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=500, random_state=0)

    # clf_nn_prob.fit(train_scale_X, train_Y)
    # nn_prob_predictions = clf_nn_prob.predict(test_scale_X)
    # print("predictions:", list(nn_prob_predictions))
    # print("各类准确率：", accuracy(test_Y, nn_prob_predictions))

    output = []  #

    # return
    for num_samples in range(1, 21):
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

    print(pd.DataFrame(data=output, columns=["综合准确率", "p0", 'p1', 'p2', "概率分类的综合准确率", 'wp0', 'wp1', 'wp2']))

f_start_calculate()