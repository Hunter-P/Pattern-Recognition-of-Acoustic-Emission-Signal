# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf
from smote import Smote
from data_transformation import prediction_use_prob, fft_four_tensor_frequency, fft_split_data


def accuracy_out(test_labels, predict_labels):
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


def generate_train_data(scale_data_X, data_Y, start, num_samples, gap):
    """
    从总样本集中抽取一部分样本作为测试集， n个信号组成一个测试样本，其余作为训练集
    :param scale_scale_X: np.ndarray，输入前进行归一化
    :param data_Y:  list[]
    :param start:  切片起始点, 0
    :param num_samples: 信号个数, 9
    :param gap:   间隔, 30
    :return:训练集和测试集 np.array
    """
    num_sum = scale_data_X.shape[0] - num_samples
    delete_rows = []  # 删除的行索引
    scale_data_X = np.array(scale_data_X)
    test_scale_X = scale_data_X[start:start+num_samples, :]
    test_Y = data_Y[start:start+num_samples, :]

    for i in range(start+gap, num_sum, gap):
        test_scale_X = np.concatenate((test_scale_X, scale_data_X[i:i+num_samples, :]), axis=0)
        test_Y = np.concatenate((test_Y, data_Y[i:i + num_samples, :]), axis=0)
        for j in range(i, i+num_samples):
            delete_rows.append(j)

    train_scale_X = np.delete(scale_data_X, delete_rows, axis=0)
    train_Y = np.delete(data_Y, delete_rows, axis=0)
    return train_scale_X, train_Y, test_scale_X, test_Y


# 定义权重
def weight_variable(shape):
    # 截断的正态分布，标准差为0.1
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32))
    # return tf.constant(1, shape=shape, dtype=tf.float32)  # 设置为常量0.1


# define weight decay loss(using L2 regulation)
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


# 定义偏置
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32))  # 设置为常量0.1


# 定义卷积层
def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化层
def max_pool_1x2(x):
    return tf.nn.pool(x, window_shape=[2], strides=[2], pooling_type='MAX', padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


pd.set_option('display.width', 1000)

# 数据准备
data_file = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\wave-2欠采样.csv"
# data_file = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\fft.csv"
aa, data_Y = fft_four_tensor_frequency(data_file, 700, 2200)

wavelet_file = "G:\\声发射试验\\pjx-节段胶拼压弯AE试验\\AE数据\\试件1-0324\\正式加载\\wavelet.csv"
with open(wavelet_file, 'r') as fl:
    data_X = pd.read_csv(fl)

# 按照比例随机划分训练集和测试集
train_X, m, train_Y, n = fft_split_data(np.array(data_X), data_Y, 0., 0., 0.)
train_X = StandardScaler().fit_transform(train_X)

# 测试集
data_file2 = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件2-0324\正式加载\wave-2欠采样.csv"
bb, initial_test_y = fft_four_tensor_frequency(data_file2, 550, 2200)
wavelet_file = "G:\\声发射试验\\pjx-节段胶拼压弯AE试验\\AE数据\\试件2-0324\\正式加载\\wavelet.csv"
with open(wavelet_file, 'r') as fl:
    initial_test_x = pd.read_csv(fl)

test_X, a, test_Y, b = fft_split_data(np.array(initial_test_x), initial_test_y, 0., 0., 0.)
test_X = StandardScaler().fit_transform(test_X)

# 创造新样本
s0 = Smote(train_X[:430], N=54, k=8)
new_samples0 = s0.over_sampling()
s2 = Smote(train_X[23954:], N=4, k=5)
new_samples2 = s2.over_sampling()
# 添加新样本
train_X = np.concatenate((train_X, new_samples0, new_samples2), axis=0)
train_Y = np.concatenate((train_Y, np.array([[1, 0, 0]] * len(new_samples0)),
                          np.array([[0, 0, 1]] * len(new_samples2))), axis=0)

# print("训练集不同类别信号的个数：", list(train_Y).count([1, 0, 0]), list(train_Y).count([0, 1, 0]),
#       list(train_Y).count([0, 0, 1]), '总数：', len(train_Y))

print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)

# 归一化
rows, columns = train_X.shape
scale_X = StandardScaler().fit_transform(np.concatenate((train_X, test_X), axis=0))
train_scale_X = scale_X[:rows]
test_scale_X = scale_X[rows:]


# train_Y = tf.reshape(train_Y, [-1, 3])
x = tf.placeholder(tf.float32, [None, 512])
y_ = tf.placeholder(tf.float32, [None, 3])
x_signal = tf.reshape(x, [-1, 512, 1])
# x_signal = tf.reshape(x, [-1, 20, 10, 1])

# 定义第一个卷积层
W_conv1 = weight_variable([5, 1, 4])
b_conv1 = bias_variable(([4]))
h_conv1 = tf.nn.relu(conv1d(x_signal, W_conv1)+b_conv1)  # 激活函数
h_pool1 = max_pool_1x2(h_conv1)

# 定义第二个卷积层
W_conv2 = weight_variable([5, 4, 8])
b_conv2 = bias_variable(([8]))
h_conv2 = tf.nn.relu(conv1d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_1x2(h_conv2)

# 定义全连接层
W_fc1 = weight_variable([128*8, 100])
b_fc1 = bias_variable([100])
h_pool2_flat = tf.reshape(h_pool2, [-1, 128*8])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

# 定义dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 连接softmax层
W_fc2 = weight_variable([100, 3])
b_fc2 = bias_variable([3])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)


# 定义损失函数
cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

# 定义指数衰减的学习率
# global_step = tf.Variable(0)
# learning_rate = tf.train.exponential_decay(0.02, global_step, 1, 0.95, staircase=True)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entroy, global_step=global_step)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entroy)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练模型
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(1, 500):
    train_accuracy = accuracy.eval(feed_dict={x: train_scale_X, y_: train_Y, keep_prob: 1.0})
    train_loss = cross_entroy.eval(feed_dict={x: train_scale_X, y_: train_Y, keep_prob: 1.0})
    print("step %d, training accuracy %g, training loss %g" % (i, train_accuracy, train_loss))
    if train_loss == np.nan:
        break
    train_step.run(feed_dict={x: train_scale_X, y_: train_Y, keep_prob: 0.5})

# 查看预测效果
prediction_prob = y_conv.eval(feed_dict={x: test_scale_X, keep_prob: 1.0})
predictions = tf.argmax(y_conv, 1)
predict_y = predictions.eval(feed_dict={y_conv: prediction_prob})   # 预测的概率转换为[0,1,2...]
test_y = predictions.eval(feed_dict={y_conv: test_Y})               # 测试集标签转换为[0,1,2...]
print(list(test_y))
print(list(predict_y))
print("test accuracy: %g" % accuracy.eval(feed_dict={x: test_scale_X, y_: test_Y, keep_prob: 1.0}))
print("各类准确率：%g, %g, %g" % accuracy_out(list(test_y), list(predict_y)))

output = []
for num_samples in range(1, 21):
    n_test_y = []
    for i in range(0, len(test_y), num_samples):
        n_test_y.append(np.argmax(np.bincount(test_y[i:i + num_samples])))
    n_predict_y = []
    for i in range(0, len(predict_y), num_samples):
        n_predict_y.append(np.argmax(np.bincount(predict_y[i:i + num_samples])))

    output.append([accuracy_score(n_test_y, n_predict_y)] + list(accuracy_out(n_test_y, n_predict_y)) +
                  [accuracy_score(n_test_y, prediction_use_prob(prediction_prob, num_samples))] +
                  list(accuracy_out(n_test_y, prediction_use_prob(prediction_prob, num_samples))))

print(pd.DataFrame(data=output, columns=["综合准确率", "p0", 'p1', 'p2', "概率分类的综合准确率", 'wp0', 'wp1', 'wp2']))





