# -*- coding:utf-8 -*-

# DBSCAN聚类算法试验

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.set_option('display.width', 1000)

file_label = r"data\正式加载\standardscale_labels.json"
file_data = r"data\正式加载\滤波后-7200.csv"
#
tunnel_index = 2
# with open(file_label, 'r') as fo:
#     data_labels = json.load(fo)
#     labels = data_labels['FCM'+str(tongdao_index)]
with open(file_data, 'r') as fl:
    data = pd.read_csv(fl)

data_1 = data.loc[data['通道'] == tunnel_index, :]
X = data_1.iloc[:, 2:-1].drop(['信号强度', '初始频率', '绝对能量', '中心频率', '峰频'], axis=1)
X_scale = StandardScaler().fit_transform(X)  # 标准化
X_dimensionality_reduction = PCA(n_components=2).fit_transform(X_scale)  # pca降维

db = DBSCAN(eps=4.5, min_samples=8).fit(X_scale)
labels = db.labels_
print(labels, set(labels))
print(db.core_sample_indices_)
n_clusters_ = len(set(labels))-(1 if -1 in labels else 0)

print(n_clusters_)
if n_clusters_ == 2:
    for label, i in zip(labels, range(len(labels))):
        if label == -1:
            plt.scatter(X_dimensionality_reduction[i, 0], X_dimensionality_reduction[i, 1], color='r', s=25, marker='o')
        elif label == 1:
            plt.scatter(X_dimensionality_reduction[i, 0], X_dimensionality_reduction[i, 1], color='k', s=25, marker='o')
        else:
            plt.scatter(X_dimensionality_reduction[i, 0], X_dimensionality_reduction[i, 1], color='b', s=25, marker='o')
    plt.show()



