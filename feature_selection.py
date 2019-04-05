# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_creation import FeatureCreation, FeatureEvaluation
import csv
from data_transformation import split_data, four_tensor_data, four_tensor_frequency, one_tensor_data
from data_transformation import four_tensor_data_added_frequency
from smote import Smote
from sklearn.metrics import accuracy_score


pd.set_option('display.width', 2000)

data_file = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\t滤波后-7200.csv"
parameter_feature, label = one_tensor_data(data_file, 700, 2200, 1)

wave_file = r"G:\声发射试验\pjx-节段胶拼压弯AE试验\AE数据\试件1-0324\正式加载\wave.csv"
with open(wave_file, 'r') as fl:
    data = pd.read_csv(fl)
"""
CHANNEL NUMBER, TIME OF TEST, 0, 1, 2 ... 1022, 1023
"""
fc = FeatureCreation(np.array(data.loc[data['CHANNEL NUMBER'] == 1, :].iloc[:, 2:]))
parameter_feature['average_x'] = fc.average_x()
parameter_feature['p_x'] = fc.p_x()
parameter_feature['rms_x'] = fc.rms_x()
parameter_feature['std_x'] = fc.std_x()
parameter_feature['sk_x'] = fc.sk_x()
parameter_feature['kv_x'] = fc.kv_x()
parameter_feature['sf_x'] = fc.sf_x()
parameter_feature['cf_x'] = fc.cf_x()
# print(parameter_feature.shape)

# parameter_feature = pd.concat((parameter_feature.iloc[:2000, :], parameter_feature.iloc[-1278:-1, :]), axis=0)
# label = label[:2000]+label[-1278:-1]
# print(FeatureEvaluation(parameter_feature, label).calculate_index())
f, x_test, l, y_test = split_data(np.array(parameter_feature), label, 0.1, 0.8, 0.1, random=1)
f = StandardScaler().fit_transform(f)
f = pd.DataFrame(data=f, columns=parameter_feature.columns)
print(FeatureEvaluation(f, l).calculate_index())




