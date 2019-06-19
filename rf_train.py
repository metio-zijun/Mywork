import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import re
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')  # 忽略warning
pd.set_option('display.max_columns', None)  # 输出结果显示全部列
"""
读取文件
"""
path = '/home/metio/桌面/1train.csv'
df = pd.read_csv(path)
print(df.info())
"""
RangeIndex: 905 entries, 0 to 904
Data columns (total 14 columns):
Unnamed: 0     905 non-null float64
Unnamed: 1     905 non-null float64
Unnamed: 2     905 non-null float64
Unnamed: 3     905 non-null float64
Unnamed: 4     905 non-null float64
Unnamed: 5     905 non-null float64
Unnamed: 6     905 non-null float64
Unnamed: 7     905 non-null float64
Unnamed: 8     905 non-null float64
Unnamed: 9     905 non-null float64
Unnamed: 10    905 non-null float64
Unnamed: 11    905 non-null float64
Unnamed: 12    905 non-null float64
quality        905 non-null int64
dtypes: float64(13), int64(1)
memory usage: 99.1 KB
"""
columns_lst = list(range(13))
df.columns = columns_lst + ['y']
corr = df.corr()
print(corr)
sns.heatmap(corr)
print(corr.y.sort_values())
"""
9    -0.194697
5    -0.154558
7    -0.103132
2    -0.099346
4    -0.096463
1    -0.052648
10   -0.018254
11   -0.017419
0     0.005215
12    0.011436
6     0.066338
3     0.071486
8     0.111926
y     1.000000
Name: y, dtype: float64
删掉 10，11，0，12 没用的特征
"""
df = df.drop([10, 11, 0, 12], axis=1)
# 5，6和8，9特征重复，删除5，6
df = df.drop([5, 6], axis=1)

train, test = train_test_split(df, test_size=0.2)
"""
train.shape
(724, 8)
test.shape
(181, 8)
"""


ntrain = train.shape[0]
ntest = test.shape[0]
x_train = train.iloc[:, :-1]
x_test = test.iloc[:, :-1]
y_train = train.iloc[:, -1]
y_test = test.iloc[:, -1]
SEED = 0
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, random_state=SEED)
kf = kf.split(train)

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    i = -1
    for train_index, test_index in list(kf):
        i += 1
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# 随机森林的参数
rf_params = {
    'n_jobs': -1,
    'n_estimators': 100,
     'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

# Extra Trees的参数
et_params = {
    'n_jobs': -1,
    'n_estimators': 100,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost的参数
ada_params = {
    'n_estimators': 100,
    'learning_rate': 0.01
}

# Gradient Boosting的参数
gb_params = {
    'n_estimators': 100,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier的参数
svc_params = {
    'kernel': 'linear',
    'C': 0.025
}

rf_features = rf.feature_importances(x_train, y_train)
et_features = et.feature_importances(x_train, y_train)
ada_features = ada.feature_importances(x_train, y_train)
gb_features = gb.feature_importances(x_train, y_train)

cols = train.columns.values
feature_dataframe = pd.DataFrame({'features': cols,
     'Random Forest feature importances': rf_features,
     'Extra Trees  feature importances': et_features,
      'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features})

feature_dataframe['mean'] = feature_dataframe.mean(axis=1)  # axis = 1 computes the mean row-wise
print(feature_dataframe.head(11))
