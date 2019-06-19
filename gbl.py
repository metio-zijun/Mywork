import pandas as pd
import lightgbm as lgb
import numpy as np

df = pd.read_csv('/home/metio/桌面/1train.csv')
train_features, train_labels = df.iloc[:, :-1], df.iloc[:, -1]

train_set = lgb.Dataset(train_features, train_labels)

cv_results = lgb.cv(train_set, nfold=10, num_boost_round=1000,
                    early_stopping_rounds=100, metrics='auc', seed=50, verbose_eval=0)

best_score = max(cv_results['auc-mean'])

# Loss must be minimized
loss = 1 - best_score
