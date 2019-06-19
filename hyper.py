import pandas as pd
import lightgbm as lgb
import numpy as np


df = pd.read_csv('/home/metio/桌面/1train.csv')
train_features, train_labels = df.iloc[:, :-1], df.iloc[:, -1]


from hyperopt import STATUS_OK

N_FOLDS = 10

# Create the dataset
train_set = lgb.Dataset(train_features, train_labels)


def objective(params, n_folds=N_FOLDS):
   '''Objective function for Gradient Boosting Machine Hyperparameter Tuning'''

   # Perform n_fold cross validation with hyperparameters
   # Use early stopping and evalute based on ROC AUC
   cv_results = lgb.cv(params, train_set, nfold=n_folds, num_boost_round=1000,
                       early_stopping_rounds=100, metrics='auc', seed=50, verbose_eval=0)

   # Extract the best score
   best_score = max(cv_results['auc-mean'])

   # Loss must be minimized
   loss = 1 - best_score

   # Dictionary with information for evaluation
   return {'loss': loss, 'params': params, 'status': STATUS_OK}

def objective_bymse(params, n_folds=N_FOLDS):
   '''Objective function for Gradient Boosting Machine Hyperparameter Tuning'''

   # Perform n_fold cross validation with hyperparameters
   # Use early stopping and evalute based on ROC AUC
   cv_results = lgb.cv(params, train_set, nfold=n_folds, num_boost_round=1000,
                       early_stopping_rounds=100, metrics='mse', seed=50)

   loss = min(cv_results['l2-mean'])

   # Dictionary with information for evaluation
   return {'loss': loss, 'params': params, 'status': STATUS_OK}

# Default gradient boosting machine classifier
# model = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=100,
#               class_weight=None, colsample_bytree=1.0,
#               learning_rate=0.1, max_depth=-1,
#               min_child_samples=20,
#               min_child_weight=0.001, min_split_gain=0.0,
#               n_jobs=-1, num_leaves=31, objective=None,
#               random_state=None, reg_alpha=0.0, reg_lambda=0.0,
#               silent=True, subsample=1.0,
#               subsample_for_bin=200000, subsample_freq=1)

from hyperopt import hp
from hyperopt.pyll.stochastic import sample


space = {
    'boosting_type': hp.choice('boosting_type',['gbdt','dart', 'goss']),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'max_depth': hp.choice('max_depth', range(1, 10, 1)),
    'num_leaves': hp.choice('num_leaves', range(5, 15, 1)),
    'min_child_samples': hp.choice('min_child_samples', range(1, 15, 1)),
    'subsample_freq': 0,
    'subsample': 1.0,
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}

# space = {'boosting_type': hp.choice('boosting_type',['gbdt','dart', 'goss']), # 训练方式
#         'objective': 'binary', # 目标 二分类
#          'metric': 'auc', # 损失函数
#          'is_unbalance': False,
#          'max_depth':hp.randint("max_depth",10),
#          'num_leaves':hp.choice('num_leaves',range(20, 80, 5)),
#          'max_bin':hp.choice('max_bin',range(1,255,5)),
#          'min_data_in_leaf': hp.choice('min_data_in_leaf',range(10,200,5)),
#          'min_sum_hessian_in_leaf':hp.loguniform('min_sum_hessian_in_leaf', np.log(0.0001), np.log(0.005)),
#          'min_split_gain':hp.loguniform( 'min_split_gain' ,np.log(0.1), np.log(1)),
#         'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2))}

boosting_type = {'boosting_type': hp.choice('boosting_type',
                                           [{'boosting_type': 'gbdt',
                                                 'subsample': hp.uniform('subsample', 0.5, 1)},
                                            {'boosting_type': 'dart',
                                                 'subsample': hp.uniform('subsample', 0.5, 1)},
                                            {'boosting_type': 'goss',
                                                 'subsample': 1.0}])}
 # Sample from the full space
# example = sample(space)
# # Dictionary get method with default
# subsample = example['boosting_type'].get('subsample', 1.0)
#  # Assign top-level keys
# example['boosting_type'] = example['boosting_type']['boosting_type']
# example['subsample'] = subsample

from hyperopt import tpe
# Algorithm
tpe_algorithm = tpe.suggest

from hyperopt import Trials
# Trials object to track progress
bayes_trials = Trials()

from hyperopt import fmin
MAX_EVALS = 5
# Optimize
best = fmin(fn=objective, space=space, algo=tpe.suggest,
           max_evals=MAX_EVALS, trials=bayes_trials)
print(best)
# import csv
#
# # File to save first results
# out_file = 'gbm_trials.csv'
# # Write to the csv file ('a' means append)
# of_connection = open(out_file, 'a')
# writer = csv.writer(of_connection)
# writer.writerow([loss, params, iteration, n_estimators, run_time])
# of_connection.close()