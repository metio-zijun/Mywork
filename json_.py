import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

b = open(r"data_source_list.txt", "r",encoding='UTF-8')
out = b.read()
result = json.loads(out)

num_leaves = [i['params']['num_leaves'] for i in result]
max_depth = [i['params']['max_depth'] for i in result]
reg_alpha = [i['params']['reg_alpha'] for i in result]
min_child_samples = [i['params']['min_child_samples'] for i in result]
boosting = [i['params']['boosting_type'] for i in result]
colsample_bytree = [i['params']['colsample_bytree'] for i in result]
subsample = [i['params']['subsample'] for i in result]

loss = [1-i['loss'] for i in result]
idx = np.arange(0, 1000, 1).tolist()
def df_build(x, y):
    df = pd.DataFrame([x,y])
    df = df.transpose()
    df.columns = [['x', 'y']]
    return df

df_leaves = df_build(num_leaves, loss)
df_depth = df_build(max_depth, loss)
df_reg = df_build(reg_alpha, loss)
df_samples = df_build(min_child_samples, loss)
df_subsample = df_build(subsample, loss)
df_colsample_bytree = df_build(colsample_bytree, loss)
df_boosting = df_build(boosting, loss)


sns.jointplot(x='x',y='y',data=df_leaves)
plt.show()

sns.jointplot(x='x',y='y',data=df_depth)
plt.show()

sns.jointplot(x='x',y='y',data=df_depth, kind='kde')
plt.show()

sns.jointplot(x='x',y='y',data=df_reg, kind='kde')
plt.show()

sns.jointplot(idx, loss)
plt.show()

sns.regplot(idx, loss)
plt.show()

sns.distplot(df_reg['x'],hist=False,rug=True, label='reg_alpha')
plt.show()
sns.distplot(df_depth['x'],hist=False,rug=True, label='max_depth')
plt.show()
sns.distplot(df_leaves['x'],hist=False,rug=True, label='num_leaves')
plt.show()
sns.distplot(df_samples['x'],hist=False,rug=True, label='min_child_samples')
plt.show()


def barplot_(df):
    array = df.values
    x = array[:, 0]
    y = array[:, 1]
    sns.barplot(x, y, capsize=.2)
    plt.show()

def kdeplot_(df):
    array = df.values
    x = array[:, 0]
    y = array[:, 1]
    sns.kdeplot(x, y, shade=True)
    plt.show()