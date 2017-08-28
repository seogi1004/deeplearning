import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import random
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
sns.set(style='ticks') #http://seaborn.pydata.org/tutorial/color_palettes.html
color = sns.color_palette()

train_df = pd.read_csv("data/cpu_train.csv")
test_df = pd.read_csv('data/cpu_test.csv')

# 필드 추가
count = len(train_df["프로세스 CPU사용률 (%)"])
levels = []
for i in range(count):
    rate = train_df["프로세스 CPU사용률 (%)"][i]
    levels.append(int(rate))
train_df["구간"] = pd.Series(levels, index=train_df.index)


# 0값 처리
train_df = train_df[train_df["Current Thread 수"] != 0]
train_df = train_df[train_df["호출 건수"] != 0]
train_df = train_df[train_df["동시 사용자"] != 0]
train_df = train_df[train_df["ERROR 수"] != 0]
train_df = train_df[train_df["프로세스 CPU사용률 (%)"] != 0]

# print(train_df.head())
print(train_df.describe())
# print(train_df.iloc[0][['시간', 'Current Thread 수' ]])
# ax = sns.countplot(x="구간", data=train_df, order=range(99))
# plt.show()

print(train_df.groupby('구간').mean())

sns.distplot(train_df['호출 건수'], hist=False).set(xlim=(0, 10000))

# for l, g in train_df.groupby('구간'):
#     sns.kdeplot(np.log(g['동시 사용자']), label=l, cumulative = False).set(xlim=(1, 8640))
plt.show()