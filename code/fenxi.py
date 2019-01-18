# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

train_data = pd.read_csv('../train.csv')
print(train_data.info())
feature_name = ['Pclass', 'Sex', 'Age', 'SibSp',
                'Parch', 'Fare', 'Cabin', 'Embarked']
train_x = train_data.loc[:, feature_name]
train_y = train_data.loc[:, 'Survived']
print(train_x)

# 查看存活与死亡人数,0表示死,1表示活
# mp.figure()
# mp.title('Survived', fontsize=20)
# mp.ylabel('num', fontsize=14)
# train_data.Survived.value_counts().plot(kind='bar')


# # 查看船舱登记人数
# mp.figure()
# mp.title('Pclass', fontsize=20)
# mp.ylabel('num', fontsize=14)
# train_data.Pclass.value_counts().plot(kind='bar')

# live = train_data.Pclass[train_data.Survived == 1].value_counts()
# died = train_data.Pclass[train_data.Survived == 0].value_counts()
# df = pd.DataFrame({'live': live, 'died': died})
# df.plot(kind='bar')
# mp.title('Pclass-Survived', fontsize=20)
# mp.ylabel('num', fontsize=14)


# # 查看性别情况
# mp.figure()
# mp.title('Sex', fontsize=20)
# mp.ylabel('num', fontsize=14)
# train_data.Sex.value_counts().plot(kind='bar')

# live = train_data.Sex[train_data.Survived == 1].value_counts()
# died = train_data.Sex[train_data.Survived == 0].value_counts()
# df = pd.DataFrame({'live': live, 'died': died})
# df.plot(kind='bar')
# mp.title('Sex-Survived', fontsize=20)
# mp.ylabel('num', fontsize=14)

# 查看年龄情况
# mp.figure()
# mp.title('Age', fontsize=20)
# mp.ylabel('num', fontsize=14)
# train_data.Age.value_counts().plot(kind='bar')

# live = (train_data.Age[train_data.Survived == 1][
#     train_data.Age == train_data.Age] / 5).astype(int).value_counts()
# died = (train_data.Age[train_data.Survived == 0][
#     train_data.Age == train_data.Age] / 5).astype(int).value_counts()
# df = pd.DataFrame({'live': live, 'died': died})
# df.plot()
# mp.title('Age-Survived', fontsize=20)
# mp.ylabel('num', fontsize=14)

# Q = (train_data.Age[train_data.Embarked == 'Q'][
#     train_data.Age == train_data.Age] / 5).astype(int).value_counts()
# S = (train_data.Age[train_data.Embarked == 'S'][
#     train_data.Age == train_data.Age] / 5).astype(int).value_counts()
# C = (train_data.Age[train_data.Embarked == 'C'][
#     train_data.Age == train_data.Age] / 5).astype(int).value_counts()
# df = pd.DataFrame({'Q': Q, 'S': S, 'C': C})
# df.plot(kind='bar')
# mp.title('Age-Survived', fontsize=20)
# mp.ylabel('num', fontsize=14)

# y = train_data.Age
# for i, C in enumerate(train_data.Age):
#     if C == C:
#         train_data.Age[i] = 1
#     else:
#         train_data.Age[i] = 0
# live = train_data.Age[train_data.Survived == 1].astype(int).value_counts()
# died = train_data.Age[train_data.Survived == 0].astype(int).value_counts()
# df = pd.DataFrame({'live': live, 'died': died})
# df.plot(kind='bar')
# mp.title('Age-Survived', fontsize=20)
# mp.ylabel('num', fontsize=14)

# # 兄弟姐妹及配偶个数-存活情况
# live = train_data.SibSp[train_data.Survived == 1].value_counts()
# died = train_data.SibSp[train_data.Survived == 0].value_counts()
# df = pd.DataFrame({'live': live, 'died': died})
# df.plot(kind='bar')
# mp.title('SibSp-Survived', fontsize=20)
# mp.ylabel('num', fontsize=14)

# # 父母子女个数-存活情况
# live = train_data.Parch[train_data.Survived == 1].value_counts()
# died = train_data.Parch[train_data.Survived == 0].value_counts()
# df = pd.DataFrame({'live': live, 'died': died})
# df.plot(kind='bar')
# mp.title('Parch-Survived', fontsize=20)
# mp.ylabel('num', fontsize=14)

# # 票价
# mp.figure()
# mp.title('Fare', fontsize=20)
# mp.ylabel('num', fontsize=14)
# (train_data.Fare / 20).astype(int).value_counts().plot(kind='bar')

# live = (train_data.Fare[train_data.Survived == 1] /
#         20).astype(int).value_counts()
# died = (train_data.Fare[train_data.Survived == 0] /
#         20).astype(int).value_counts()
# df = pd.DataFrame({'live': live, 'died': died})
# df.plot(kind='bar')
# mp.title('Fare-Survived', fontsize=20)
# mp.ylabel('num', fontsize=14)

# # 舱位
# mp.figure()
# mp.title('Cabin', fontsize=20)
# mp.ylabel('num', fontsize=14)
# train_data.Cabin.value_counts().plot(kind='bar')

# # 将Cabin变成有船舱和没船舱两类，有/无船舱的人的存活情况
# for i, C in enumerate(train_data.Cabin):
#     if C == C:
#         train_data.Cabin[i] = 1
#     else:
#         train_data.Cabin[i] = 0
# live = train_data.Cabin[train_data.Survived == 1].value_counts()
# died = train_data.Cabin[train_data.Survived == 0].value_counts()
# df = pd.DataFrame({'live': live, 'died': died})
# df.plot(kind='bar')
# mp.title('Cabin-Survived', fontsize=20)
# mp.ylabel('num', fontsize=14)

# 登船地点
# mp.figure()
# mp.title('Embarked', fontsize=20)
# mp.ylabel('num', fontsize=14)
# train_data.Embarked.value_counts().plot(kind='bar')

# live = train_data.Embarked[train_data.Survived == 1].value_counts()
# died = train_data.Embarked[train_data.Survived == 0].value_counts()
# df = pd.DataFrame({'live': live, 'died': died})
# df.plot(kind='bar')
# mp.title('Embarked-Survived', fontsize=20)
# mp.ylabel('num', fontsize=14)

P1 = train_data.Embarked[train_data.Pclass == 1].value_counts()
P2 = train_data.Embarked[train_data.Pclass == 2].value_counts()
P3 = train_data.Embarked[train_data.Pclass == 3].value_counts()
df = pd.DataFrame({'P1': P1, 'P2': P2, 'P3': P3})
df.plot(kind='bar')
mp.title('Embarked-Pclass', fontsize=20)
mp.ylabel('num', fontsize=14)

mp.show()
