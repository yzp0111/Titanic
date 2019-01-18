# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pandas as pd
import numpy as np
import sklearn.ensemble as se
import sklearn.metrics as sm
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.preprocessing as sp
import sklearn.model_selection as ms


# 导入数据
train_data = pd.read_csv('../train.csv')
test_data_x = pd.read_csv('../test.csv')
test_data_y = pd.read_csv('../gender_submission.csv')
# print('train')
# print(train_data.info())
# print(train_data.head())
# print('test')
# print(test_data_x.info())
# print(test_data_x.head())
print(test_data_y.info())

'''
train数据中缺失了部分Age,Cabin,Embarked数据
tset数据中缺失了部分Age,Fare,Cabin数据
'''

# 将性别male/female转为0/1


def Sex2int(data):
    data.loc[data.Sex == 'male', 'Sex'] = 0
    data.loc[data.Sex == 'female', 'Sex'] = 1


# 将Cabin的值补全并变成0/1


def set_Cabin(data):
    data.loc[data.Cabin.notnull(), 'Cabin'] = 1
    data['Cabin'] = data.Cabin.fillna(0)

# 补全Embarked的值S/Q/C,并变成1,2,3


def set_Embarked(data):
    # 缺失登船信息的两人预计是母女,为同一登岸口，1等舱的旅客大部分在S口岸上船，所以将这对母女归为S
    print(data.loc[data.Embarked.isnull(), :])
    data.loc[data.Embarked.isnull(), 'Embarked'] = 'S'
    # 将 S/Q/C变为 1/2/3
    data.loc[data.Embarked == 'S', 'Embarked'] = 1
    data.loc[data.Embarked == 'Q', 'Embarked'] = 2
    data.loc[data.Embarked == 'C', 'Embarked'] = 3

# 补全票价


def set_Fare(data1, data2):
    # 由于缺少的事3等舱票价，采用三等舱中位数票价
    data2.loc[data2.Fare.isnull(), 'Fare'] = data2.loc[
        data2.Pclass == 3, 'Fare'].median()
    # 归一化
    zore2one = sp.MinMaxScaler(feature_range=(0, 1))
    zore2one.fit_transform(data1['Fare'].reshape(-1, 1))
    data2.loc[:, ['Fare']] = (zore2one.fit_transform(
        data2['Fare'].reshape(-1, 1)).T)[0]
    data1.loc[:, ['Fare']] = (zore2one.fit_transform(
        data1['Fare'].reshape(-1, 1)).T)[0]
    # data1.loc[data1.Fare.notnull(), 'Fare'] = (
    #     data1.loc[data1.Fare.notnull(), 'Fare'] / 5).astype(int)
    # data2.loc[data1.Fare.notnull(), 'Fare'] = (
    #     data2.loc[data2.Fare.notnull(), 'Fare'] / 5).astype(int)


# 补全年龄


def set_Age(data1, data2):
    data1.loc[data1.Age.notnull(), 'Age'] = (
        data1.loc[data1.Age.notnull(), 'Age'] / 5).astype(int)
    data2.loc[data1.Age.notnull(), 'Age'] = (
        data2.loc[data2.Age.notnull(), 'Age'] / 5).astype(int)
    age_feature = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
    know_age = data1.loc[data1.Age.notnull(), :]
    unknow_age_x = data1.loc[data1.Age.isnull(), age_feature]
    model = se.RandomForestRegressor(
        max_depth=4, n_estimators=50, random_state=7)
    model.fit(know_age[age_feature], know_age[['Age']])
    data1.loc[data1.Age.isnull(), 'Age'] = model.predict(
        unknow_age_x).astype(int)
    print('----------------------')
    fill_age = model.predict(
        data2.loc[data2.Age.isnull(), age_feature]).astype(int)
    data2.loc[data2.Age.isnull(), 'Age'] = fill_age
    data1['Age'] = data1['Age'].astype(int)
    data2['Age'] = data2['Age'].astype(int)

    # # 归一化
    # zore2one = sp.MinMaxScaler(feature_range=(0, 1))
    # zore2one.fit_transform(data1['Age'].reshape(-1, 1))
    # data2.loc[:, ['Age']] = (zore2one.fit_transform(
    #     data2['Age'].reshape(-1, 1)).T)[0]
    # data1.loc[:, ['Age']] = (zore2one.fit_transform(
    #     data1['Age'].reshape(-1, 1)).T)[0]

# 逻辑回归


def isSurvived1(data1, data2, data3):
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    model = lm.LogisticRegression(solver='liblinear', C=1)
    model.fit(data1[features], data1[['Survived']])
    pred_test_y = model.predict(data2[features])
    print('***')
    print(sm.r2_score(data3[['Survived']], pred_test_y))
    print('***')
    print(data2['PassengerId'].as_matrix(),
          type(data2['PassengerId'].as_matrix()))
    print(pred_test_y, type(pred_test_y))
    result = pd.DataFrame(
        {'PassengerId': data2['PassengerId'], 'Survived': pred_test_y})
    result.to_csv('./predict1.csv', index=False)


# 随机森林分类
def isSurvived2(data1, data2, data3):
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    # params = [
    #     {'max_depth': [3, 4, 5, 6, 7, 8], 'n_estimators':[i for i in range(20, 700, 20)]}]
    # model = ms.GridSearchCV(se.RandomForestRegressor(), params, cv=5)
    # model.fit(data1[features], data1[['Survived']])
    # for param, score in zip(model.cv_results_['params'], model.cv_results_['mean_test_score']):
    #     print(param, score)
    # print(model.best_params_)
    # print(model.best_score_)
    # print(model.best_estimator_)
    model = se.RandomForestClassifier(
        max_depth=4, n_estimators=400, random_state=7)
    model.fit(data1[features], data1[['Survived']])
    pred_test_y = model.predict(data2[features])
    print(sm.r2_score(data3[['Survived']], pred_test_y))
    result = pd.DataFrame(
        {'PassengerId': data2['PassengerId'], 'Survived': pred_test_y})
    result.to_csv('./predict2.csv', index=False)


# 向量机


def isSurvived3(data1, data2, data3):
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    model = svm.SVC(kernel='linear')
    model.fit(data1[features], data1[['Survived']])
    pred_test_y = model.predict(data2[features])
    print(sm.r2_score(data3[['Survived']], pred_test_y))
    print(pred_test_y)
    result = pd.DataFrame(
        {'PassengerId': data2['PassengerId'], 'Survived': pred_test_y})
    result.to_csv('./predict3.csv', index=False)


if __name__ == '__main__':
    set_Cabin(train_data)
    set_Cabin(test_data_x)
    Sex2int(train_data)
    Sex2int(test_data_x)
    set_Embarked(train_data)
    set_Embarked(test_data_x)
    set_Fare(train_data, test_data_x)
    set_Age(train_data, test_data_x)
    isSurvived2(train_data, test_data_x, test_data_y)
    print(test_data_x)
    print(test_data_x.info())
