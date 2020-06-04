import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
combine_df = train_df.append(test_df, ignore_index=True)

# 数据的粗略查看
# print(combine_df.shape)
# print(combine_df.columns.values)
# print(combine_df.head())
# print(combine_df.tail())
# print(combine_df.info())
# print(combine_df.describe())

# 数据清洗（缺失处理、异常值）
# 查看各个属性的缺失值
total = combine_df.isnull().sum().sort_values(ascending=False)
# print(total)

# 填补Embarked缺失值
# print(combine_df['Embarked'].value_counts())
combine_df['Embarked'] = combine_df['Embarked'].fillna('S')
# 填补Cabin缺失值
# print(combine_df['Cabin'].value_counts())
combine_df['Cabin'] = combine_df['Cabin'].fillna('U')

# 填补Fare
combine_df['Fare'] = combine_df['Fare'].fillna(combine_df['Fare'].mean())

# 填补Age
g = sns.FacetGrid(combine_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
# plt.show()



# 检查处理后的缺失值个数
total = combine_df.isnull().sum().sort_values(ascending=False)
# print(total)

# 做散点图检查异常值
data = pd.concat([combine_df['Pclass'], combine_df['Fare']], axis=1)
data.plot.scatter(x='Pclass', y='Fare', ylim=(0, 1000))
# plt.show()

# print(combine_df.sort_values(by='Fare', ascending=False)[:10])  # 删除此行
combine_df = combine_df.drop(combine_df[combine_df['PassengerId'] == 1234].index)
#
#

#
#
# # 标准化 标准差标准化
# X = combine_df['Fare'].copy()
# X_mean = X.mean(axis=0)
# X_std = X.std(axis=0)
# X1 = (X-X_mean)/X_std
# X_scale = preprocessing.scale(X)
# print(X_scale)
#

# 数据规约
# 特征工程
# Sex
sex_mapDict = {'male': 1,
               'female': 0}
combine_df['Sex'] = combine_df['Sex'].map(sex_mapDict)
# print(combine_df['Sex'].head())

# Embarked
embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies(combine_df['Embarked'], prefix='Embarked')
combine_df = pd.concat([combine_df, embarkedDf], axis=1)  # 水平合并
combine_df.drop('Embarked', axis=1, inplace=True)  # 删除列
# print(combine_df.head())

# Name


def get_title(name):
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]
    str3 = str2.strip()
    return str3


titleDf = pd.DataFrame()
titleDf['Title'] = combine_df['Name'].map(get_title)
# print(titleDf.head())
# print(titleDf['Title'].value_counts())

title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir":       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess": "Royalty",
                    "Dona":       "Miss",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr":        "Mr",
                    "Mrs":       "Mrs",
                    "Miss":      "Miss",
                    "Master":    "Master",
                    "Lady":      "Royalty"
                    }
titleDf['Title'] = titleDf['Title'].map(title_mapDict)

titleDf = pd.get_dummies(titleDf['Title'])
# print(titleDf.head())

combine_df = pd.concat([combine_df, titleDf], axis=1)
combine_df.drop('Name', axis=1, inplace=True)
# print(combine_df.head())

# 填补Age缺失值
# 求不同头衔的平均年龄后填补


def get_mean_age(df, title):
    title_age_df = df['Age'].groupby(df[title]).mean()
    return title_age_df.values[-1]


print('Mr mean age:', get_mean_age(combine_df, 'Mr'))
print('Miss mean age:', get_mean_age(combine_df, 'Miss'))
print('Officer mean age:', get_mean_age(combine_df, 'Officer'))
print('Royalty mean age:', get_mean_age(combine_df, 'Royalty'))
print('Mrs mean age:', get_mean_age(combine_df, 'Mrs'))
print('Master mean age:', get_mean_age(combine_df, 'Master'))


combine_df.ix[combine_df['Mr'] == 1, 'Age'] = combine_df.ix[combine_df['Mr'] == 1, 'Age'].fillna(get_mean_age(combine_df, 'Mr'))
combine_df.ix[combine_df['Mrs'] == 1, 'Age'] = combine_df.ix[combine_df['Mrs'] == 1, 'Age'].fillna(get_mean_age(combine_df, 'Mrs'))
combine_df.ix[combine_df['Miss'] == 1, 'Age'] = combine_df.ix[combine_df['Miss'] == 1, 'Age'].fillna(get_mean_age(combine_df, 'Miss'))
combine_df.ix[combine_df['Officer'] == 1, 'Age'] = combine_df.ix[combine_df['Officer'] == 1, 'Age'].fillna(get_mean_age(combine_df, 'Officer'))
combine_df.ix[combine_df['Royalty'] == 1, 'Age'] = combine_df.ix[combine_df['Royalty'] == 1, 'Age'].fillna(get_mean_age(combine_df, 'Royalty'))
combine_df.ix[combine_df['Master'] == 1, 'Age'] = combine_df.ix[combine_df['Master'] == 1, 'Age'].fillna(get_mean_age(combine_df, 'Master'))

# Mrdf = pd.concat(combine_df['Age'], combine_df['Mr'])
# print(Mrdf)
# Mrdf.ix[df['Age'] == NULL, 'Mr'] =


total = combine_df.isnull().sum().sort_values(ascending=False)
print(total)

# Cabin
combine_df['Cabin'] = combine_df['Cabin'].map(lambda c: c[0])
cabinDf = pd.DataFrame()
cabinDf = pd.get_dummies(combine_df['Cabin'], prefix='Cabin')
# print(cabinDf.head())
combine_df = pd.concat([combine_df, cabinDf], axis=1)
combine_df.drop('Cabin', axis=1, inplace=True)
# print(combine_df.head())

# Family_Size
familysizeDf = pd.DataFrame()
familysizeDf['Family_Size'] = combine_df['Parch'] + combine_df['SibSp'] + 1
combine_df = pd.concat([combine_df, familysizeDf], axis=1)
combine_df.drop('SibSp', axis=1, inplace=True)
combine_df.drop('Parch', axis=1, inplace=True)
# print(combine_df.head())

# print(combine_df.info())

# 特征选择
corrDf = combine_df.corr()
# print(corrDf['Survived'].sort_values(ascending=False))
full_X = pd.concat([titleDf,
                    combine_df['Sex'],
                    cabinDf,
                    combine_df['Pclass'],
                    combine_df['Fare'],
                    embarkedDf],
                    axis=1)
# print(full_X.head())

# # 离散化 等宽离散

data = combine_df[u'Fare'].copy()
d1 = pd.cut(data, 10, labels=range(10))


def cluster_plot(d, k):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 4))
    for j in range(0, k):
        plt.plot(data[d == j], [j for i in d[d == j]], 'o')

    plt.ylim(-0.5, k - 0.5)
    return plt


# cluster_plot(d1, 5).show()


# #
#
#
# def RandomSampling(dataMat,number):
#     try:
#          slice = random.sample(dataMat, number)
#          return slice
#     except:
#          print('sample larger than population')
#
#
#
# print(RandomSampling(train_df, 2))



# 随机抽样
# print(combine_df.sample(n=10, random_state=1, axis=0))