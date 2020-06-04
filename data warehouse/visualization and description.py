import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


origin_df = pd.read_csv('./data/origin.csv')

fact_df = pd.read_csv('./data/fact.csv')

# # 数据的粗略查看
print(fact_df.shape)
# print(fact_df.columns.values)
# print(fact_df.head())
# print(fact_df.tail())
# print(fact_df.info())
# print(fact_df.describe())
#
# # 检查是否有缺失值
# total = fact_df.isnull().sum().sort_values(ascending=False)
# print(total)
#
# # 查看各列信息
# # 查看商店id信息
# print(len(fact_df['shop_id'].unique()))
# print(fact_df['shop_id'].value_counts())
# print('*'*10)
# # 查看产品id信息

# # 做散点图检查异常值
# item_price_max = fact_df['item_price'].max()
# fact_df.plot.scatter(x='item_id', y='item_price', ylim=(0, item_price_max))
# plt.show()
# # 删除异常值
# print(fact_df.sort_values(by='item_price', ascending=False)[10:])
# fact_df = fact_df.drop(fact_df[fact_df['item_price'] > 100000].index) # 删除此行
# print(fact_df.sort_values(by='item_price', ascending=False)[10:])
# print(fact_df.shape)
# item_price_max = fact_df['item_price'].max()
# fact_df.plot.scatter(x='item_id', y='item_price', ylim=(0, item_price_max))
# plt.show()

# 查看data与item_cnt_day的关系
item_cnt_date_df = fact_df['item_cnt_day'].groupby(fact_df['date_id']).sum()
print(item_cnt_date_df)



data = pd.DataFrame(item_cnt_date_df)
print(data)

# 绘制折线图
plt.plot(item_cnt_date_df, square, linewidth=5, color='b') # 将列表传递给plot,并设置线宽，设置颜色，默认为蓝色
plt.title("Squares Number", fontsize=24, color='r')#设置标题，并给定字号,设置颜色
plt.xlabel("Value", fontsize=14, color='g')#设置轴标题，并给定字号,设置颜色
plt.ylabel("Squares Of Value", fontsize=14, color='g')
plt.tick_params(axis='both', labelsize=14)#设置刻度标记的大小
plt.show()







# print(len(fact_df['item_id'].unique()))
# print(fact_df['item_id'].value_counts())

# # 抽样3万行
# Random_fact = fact_df.sample(n=30000, random_state=None, axis=0)
# print(Random_fact)
# # with open('Random_fact.csv', 'a+') as f:
# #     for line in Random_fact:
# #         f.write(line)
# Random_fact.to_csv('Random_fact1.csv')