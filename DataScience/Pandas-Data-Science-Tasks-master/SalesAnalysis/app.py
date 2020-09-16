# %%
from collections import Counter
from itertools import combinations
import pandas as pd
from pandas import Series, DataFrame
import os
import glob
import matplotlib.pyplot as plt

# combine 12 months data into one
# %%

datapath: str = os.path.dirname(__file__) + '/Sales_Data/'
all_data = DataFrame()
# for f in os.listdir(datapath):
#  if f.endswith('.csv'):
#    f = os.path.join(datapath,f)
#    all_data = pd.concat([all_data,pd.read_csv(f)])
# 遍历 csv 文件也可以用如下方法
for f in glob.glob(datapath + "*.csv"):
    all_data = pd.concat([all_data, pd.read_csv(f)])
all_data = all_data.reset_index(drop=True)
print(all_data.shape)
# %%
# 问题1: 哪个月份销售额最高，销售额是多少？

all_data['Order ID'] = pd.to_numeric(all_data['Order ID'], errors='coerce')
all_data['Quantity Ordered'] = pd.to_numeric(
    all_data['Quantity Ordered'], errors='coerce')
all_data['Price Each'] = pd.to_numeric(all_data['Price Each'], errors='coerce')
all_data = all_data.dropna(
    how='all', subset=['Order ID', 'Quantity Ordered', 'Price Each'])
all_data['Order ID'] = all_data['Order ID'].astype('int32')
all_data['Quantity Ordered'] = all_data['Quantity Ordered'].astype('int32')
all_data['Price Each'] = all_data['Price Each'].astype('int32')
print(all_data.shape)
all_data.head()
# %%
all_data['Month'] = all_data['Order Date'].str[0:2].astype('int32')
print(all_data.shape)
all_data.head()
all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']
months = range(1, 13)
month_sales = all_data.groupby('Month')['Sales'].sum()
print(all_data.shape)
all_data.head()
# %%
plt.bar(months, month_sales)
plt.xticks(months)
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()
# %%
# 问题2：哪个城市的销售额最高？

# %%


def get_city(address):
    return address.split(', ')[1]


def get_state(address):
    return address.split(', ')[2].split(' ')[0]


all_data['City'] = all_data['Purchase Address'].apply(
    lambda x: get_city(x)+","+get_state(x))
cities = [city for city, _ in all_data.groupby('City')]
city_sales = all_data.groupby('City')['Sales'].sum()
plt.bar(cities, city_sales)
plt.xticks(cities, rotation='vertical')
plt.xlabel('City')
plt.ylabel('Sales')
plt.grid(axis='x')
plt.show()
# %%
# 问题3：我们什么时间做广告比较好？
all_data['Hour'] = all_data['Order Date'].apply(
    lambda x: x.split(' ')[1][0:2]).astype('int32')
Hour_sales = all_data.groupby('Hour')['Sales'].sum()
Hours = range(0, 24)
plt.bar(Hours, Hour_sales)
plt.xticks(Hours)
plt.xlabel('Hour')
plt.ylabel('Sales in USD($)')
plt.grid(axis='x')
plt.show()
# %%
# 问题4：哪些产品经常一起卖出？
product_data = all_data[['Order ID', 'Product']].copy()
product_data = product_data[product_data['Order ID'].duplicated(keep=False)]
product_data['Product'] = product_data.groupby(
    'Order ID')['Product'].transform(lambda x: ','.join(x))
product_data = product_data.drop_duplicates()
print(product_data.shape)
product_data.head(20)
# %%
counter = Counter()
for row in product_data['Product']:
    row_list = row.split(',')
    counter.update(Counter(combinations(row_list, 2)))
for key, value in counter.most_common(10):
    print(key, value)


# %%
# 问题5：什么产品卖的最好，为什么？
quantities = all_data.groupby('Product')['Quantity Ordered'].sum()
price = all_data.groupby('Product')['Price Each'].mean()
products = quantities.index

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(products, quantities)
ax2.plot(products, price, 'r-')
ax1.set_axisbelow(True)
ax1.grid()
ax2.grid()
ax1.set_xticklabels(products, rotation='vertical')
ax1.set_xlabel('Product')
ax1.set_ylabel('Quantities of Order')
ax2.set_ylabel('Each Price')
plt.show()
# %%
