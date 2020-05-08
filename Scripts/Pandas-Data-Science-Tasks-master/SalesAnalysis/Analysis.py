#!/usr/bin/env python
# coding: utf-8

# #### import Necessary Libraries

# In[1]:


import pandas as pd
import os


# #### Merging 12 months of sales data into a single file

# In[2]:


df = pd.read_csv("./Sales_Data/Sales_April_2019.csv")
files = [file for file in os.listdir('./Sales_Data') if not file.startswith('.')]
all_months_data = pd.DataFrame()
for file in files:
    df = pd.read_csv("./Sales_Data/"+ file)
    all_months_data = pd.concat([all_months_data, df])
    
all_months_data.to_csv("all_data.csv", index=False)


# #### Read in updated dataframe

# In[3]:


all_data = pd.read_csv("all_data.csv")
all_data.head()


# #### drop NaN

# In[4]:


nan_df = all_data[all_data.isna().any(axis=1)]
nan_df.head()
all_data = all_data.dropna(how='all')
all_data.head()


# #### drop 'Or'

# In[5]:


temp_data = all_data[all_data['Order Date'].str[0:2] !='Or']
all_data = temp_data


# #### Convert columns to the correct type

# In[6]:


all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered']) # make int.
all_data['Price Each'] = pd.to_numeric(all_data['Price Each']) # make float


# #### Augment data with additional columns

# #### Task2: add month column

# In[7]:


all_data['Month'] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype('int32')
all_data.head()


# #### Task 3: add a sales column

# In[8]:


all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']
all_data.head()


# #### Task 4: Add a city column

# In[9]:


# Let's use .apply()
def get_city(address):
    return address.split(', ')[1]
def get_state(address):
    return address.split(', ')[2].split(' ')[0]

all_data['City'] = all_data['Purchase Address'].apply(lambda x: f"{get_city(x)} ({get_state(x)})")
#all_data.drop(columns='Column',inplace=True)
all_data.head()


# #### Question 1: What was the best month for sales? How much was earned that month?

# In[10]:


results = all_data.groupby('Month').sum()


# In[11]:


import matplotlib.pyplot as plt
months = range(1,13)
plt.bar(months,results['Sales'])
plt.xticks(months)
plt.ylabel('Sales in USD($)')
plt.xlabel('Month number')
plt.show()


# #### Question 2: What city had the highest number of sales?

# In[12]:


results = all_data.groupby('City').sum()
results


# In[13]:


import matplotlib.pyplot as plt
cities = [city for city, df in all_data.groupby('City')]

plt.bar(cities,results['Sales'])
plt.xticks(cities,rotation = 'vertical',size=8)
plt.ylabel('Sales in USD($)')
plt.xlabel('City name')
plt.show()


# #### Question 3: What time should we display advertisements to maximize likelihood of customer's buying product?

# In[14]:


# first, change 'Order Date' to date format
all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])


# In[15]:


# exact Hour and Min into new columns
all_data['Hour'] = all_data['Order Date'].dt.hour
all_data['Minute'] = all_data['Order Date'].dt.minute
all_data.head()


# In[16]:


hours = [hour for hour,df in all_data.groupby('Hour')]
plt.plot(hours, all_data.groupby(['Hour']).count())
plt.xticks(hours)
plt.xlabel('Hour')
plt.ylabel('Numbers of Orders')
plt.grid()
plt.show()
# My recommendation is around 11am or 7pm


# #### Question 4: What products are most often sold together?

# In[17]:


df = all_data[all_data['Order ID'].duplicated(keep=False)]
df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
df = df[['Order ID','Grouped']].drop_duplicates()
df.head(20)


# In[18]:


# Referenced: https://stackoverflow.com/questions/52195887/counting-unique-pairs-of-numbers-into-a-python-dictionary
from itertools import combinations
from collections import Counter
count = Counter()
for row in df['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list,2)))
for key, value in count.most_common(10):
    print(key,value)


# #### Question 5: What product sold the most? Why do you think it sold the most?

# In[19]:


product_group = all_data.groupby('Product')
products = [product for product, df in product_group]
quantity_ordered = product_group.sum()['Quantity Ordered']

plt.bar(products,quantity_ordered)
plt.xticks(products,rotation='vertical')
plt.xlabel('Product')
plt.ylabel('Quantity Ordered')
plt.show()


# In[20]:


price = all_data.groupby('Product').mean()['Price Each']

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(products,quantity_ordered)
ax2.plot(products,price,'r-')
plt.grid()
ax1.set_xlabel('Product')
ax1.set_ylabel('Quantity Ordered')
ax2.set_ylabel('Price')
ax1.set_xticklabels(products,rotation='vertical')
plt.show()


# In[ ]:




