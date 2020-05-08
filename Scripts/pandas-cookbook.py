# cookbook from https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html
# %%
import numpy as np
import pandas as pd

# %%
# if-then
df = pd.DataFrame({'AAA': [4, 5, 6, 7],
                   'BBB': [10, 20, 30, 40],
                   'CCC': [100, 50, -30, -30]})
df
# %% An if-then on one column
df.loc[df.AAA >= 5, 'BBB'] = -1
df
# %% An if-then with assignment to 2 columns:
df.loc[df.AAA >= 5, ['BBB', 'CCC']] = 555
df
# %% Add another line with different logic, to do the -else
df.loc[df.AAA < 5, ['BBB', 'CCC']] = 2000
df

# %% Or use pandas where after you’ve set up a mask
df_mask = pd.DataFrame({'AAA': [True] * 4,
                        'BBB': [False] * 4,
                        'CCC': [True, False] * 2})
df.where(df_mask, -1000)

# %%
df = pd.DataFrame({'AAA': [4, 5, 6, 7],
                   'BBB': [10, 20, 30, 40],
                   'CCC': [100, 50, -30, -50]})
df
df['logic'] = np.where(df['AAA'] > 5, 'high', 'low')
df
# %%
# Split
# Split a frame with a boolean criterion
# https://stackoverflow.com/questions/14957116/how-to-split-a-dataframe-according-to-a-boolean-criterion

df
d = dict(list(df.groupby(df["logic"] != "low")))
d[True]
d[False]
# %% another way to do above
m = df['logic'] != 'low'
a, b = df[m], df[~m]
df
# %% or this way
d = df.groupby(df["logic"] != "low")
d.get_group(True)
d.get_group(False)
# %%
# Building criteria
# Select with multi-column criteria
# https://stackoverflow.com/questions/15315452/selecting-with-complex-criteria-from-pandas-dataframe
df = pd.DataFrame({'AAA': [4, 5, 6, 7],
                   'BBB': [10, 20, 30, 40],
                   'CCC': [100, 50, -30, -50]})

df
# and (without assignment returns a Series)
df.loc[(df['BBB'] < 25) & (df['CCC'] >= -40), 'AAA']
# or (without assignment returns a Series)
df.loc[(df['BBB'] > 25) | (df['CCC'] >= -40), 'AAA']
# or (with assignment modifies the DataFrame.)
df.loc[(df['BBB'] > 25) | (df['CCC'] >= 75), 'AAA'] = 0.1
df
# %%
# Select rows with data closest to certain value using argsort
# https://stackoverflow.com/questions/17758023/return-rows-in-a-dataframe-closest-to-a-user-defined-number
df = pd.DataFrame({'AAA': [4, 5, 6, 7],
                   'BBB': [10, 20, 30, 40],
                   'CCC': [100, 50, -30, -50]})
df
aValue = 43.0
df.loc[(df.CCC - aValue).abs().argsort()]
# argsort函数返回的是数组值从小到大的索引值

# %% build a DataFrame with defined order columns.
data = {'name': ["张三", "李四", "王五", "马六"],
        "student_id": [101001, 101002, 101003, 101004],
        "scores": [30, 42, 110, 150]}
df = pd.DataFrame(data, columns=["student_id", "name", "scores"],
                  index=["1", "2", "3", "4"])
print(df)
# %%
