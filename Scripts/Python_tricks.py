#  %%
# Ternary Conditionals
condition = True
x = 1 if condition else 0
print(x)

# %%
# Underscore Placeholders
# num1 = 10000000000
# num2 = 100000000
# total = num1+num2
# print(total)

# do this
num1 = 10_000_000_000
num2 = 100_000_000
total = num1+num2
print(f'{total:,}')

# %%
# Context Managers
# f = open('test.txt', 'r')
# file_contents = f.read()
# f.close()
# 由于缓存问题，通常 open 之后需要 close
# 而用with 结构则可以自动开和闭文件

with open('test.txt', 'r') as f:
    file_contents = f.read()
words = file_contents.split(' ')
word_count = len(words)
print(word_count)

# %%
# Enumerate
names = ['Corey', 'Chris', 'Dave', 'Travis']
# index = 0
# for name in names:
#     print(index, name)
#     index += 1
#
for index, name in enumerate(names):
    print(index, name)

# %%
# zip
names = ['Peter Parker', 'Clark Kent', 'Wad Wilson', 'Bruce Wayne']
heroes = ['Spiderman', 'Superman', 'Deadpool', 'Batman']

# for index, name in enumerate(names):
#     hero = heroes[index]
#     print(f'{name} is actually {hero}')
for name, hero in zip(names, heroes):
    print(f'{name} is actually {hero}')

# %%
