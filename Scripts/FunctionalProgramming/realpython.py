# %%
# https://realpython.com/courses/functional-programming-python/
import itertools
from functools import reduce
import collections
from pprint import pprint

# import multiprocessing

# %%
# Map

scs = collections.namedtuple('Scientist', ['name', 'field', 'born', 'Nobel'])
scslist = (
        scs(name='Ada Lovelace', field='Math', born=1815, Nobel=False),
        scs(name='Emmy Noether', field='Math', born=1882, Nobel=False),
        scs(name='Marie Curie', field='Physics', born=1867, Nobel=True),
        scs('Tu youyou', 'Chemistry', 1930, True),
        scs('Ada Yonath', 'Chemistry', 1939, True),
        scs('Vera Rubin', 'Astronomy', 1928, False),
        scs('Sally Ride', 'Physics', 1951, False))

names_and_ages = tuple(map(
    lambda x: {'name': x.name, 'age': 2020 - x.born},
    scslist
    ))

names_and_ages2 = tuple({'name': x.name, 'age': 2020 - x.born} for x in scslist)

pprint(names_and_ages)
pprint(names_and_ages2)

# %%
# Reduce

total_age = reduce(
        lambda acc, val: acc + val['age'],
        names_and_ages,
        0)

total_age2 = reduce(
        lambda acc, val: acc + 2020 - val.born,
        scslist,
        0)

total_age3 = sum(x['age'] for x in names_and_ages)

print(total_age)
print(total_age2)
print(total_age3)

# %%
# try to group people by 'field'
template = {'Math': [], 'Physics': [], 'Chemistry': [], 'Astronomy': []}


def reducer(acc, val):
    '''
    this is docstring of reducer
    '''
    acc[val.field].append(val.name)
    return acc


scs_by_field = reduce(
        reducer, scslist, template)
pprint(scs_by_field)

# %%
# reduce & collections.defaultdict function

another_scs_by_field = reduce(
        reducer, scslist, collections.defaultdict(list))
pprint(another_scs_by_field)

# %%
# itertools.groupby

scs_by_field = {
        item[0]: list(item[1])
        for item in itertools.groupby(scslist, lambda x: x.field)
        }

pprint(scs_by_field)

# %%
# multiprocessing-testbed (Parallel)
# 数据量少的时候，下面这个map的执行过程会很快，但当处理复杂事物或数据量巨大时，
# map的效率就降低了，这里通过time.sleep()来模拟。
# 这里，realpython的教程使用multiprocessing.Pool来实现并行，但Scientist类不能
# 被Pickle，出现PicklingError，自己试验后，将Scientist改为常规的类，再加上对
# __name__ 的判断，成功了，见 realpython2.py 和 realpython3.py


# def transform(x):
#     print(f'Processing record {x.name}')
#     time.sleep(1)
#     result = {'name': x.name, 'age': 2020 - x.born}
#     print(f'Done processing record {x.name}')
#     return result


# if __name__ == '__main__':
#     start = time.time()

#     pool = multiprocessing.Pool()
#     result = pool.map(transform, scslist)

# # result = tuple(map(
# #     transform,
# #     scslist
# #     ))

#     end = time.time()

#     print(f'\nTime to complete: {end - start:.2f}s\n')
#     pprint(result)

# %%
