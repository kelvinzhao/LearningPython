# %%
# https://realpython.com/courses/functional-programming-python/
# map / reduce / parallel processing
import itertools
from functools import reduce
import collections
from pprint import pprint

# import multiprocessing

scs = collections.namedtuple('Scientist', ['name', 'field', 'born', 'Nobel'])
scslist = (
        scs(name='Ada Lovelace', field='Math', born=1815, Nobel=False),
        scs(name='Emmy Noether', field='Math', born=1882, Nobel=False),
        scs(name='Marie Curie', field='Physics', born=1867, Nobel=True),
        scs('Tu youyou', 'Chemistry', 1930, True),
        scs('Ada Yonath', 'Chemistry', 1939, True),
        scs('Vera Rubin', 'Astronomy', 1928, False),
        scs('Sally Ride', 'Physics', 1951, False))

total_age2 = reduce(
        lambda acc, val: acc + 2020 - val.born,
        scslist,
        0)

print(total_age2)

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


def transform(x):
    print(f'Processing record {x.name}')
    result = {'name': x.name, 'age': 2020 - x.born}
    print(f'Done processing record {x.name}')
    return result


result = tuple(map(
    transform,
    scslist
    ))


pprint(result)
