# %%
# https://realpython.com/courses/functional-programming-python/
import collections
from pprint import pprint
import time
import multiprocessing
import os

# %%
# Map
# 下面这里，变量必须和namedtuple中的class name一致才能被pickle
Scientist = collections.namedtuple('Scientist', ['name', 'field', 'born', 'Nobel'])
scientlists = (
        Scientist(name='Ada Lovelace', field='Math', born=1815, Nobel=False),
        Scientist(name='Emmy Noether', field='Math', born=1882, Nobel=False),
        Scientist(name='Marie Curie', field='Physics', born=1867, Nobel=True),
        Scientist(name='Tu youyou', field='Chemistry', born=1930, Nobel=True),
        Scientist(name='Ada Yonath', field='Chemistry', born=1939, Nobel=True),
        Scientist(name='Vera Rubin', field='Astronomy', born=1928, Nobel=False),
        Scientist(name='Sally Ride', field='Physics', born=1951, Nobel=False)
        )


def transform(x):
    print(f'Process {os.getpid()} working record {x.name}')
    time.sleep(1)
    result = {'name': x.name, 'age': 2020 - x.born}
    print(f'Process {os.getpid()} done processing record {x.name}')
    return result


if __name__ == '__main__':
    start = time.time()
    pool = multiprocessing.Pool(processes=2, maxtasksperchild=1)
    result = pool.map(transform, scientlists)
    end = time.time()
    print(f'\nTime to complete: {end - start:.2f}s\n')
    pprint(result)
# %%
