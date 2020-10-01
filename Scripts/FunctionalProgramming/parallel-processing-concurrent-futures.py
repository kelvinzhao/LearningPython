# https://realpython.com/lessons/parallel-processing-concurrent-futures-overview/
# Parallel Processing with concurrent.futures

import collections
from pprint import pprint
import os
import time
# import multiprocessing
import concurrent.futures


Scientist = collections.namedtuple('Scientist', ['name', 'field', 'born', 'Nobel'])
scientists = (
        Scientist(name='Ada Lovelace', field='Math', born=1815, Nobel=False),
        Scientist(name='Emmy Noether', field='Math', born=1882, Nobel=False),
        Scientist(name='Marie Curie', field='Physics', born=1867, Nobel=True),
        Scientist(name='Tu youyou', field='Chemistry', born=1930, Nobel=True),
        Scientist(name='Ada Yonath', field='Chemistry', born=1939, Nobel=True),
        Scientist(name='Vera Rubin', field='Astronomy', born=1928, Nobel=False),
        Scientist(name='Sally Ride', field='Physics', born=1951, Nobel=False))


def transform(x):
    print(f'Process {os.getpid()} working record {x.name}')
    time.sleep(3)
    res = {'name': x.name, 'age': 2020 - x.born}
    print(f'Process {os.getpid()} done processing record {x.name}')
    return res


if __name__ == '__main__':
    pprint(scientists)
    start = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # or     with concurrent.futures.ThreadPoolExecutor() as executor:
        # Thread based function 容易受到GIL的影响，全局解释器锁
        # 所以最好用Process，每个Process中都有一个解释器
        result = executor.map(transform, scientists)

    # pool = multiprocessing.Pool()
    # result = pool.map(transform, scientists)

    end = time.time()

    print(f'\nTime to complete: {end - start:.2f}s\n')
    pprint(tuple(result))
