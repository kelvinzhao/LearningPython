# %%
# https://realpython.com/courses/functional-programming-python/
import collections
from pprint import pprint
import time
import multiprocessing

# %%
# Map


class Scientist:

    def __init__(self, name, field, born, Nobel):
        self.name = name
        self.field = field
        self.born = born
        self.Nobel = Nobel

    def getNameAndAge(x):
        print(f'Processing record {x.name}')
        time.sleep(1)
        result = {'name': x.name, 'age': 2020 - x.born}
        print(f'Done processing record {x.name}')
        return result


scslist = (
        Scientist(name='Ada Lovelace', field='Math', born=1815, Nobel=False),
        Scientist(name='Emmy Noether', field='Math', born=1882, Nobel=False),
        Scientist(name='Marie Curie', field='Physics', born=1867, Nobel=True),
        Scientist(name='Tu youyou', field='Chemistry', born=1930, Nobel=True),
        Scientist(name='Ada Yonath', field='Chemistry', born=1939, Nobel=True),
        Scientist(name='Vera Rubin', field='Astronomy', born=1928, Nobel=False),
        Scientist(name='Sally Ride', field='Physics', born=1951, Nobel=False)
        )
# scs = collections.namedtuple('Scientist', ['name', 'field', 'born', 'Nobel'])
# scslist = (
#         scs(name='Ada Lovelace', field='Math', born=1815, Nobel=False),
#         scs(name='Emmy Noether', field='Math', born=1882, Nobel=False),
#         scs(name='Marie Curie', field='Physics', born=1867, Nobel=True),
#         scs(name='Tu youyou', field='Chemistry', born=1930, Nobel=True),
#         scs(name='Ada Yonath', field='Chemistry', born=1939, Nobel=True),
#         scs(name='Vera Rubin', field='Astronomy', born=1928, Nobel=False),
#         scs(name='Sally Ride', field='Physics', born=1951, Nobel=False)
#         )

if __name__ == '__main__':
    pprint(scslist)

    start = time.time()

    pool = multiprocessing.Pool()
    result = pool.map(Scientist.getNameAndAge, scslist)

# result = tuple(map(
#     getNameAndAge,
#     scslist
#     ))

    end = time.time()

    print(f'\nTime to complete: {end - start:.2f}s\n')
    pprint(result)

# %%
