# %%
# https://realpython.com/courses/functional-programming-python/
import collections
from pprint import pprint

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
