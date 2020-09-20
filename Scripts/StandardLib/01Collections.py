import collections
from pprint import pprint


def isNobel(x):
    return x.Nobel is False


scs = collections.namedtuple('Scientist', ['name', 'field', 'born', 'Nobel'])
scslist = (
        scs(name='Ada Lovelace', field='Math', born=1815, Nobel=False),
        scs(name='Emmy Noether', field='Math', born=1882, Nobel=False),
        scs(name='Marie Curie', field='Physics', born=1867, Nobel=True),
        scs('Tu youyou', 'Chemistry', 1930, True),
        scs('Ada Yonath', 'Chemostry', 1939, True),
        scs('Vera Rubin', 'Astronomy', 1928, False),
        scs('Sally Ride', 'Physics', 1951, False))
pprint(scslist)

nobeltruelist = tuple(filter(lambda x: x.Nobel is True, scslist))
nobelfalselist = tuple(filter(isNobel, scslist))
pprint(nobeltruelist)
pprint(nobelfalselist)
