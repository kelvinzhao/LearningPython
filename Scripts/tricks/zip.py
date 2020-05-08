# zip
names = ['Peter Parker', 'Clark Kent', 'Wad Wilson', 'Bruce Wayne']
heroes = ['Spiderman', 'Superman', 'Deadpool', 'Batman']

# for index, name in enumerate(names):
#     hero = heroes[index]
#     print(f'{name} is actually {hero}')
for name, hero in zip(names, heroes):
    print(f'{name} is actually {hero}')

# zip and unpack(tuple)
for value in zip(names, heroes):  # zip
    print(value)  # unpack
