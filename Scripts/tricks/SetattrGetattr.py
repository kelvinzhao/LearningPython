# Setattr/Getattr
# 通过变量指定增加属性


class Person():
    pass


person = Person()

# person.first = "Corney"
# person.last ="Schafer"
# print(person.first)
# print(person.last)

first_key = 'first'
first_val = 'Corney'

setattr(person, first_key, 'Corney')
first = getattr(person, first_key)

print(person.first)
print(first)


person1 = Person()
person_info = {'first': 'Corney', 'last': 'Schafer'}

for key, value in person_info.items():
    setattr(person1, key, value)

# print(person1.first)
# print(person1.last)
for key in person_info.keys():
    # print(person1.__getattribute__(key)) # 这样行吗？
    print(getattr(person1, key))
