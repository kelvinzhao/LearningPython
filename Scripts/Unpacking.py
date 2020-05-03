# Unpacking
a, _ = (1, 2)
print(a)

a, b, *c = (1, 2, 3, 4, 5)
print(a)
print(b)
print(c)

a, b, *_ = (1, 2, 3, 4, 5)
print(a)
print(b)

a, b, *c, d = (1, 2, 3, 4, 5)
print(a)
print(b)
print(c)
print(d)
