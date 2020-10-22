def double_inputs():
    while True:
        x = yield
        yield x * 2


gen = double_inputs()
next(gen)       # run up to the first yield
print(gen.send(10))    # goes into 'x' variable
next(gen)       # run up to the next yield
print(gen.send(6))     # goes into 'x' again
next(gen)       # run up to the next yield
print(gen.send(94.3))  # goes into 'x' again
