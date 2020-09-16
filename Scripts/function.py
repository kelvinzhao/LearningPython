# ----Parameter and argument----------------------------------------------------
def greet_user(first_name, last_name):    # 'name' here is "parameters"
    print(f'Hi there {first_name} {last_name}!')
    print('Welcome aboard')


print('Start')
greet_user('Mayer', 'John')  # 'John' <-- here is "arguments"
print('Finish')
# position arguments

greet_user(last_name='Mayer', first_name='John')
# keyword arguments, can improve code readablity
# but keyword arguments should always after position argument

# ----Return statement----------------------------------------------------------


def square(number):
    return number * number

# by default, Python return None if there is no return statement in function.
