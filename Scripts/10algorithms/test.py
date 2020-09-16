# %%
# 01 reverse user input string


def reverseint(astring: str):
    try:
        _result = int(astring)
        if astring[0] == '-':
            _result = int("-" + astring[:0:-1])
        else:
            _result = int(astring[::-1])
        return _result
    except ValueError:
        print("invalid input, try again.")
        return None


while True:
    userinput = input("please input an integer number ")
    result = reverseint(userinput)
    if result is not None:
        print(f"the reverse number is {result}")
        break
# %%
# 02 input a sentense, caculate average length of words


def avglen(sentense: str):
    for _p in """",.<>;':"/?[]{}\-=_+`~!@#$%^&*()""":
        if _p in sentense:
            sentense = sentense.replace(_p, ' ')
    _wordlist = sentense.split()
    return round(sum(len(_x) for _x in _wordlist)/len(_wordlist), 2)


userinput = input("please input a sentense: ")
result = avglen(userinput)
print(f"the reverse result is {result}")
# %%
# 03 add two string numbers


def add2strnumber(num1: str, num2: str):
    try:
        return(eval(num1) + eval(num2))
    except NameError:
        print("invalid input")
        return None


try:
    number1, number2 = input("input two numbers :").split()
    print('The sum is', add2strnumber(number1, number2))
except ValueError:
    print("not enough numbers")

# %%
# 04 Given a string, find the first non-repeating character in it and return its
# index. If it doesn't exist, return -1.
# Note: all the input strings are already lowercase.
from collections import Counter


def nonrepeatstr(string: str):
    counter = Counter(string)
    for _id, _char in enumerate(string):
        if counter[_char] == 1:
            return _id
    return -1


nonrepeatstr('abracadabra')

# %%
# 05 Given a non-empty string s, you may delete at most one character. Judge
# whether you can make it a palindrome. The string will only contain lowercase
# characters a-z.


def isPalindrome(string: str) -> bool:
    # step3 : index+1, loop again till the end
    for idx in range(len(string)):
        # step1 : remove one char to get a new string
        new_str = ''.join([string[i] for i in range(len(string)) if i != idx])
        # step2 : check whether the new string is a palindrome
        if new_str == new_str[::-1]:
            print(new_str)
            return True
        else:
            pass
    return False


astring = input('pleas enter a sentense: ')
print(isPalindrome(astring))

# %%
# 06 Given an array of integers, determine whether the array is monotonic or not


def isMonotonic(alist: list) -> bool:
    return (all(alist[i] <= alist[i+1] for i in range(len(alist)-1)) or
            all(alist[i] >= alist[i+1] for i in range(len(alist)-1)))


A = [1, 2, 3, 4, 5]
B = [5, 4, 3, 3, 1]
C = [2, 3, 1, 4, 5]
D = [5, 4, 4, 1, 2]
print(isMonotonic(A), isMonotonic(B), isMonotonic(C), isMonotonic(D))

# %%
# 07 Given an array nums, write a function to move all zeroes to the end of it
# while maintaining the relative order of the non-zero elements.


def moveZero(alist: list) -> list:
    result = []
    while True:
        try:
            alist.remove(0)
            result.append(0)
        except ValueError:
            return alist + result


print(moveZero([1, 0, 2, 0, 4, 0]))


# %%
# 08 Given an array containing None values fill in the None values with most
# recent non None value in the array

# idea: traverse the array, if found a 'None', check the previous and the next,
#       change 'None' to it if any is non-None. Else leave it there. The out
#       level should be a while loop, condition is 'not all non-None'.


def getPandN(array: list, idx: int):
    if idx == 0:
        return None if array[1] is None else array[1]
    elif idx == len(array) - 1:
        return None if array[idx-1] is None else array[idx-1]
    else:
        return array[idx-1] if array[idx-1] is not None else array[idx+1]


array = [1, None, None, None, None, 2, None, None, None, 3, 4, None]
_array = array.copy()
while not all(x is not None for x in array):
    for i in range(len(array)):
        if array[i] is None:
            print(_array)
            _array[i] = getPandN(array, i)
    array = _array.copy()

array

# %%
# 09 Given two sentences, return an array that has the words that appear in one
# sentence and not the other and an array with the words in common.


# step 1: build set for each sentence.
# step 2: use operation of set to figure out the results.

sentence1 = "a b c d e"
sentence2 = "a f d g i"


def findSingle(s1: str, s2: str) -> list:
    _set_1 = set(s1.split())
    _set_2 = set(s2.split())
    return list(_set_1 ^ _set_2), list(_set_1 & _set_2)


print(findSingle(sentence1, sentence2))

# %%
# 10 Given k numbers which are less than n, return the set of prime number among
# them
# Note: The task is to write a program to print all Prime numbers in an Interval.
# Definition: A prime number is a natural number greater than 1 that has no
# positive divisors other than 1 and itself.


n = 37


def Primenum(n: int) -> list:
    prime_nums = []
    if n <= 0:
        return None
    elif n == 1:
        return None
    elif n == 2:
        return [2]
    else:
        prime_nums = [2]
        for _num in range(3, n+1):
            for i in range(2, _num):
                if _num % i == 0:
                    break
            else:
                prime_nums.append(_num)
    return prime_nums


print(Primenum(n))
