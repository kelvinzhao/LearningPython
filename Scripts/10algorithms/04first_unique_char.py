# Given a string, find the first non-repeating character in it and return its
# index. it doesn't exist, return -1.
# Note: all the input strings are already lowercase.

import collections

# Approach 1


def solution(s):
    frequency = {}
    for i in s:
        if i not in frequency:
            frequency[i] = 1
        else:
            frequency[i] += 1
    for i in range(len(s)):
        if frequency[s[i]] == 1:
            return i
    return -1


print(solution('alphabet'))
print(solution('barbados'))
print(solution('crunchy'))

print('###')

# Approach 2


def solution2(s):
    # build hash map : character and how often it appears
    # it gives back a dictionary with words occurrence count
    count = collections.Counter(s)
    # Counter({'l': 1, 'e': 3, 't': 1, 'c': 1, 'o': 1, 'd': 1})
    # find the index
    for idx, ch in enumerate(s):
        if count[ch] == 1:
            return idx
    return -1


print(solution2('alphabet'))
print(solution2('barbados'))
print(solution2('crunchy'))


# Also in this case, two potential solutions are provided and I guess that, if
# you are pretty new algorithms, the first approach looks a bit more familiar as
# it builds as simple counter from an empty dictionary.

# However understanding the second approach will help you much more in the
# longer term and this is because in this algorithm I simply used
# collection.Counter(s)instead of building a chars counter myself and replaced
# range(len(s)) with enumerate(s), a function that can help you identify the
# index more elegantly.
