# Given an array containing None values fill in the None values with most recent
# non None value in the array

array1 = [1, None, 2, 3, None, None, 5, None]


def solution(array: list) -> list:
    valid = 0
    res = []
    for i in array:
        if i is not None:
            res.append(i)
            valid = i
        else:
            res.append(valid)
    return res


print(solution(array1))


# I was asked to solve this problem a couple of times in real interviews, both
# times the solution had to include edge cases (that I omitted here for
# simplicity). On paper, this an easy algorithm to build but you need to have
# clear in mind what you want to achieve with the for loop and if statement and
# be comfortable working with None values.
