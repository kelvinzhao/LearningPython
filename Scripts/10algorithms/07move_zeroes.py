# Given an array nums, write a function to move all zeroes to the end of it while maintaining the
# relative order of the non-zero elements.

array1 = [0, 1, 0, 3, 12]
array2 = [1, 7, 0, 0, 8, 0, 10, 12, 0, 4]


def solution(nums: list) -> list:
    for i in nums:
        if 0 in nums:
            nums.remove(0)
            nums.append(0)
    return nums


print(solution(array1))
print(solution(array2))


# When you work with arrays, the .remove() and .append() methods are precious allies. In this
# problem I have used them to first remove each zero that belongs to the original array and then
# append it at the end to the same array.
