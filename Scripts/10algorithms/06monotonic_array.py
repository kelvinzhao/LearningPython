# Given an array of integers, determine whether the array is monotonic or not.

A = [6, 5, 4, 4]
B = [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
C = [1, 1, 2, 3, 7]


def solution(nums):
    return (all(nums[i] <= nums[i+1] for i in range(len(nums)-1)) or
            all(nums[i] >= nums[i+1] for i in range(len(nums)-1)))


print(solution(A))
print(solution(B))
print(solution(C))

# This is another very frequently asked problem and the solution provided above
# is pretty elegant as it can be written as a one-liner. An array is monotonic
# if and only if it is monotone increasing, or monotone decreasing and in order
# to assess it, the algorithm above takes advantage # of the all() function that
# returns Trueif all items in an iterable are true, otherwise it returns False.
# If the iterable object is empty, the all() function also returns True.
