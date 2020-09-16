# Given k numbers which are less than n, return the set of prime number among them
# Note: The task is to write a program to print all Prime numbers in an Interval.
# Definition: A prime number is a natural number greater than 1 that has no positive divisors other
# than 1 and itself.

n = 37


def solution(n: int) -> list:
    prime_nums = []
    for num in range(n):
        if num > 1:
            for i in range(2, num):
                if (num % i) == 0:
                    break
            else:
                prime_nums.append(num)
    return prime_nums


print(solution(n))


# I wanted to close this section with another classic problem. A solution can be found pretty
# easily looping trough range(n) if you are familiar with both the prime numbers definition and the
# modulus operation.
