# Given an integer, return the integer with reversed digits.
# Note: The integer could be either positive or negative.

def solution(x: int) -> int:
    string = str(x)

    if string[0] == '-':
        return int('-'+string[:0:-1])
    else:
        return int(string[::-1])


print(solution(-231))
print(solution(345))

# A warm-up algorithm, that will help you practicing your slicing skills. In
# effect the only tricky bit is to make sure you are taking into account the
# case when the integer is negative. I have seen this problem presented in many
# different ways but it usually is the starting point for more complex
# requests.
