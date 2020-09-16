# For a given sentence, return the average word length.
# Note: Remember to remove punctuation first.

sentence1 = "Hi all, my name is Tom...I am originally from Australia."
sentence2 = "I need to work very hard to learn more about algorithms \
        in Python!"


def solution(sentence: str) -> round:
    for p in "!?',;.":
        sentence = sentence.replace(p, ' ')
    words = sentence.split()
    print(words)
    return round(sum(len(word) for word in words)/len(words), 2)


print(solution(sentence1))
print(solution(sentence2))

# Algorithms that require you to apply some simple calculations using strings
# are very common, therefore it is important to get familiar with methods like .
# replace() and .split()that in this case helped me removing the unwanted
# characters and create a list of words, the length of which can be easily
# measured and summed.
