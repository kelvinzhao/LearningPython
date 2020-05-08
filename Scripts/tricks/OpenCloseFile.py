# Context Managers
# f = open('test.txt', 'r')
# file_contents = f.read()
# f.close()
# 由于缓存问题，通常 open 之后需要 close
# 而用with 结构则可以自动开和闭文件

with open('test.txt', 'r') as f:
    file_contents = f.read()
words = file_contents.split(' ')
word_count = len(words)
print(word_count)
