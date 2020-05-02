Python 特点：

- 脚本语言，无需编译
- 逐行解释执行，运行速度慢（但用户体验不到）
- 代码无法加密，发布程序即发布代码

在 Mac 环境下可直接运行 Python 文件，需要在文件第一行加一个注释。

```python
#!/usr/bin/env python3
print('hello, world')
```

然后，在终端给 py 文件加执行权限即可。

```shell
$ chmod a+x hello.py
```

Python 程序是大小写敏感的。

---

## 输出

`print()​`  函数，单引号双引号都可以，但不能混用。

可以接受多个字符串，用逗号隔开。逗号会产生一个空格。例如：

```python
>>> print('The quick brown fox', 'jumps over', 'the lazy dog')
The quick brown fox jumps over the lazy dog
```

`print()` 可以打印整数，或者计算结果。例如：

```python
>>> print(300)
300
>>> print(100 + 200)
300
```

## 输入

`input()` 可以让用户输入字符串，并存放到一个变量里。例如输入用户的名字：

```python
>>> name = input()
Michael
```

完成输入后，查看变量。

```python
>>> name
'Michael'
```

`input()` 函数可以接受一个字符串作为输入时给用户的提示。例如：

```python
name = input('please enter your name: ')
print('hello,', name)
```

---

## 注释

但行注释以 `#` 开头

## 转义

`\n` 回车  `\t` 制表符  `\\`  字符 \ 本身

如果很多字符需要转义，可以用 `r''` 。例如：

```python
>>> print('\\\t\\')
\       \
>>> print(r'\\\t\\')
\\\t\\
```

如果字符串内有很多换行，则可以用 `'''...'''` 的格式来表示多行内容。例如：

```python
>>> print('''line1
... line2
... line3''')
line1
line2
line3
```

注意：`...` 是终端提示符，不是代码的一部分。

## 空值

空值是 Python 里一个特殊的值，用 `None` 表示。

## 变量

布尔值：`True` `False` ，注意大小写。

Python 是动态语言，变量类型可以随时变化。

```python
a = 123 # a是整数
print(a)
a = 'ABC' # a变为字符串
print(a)
```

## Py2 和 Py3 在整数除法的区别

Python2 输出的是整数，Python3 总是输出浮点类型，例如：

```powershell
$ python
>>> print(10/2)
5
$ python3
>>> print(10/2)
5.0
```

## Try Except

```python
astr = 'Hello Bob'
try:
  print('Hello')
  istr = int(astr)
  print('There')
except:
  istr = -1
print('Done', istr)
```

## def 

```python
def greet(lang):
  if lang == 'es':
    print('Hola')
  elif lang == 'fr'
  	print('Bonjour')
  else
  	print('Hello')
```

## while True

```python
while True:
    line = input('> ')
    if line[0] == '#':
        continue
    if line == 'done':
        break
    print(line)
print('Done')
```

## For Loop

```python
# for loop
for i in [5,4,3,2,1]:
    print(i)
print('Done')
```

