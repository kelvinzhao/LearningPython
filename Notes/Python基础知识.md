# Python 学习笔记


*Python* 特点：
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

## 输出

`print()`  函数，单引号双引号都可以，但不能混用。

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

Python3 中有六个标准的数据类型：
- Number（数字）
- String（字符串）
- List（列表）
- Tuple（元组）
- Set（集合）
- Dictionary（字典）

其中，
- 不可变数据为：Number、String、Tuple
- 可变数据为：List、Set、Dictionary



```python
a = 123 # a是整数
print(a)
a = 'ABC' # a变为字符串
print(a)
```

## Py2 和 Py3 在整数除法的区别

Python2 输出的是整数，Python3 总是输出浮点类型，例如：
```shell
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

## 字符串索引

Python 字符串有两种索引方法，从左往右以0开始，从右往左以-1开始。

以下代码实现字符串反转

```python
def reverseWords(input):
	# 通过空格将字符串分割，把各个单词分割为列表
	inputWords = input.split("")

	# 反转字符串
	# 假设立标 list = [1,2,3,4],
	# list[0] = 1, list[1] = 2, 而 -1 表示最后一个元素 list[-1] = 4 (与 list[3]=4 一样)
	# inputWords[-1::-1]有三个参数
	# 第一个参数 -1 表示最后一个元素
	# 第二个参数为空，表示移动到列表末尾
	# 第三个参数为步长，-1 表示逆向
	inputWords = inputWords[-1::-1]

	# 重新组合字符串
	output = ' '.join(inputWords)

	return output


if __name__ == "__main__":
	input = 'I like runoob'
	rw = reverseWords(input)
	print(rw)

```
输出结果为： `runoob like I`

## 赋值运算符

//=		取整除复制运算符（向下取接近商的整数）
例如：9//2 = 4； -9//2 = -5
:= 		海象运算符，可在表达式内部为变量赋值
例如：
```python
if (n := len(a)) > 10:
	print(f"List is too long ({n} elements, expected <=10)")
```
## 逻辑运算符
`and / or / not`

## 成员运算符
`in / not in`
可用于字符串、列表以及元组

## 身份运算符

is / is not  x is y, 类似id(x)==id(y)

id()函数用于获取对象内存地址。

## 数字类型转换
```
int(x)
float(x)
complex(x)
```
交互模式下，最后被输出的表达式结果被赋值给变量 `_`


## 常用数学函数

* abs(x) 	返回数字的绝对值
* ceil(x)	返回数字的上入整数，例如math.ceil(4.1) 返回 5
* exp(x)	返回e的x次幂
* fabs(x)	返回浮点绝对值例如 math.fabs(-10) 返回 10.0
* floor(x)	返回数字的下舍整数，如 math.floor(4.9) 返回 4
* max(x1,x2,...)
* min(x1,x2,...)
* modf(x)	返回x的小数部分和整数部分，符号与x相同，整数部分以浮点表示。例如：

```powershells
In [1]: math.modf(4.123)
Out[1]: (0.12300000000000022, 4.0)
```
* pow(x,y)	返回x**y运算后的值
* round(x[,n])	返回浮点数x的四舍五入值，n为保留小数点后的位数

## 随机数函数

* choice(seq)		从序列的元素中随机挑选一个元素，例如`random.choice(range(10))`，从0到9中随机挑选一个整数
* randrange([start,]stop[,step])	从指定番位内，按指定基数递增的集合中获得一个随机数
* random()		随机生成下一个实数，它在[0,1]范围内
* shuffle(list)	将序列的所有元素随机排序
* uniform(x,y)	随机生成下一个实数，它在[x,y]范围内。

## 常用格式字符串
python 字符串格式化符号：

* %c		格式化字符及其ASCII码
* %s		格式化字符串
* %d		格式化整数
* %u		格式化无符号整数
* %o		格式化无符号八进制数
* %x		格式化无符号十六进制数
* %X		格式化无符号十六进制数（大写）
* %f		格式化浮点数字，可指定小数点后的精度
* %e		用科学计数法格式化浮点数

## Python 三引号

三引号允许一个字符串跨多行，并允许换行符、制表符及其他特殊字符。实例如下

```python
para_str = """这是一个多行字符串的实例
多行字符串可以使用制表符
TAB ( \t )。
也可以使用换行符 [ \n ]。
"""
print (para_str)
```
执行结果是
```
这是一个多行字符串的实例
多行字符串可以使用制表符
TAB (    )。
也可以使用换行符 [ 
 ]。
```

## f-string
f-string 格式化字符串以 f 开头，跟着字符串，其中的表达式用{}包起来。如下：
```python
>>> name = 'Runoob'
>>> f'Hello {name}'
'Hello Runoob'
>>> w = {'name':'Runoob','url':'www.runoob.com'}
>>> f'{w["name"]}: {w["url"]}'
'Runoob: www.runoob.com'
```
还可以通过使用`=`来拼接表达式与结果，例如：
```python
>>> x = 1
>>> print(f'{x+1=}')
'x+1=2'
```

## Python 列表函数 & 方法
函数：
* len(list)		返回列表元素个数
* max(list) / min(list)		返回列表元素最大值/最小值
* list(seq)		将元组转换为列表
方法：
* list.append(obj)		在列表末尾添加新的对象
* list.count(obj)		统计某个元素在列表中出现的次数
* list.extend(seq)		在列表末尾一次性追加另一个序列中的多个值，例如：
```python
list1 = ['Google','Runoob','Taobao']
list2 = list(range(5)) # 创建0-4 的列表
list1.extend(list2)		# 扩展列表
print("扩展后的列表：“，list1)
```
执行结果为：`['Google','Runoob','Taobao',0,1,2,3,4]`
* list.index(obj)		从列表中找出某个值第一个匹配项的索引位置
* list.pop(index=-1)	移除列表中的一个元素，默认最后一个元素，并返回该元素值
* list.remove(obj)		移除列表中某个值的第一个匹配项
* list.reverse()		反向列表中元素
* list.sort(key=None,reverse=False)	对原列表进行排序

例如：简单排序
```python
aList = ['Google','Runoob','Taobao','Facebook']
aList.sort()
print("List :", aList)
```
输出结果为：`List : ['Facebook','Google','Runoob','Taobao']`

降序排列
```python
vowels = ['e','a','u','o','i']
vowels.sort(reverse=True)
print('降序输出：',vowels)
```
输出结果为：`降序输出：['u','o','i','e','a']`

通过制定列表中的元素排序输出列表
```python
# 获取列表的第二个元素
def takeSecond(elem):
	return elem[1]


# 列表
random = [(2,2),(3,4),(4,1),(1,3)]

# 指定第二个元素排序
random.sort(key=takeSecond)

# 输出
print('排序列表：', random)
```
输出结果为：`排序列表：[(4,1),(2,2),(1,3),(3,4)]`

* list.clear()			清空列表
* list.copy()			复制列表

## 元组
- 不需要括号也可以
```python
>>> tup1 = (1,2,3,4,5)
>>> tup2 = ('Google','Runoob',1997,2000)
>>> tup3 = "a","b","c","d"
```

当元组中只包含一个元素时，需在元素后面添加逗号。
```python
>>> tup1 = (50)
>>> type(tup1)		# 不加逗号，类型为整数
<class 'int'>
```
```python
>>> tup1 = (50,)
>>> type(tup1)		# 加上逗号，类型为元组
<class 'tuple'>
```

元组中的元素不允许修改，但可以对元组进行连接组合。

元组中的元素也不允许删除。

元组可以跟列表一样索引和截取。

## 字典
- 不允许同一个键出现两次，否则后一个值会被记住
- 键必须不可变，可以用数字、字符串或元组充当，而不能用列表。

Python字典内置方法


| 序号 | 函数                              | 描述                                                                                  |
| ---- | --------------------------------- | ------------------------------------------------------------------------------------- |
| 1    | dict.clear()                      | 删除字典内所有元素                                                                    |
| 2    | dict.copy()                       | 返回一个字典的复制
| 3    | dict.fromkeys(seq[,val])          | 创建一个新字典，以序列seq中的元素作为字典的键，val为字典所有键对应的初始值
| 4    | dict.get(key,default=None)        | 返回指定键的值，如果值不在字典中则返回default的值
| 5    | key in dict                       | 如果键在字典dict里返回True，否则返回False
| 6    | dict.items()                      | 以列表返回可遍历的(键,值)元组数组
| 7    | dict.keys()                       | 返回一个迭代器，可以使用list()来转换为列表
| 8    | dict.setdefault(key,default=None) | 和get()类似，但如果键不存在字典中，将会添加键并将其值设为default
| 9    | dict.update(dict2)                | 把字典dict2的键值对儿更新到dict里
| 10   | dict.values()                     | 返回一个迭代器，可以使用list()来转换为列表
| 11   | pop(key[,default])                | 删除字典给定键key所对应的值，返回值为被删除的值。key值必须存在，否则返回default的值。
| 12   | popitem()                         | 随机返回并删除字典中的最后一对键值。

## 集合

集合是“无序”的不重复元素序列
创建空集合必须使用`set()`而不是`{}`，因为`{}`是用来创建空字典的。
```python
>>> basket = {'apple','orange','apple','pear','orange','banana'}
>>> print(basket)	# 这里演示的是去重功能
{'orange','banana','pear','apple'}
>>> 'orange' in basket	# 快速判断元素是否在集合内
True
>>> 'crabgrass' in basket
False
```
```python
# 下面展示两个集合间的运算
>>> a = set('abracadabra')
>>> b = set('alacazam')
>>> a
{'a', 'c', 'r', 'b', 'd'}
>>> a - b		# 集合a中包含而b中不包含的元素
{'b', 'r', 'd'}
>>> a | b		# 集合a或b中包含的元素，并集
{'b', 'm', 'r', 'l', 'z', 'a', 'c', 'd'}
>>> a & b		# 集合a和b中的交集
{'c', 'a'}
>>> a ^ b		# 不同时包含于a和b的元素，与非
{'l', 'b', 'd', 'm', 'r', 'z'}
```
类似列表的推导式，集合也支持推导式
```python
>>> a = {x for x in 'abracadabra' if x not in 'abc'}
>>> a
{'r','d'}
```
### 集合基本操作

1. 添加元素
`s.add(x)`
`s.update(x)` 可以添加列表，元组，字典等；x可以是多个，逗号分开

2. 移除元素
`s.remove(x)` 如果x不存在，则会报错
`s.discard(x)` 如果x不存在，不会报错
`s.pop()` 随机删除集合中的一个元素

3. 计算集合s元素个数
`len(s)`

4. 清空集合
`s.clear()`

