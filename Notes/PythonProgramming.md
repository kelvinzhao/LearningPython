# Python3 编程第一步

# 条件控制与循环语句

# Python3迭代器与生成器
迭代器对象从集合的第一个元素开始访问，直到所有的元素都被访问结束，只往前不后退。

迭代器有两个基本方法：`iter()` 和 `next()`

```python
>> > list = [1, 2, 3, 4]
>> > it = iter(list)  # 创建迭代器对象
>> > print(next(it))  # 输出迭代器的下一个元素
1
>> > print(next(it))
2
```
