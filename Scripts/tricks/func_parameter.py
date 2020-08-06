
# 传不可变对象实例
# --------------------------
def ChangeInt(a: int):
    a = 10


b = 2
ChangeInt(b)
print(b)

# 实例中有int对象2，指向它的变量b，在传递给ChangeInt函数时，按照传值的方式复制了变量b，a和b都指向了
# 同一个int对象，在a=10时，则新生成一个int值对象10，并让a指向它。

# 传可变对象实例
# --------------------------


def changeme(mylist: list):
    mylist.append([1, 2, 3, 4])
    print(f'函数内取值：{mylist}')
    return


mylist = [10, 20, 30]
changeme(mylist)
print(f'函数外取值：{mylist}')

# 可变对象在函数里修改了参数，那么在调用这个函数的函数里，原始的参数也被改变了。


# 关键字参数
# --------------------------

def printinfo1(name, age):
    "打印任何传入的字符串"
    print("名字: ", name)
    print("年龄: ", age)
    return


# 调用printinfo函数
printinfo1(age=50, name="runoob")


# 默认参数
# --------------------------

def printinfo2(name, age=35):
    "打印任何传入的字符串"
    print("名字: ", name)
    print("年龄: ", age)
    return


# 调用printinfo函数
printinfo2(age=50, name="runoob")
print("------------------------")
printinfo2(name="runoob")

# 不定长参数
# --------------------------


def printinfo3(arg1, *vartuple):
    "打印任何传入的参数"
    print("输出：")
    print(arg1)
    print("接下来打印tuple")
    print(vartuple)
    print("打印tuple中的元素")
    print(*vartuple, sep='\n')


printinfo3(70, 60, 50)


def printinfo4(arg1, **vardict):
    print("输出：")
    print(arg1)
    print(vardict)


printinfo4(1, a=2, b=3)
