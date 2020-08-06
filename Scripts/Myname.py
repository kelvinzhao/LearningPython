'''
当文件被当作脚本执行时，__name__被解释器赋值为__main__
当文件被当作模块被import时，__name__被赋值为本模块名
'''

_a = 'name'


def my_abs(num):
    return num if num >= 0 else -num


def my_name():
    return __name__


if __name__ == '__main__':
    print(my_name())
