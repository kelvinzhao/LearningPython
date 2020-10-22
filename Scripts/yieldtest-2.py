def yieldtest():
    a = 0
    while True:
        b = input('input a number: ')
        if b == 'q':
            return a  # break 或者 return 都会通过StopIteration异常终止迭代
        a += int(b)
        yield a


def proxy_test():
    while True:
        c = yield from yieldtest()
        print(c)


if __name__ == '__main__':
    func = proxy_test()
    # 通过yield from进行代理，yield from会处理exception并可通过return返回值
    print(next(func))
    print(next(func))
    print(next(func))
    print(next(func))
