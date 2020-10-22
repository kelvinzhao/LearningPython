def yieldtest():
    a = 0
    while True:
        b = input('input a number: ')
        if b == 'q':
            break  # break 或者 return 都会通过StopIteration异常终止迭代
        a += int(b)
        yield a


def proxy_test():
    while True:
        yield from yieldtest()


if __name__ == '__main__':
    func = proxy_test()  # 通过yield from进行代理，yield from会处理exception
    # func = yieldtest()  # 如果直接创建生成器，
    # 则需要自己处理当输入q时所产生的StopIteration异常。
    print(next(func))
    print(next(func))
    print(next(func))
    print(next(func))
