def yieldtest():
    a = 0
    while True:
        b = input('input a number: ')
        if b == 'q':
            break  # break 或者 return 都会通过StopIteration异常终止迭代
        a += int(b)
        yield a
    return a


def proxy_test():
    while True:
        c = yield from yieldtest()
        print("c is ", c)


if __name__ == '__main__':
    func = proxy_test()
    func_yield = yieldtest()
    # 使用list获得最终结果时，好像不会产生StopIteration异常？
    # print(list(func_yield))
    # print(list(func))
    print(next(func_yield))
    print(next(func_yield))
    print(next(func_yield))
