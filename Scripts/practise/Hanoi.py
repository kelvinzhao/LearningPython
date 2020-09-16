'''
要目标盘的上方有一个更小的盘，则需要执行move策略，因此构成递归
设目标盘是第disk个盘，要从原位置移动到目标位置，那么它上面的第disk-1盘就必须先从原位置移动到辅助位
再将第disk盘从原位置移动到目标位，最后将第disk-1盘从辅助位移动到目标位
实际上在递归过程中，A、B、C是轮番成为原位置、目标位和辅助位的。
'''


def hmove(disk: int, _from: str, _pass: str, _target: str,
         counts: int = 0) -> int:
    if disk == 1:
        counts += 1
        print("Step ", counts, " : move disk ",
              disk, ' from ', _from, ' to ', _target)
    else:
        counts = hmove(disk-1, _from, _target, _pass, counts)
        counts += 1
        print("Step ", counts, " : move disk ",
              disk, ' from ', _from, ' to ', _target)
        counts = hmove(disk-1, _pass, _from, _target, counts)
    return counts


if __name__ == '__main__':
    print('Total : ',  hmove(4, 'A', 'B', 'C'))
