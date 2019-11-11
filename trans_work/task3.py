# def mybin(num):
#     a = bin(num)
#     a = a.replace('0b', '')
#     return a
def sgn(num):
    if num < 0:
        return 0
    else:
        return 1


def input_func(m, n):
    result = m ^ n
    return result


if __name__ == "__main__":
    a = int(input())
    b = int(input())
    if a in range(7):
        print(a)
    if b in range(7):
        print(b)
    num = input_func(a, b)
    print("%s和%s的异或结果为%s." % (a, b, num))
