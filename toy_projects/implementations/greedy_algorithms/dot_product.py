#Uses python3

import sys

def max_dot_product(a, b):
    #write your code here
    res = 0
    for _ in range(len(a)):
        index1 = a.index(max(a))
        index2 = b.index(max(b))
        res += a[index1] * b[index2]
        a.pop(index1)
        b.pop(index2)
    return res

if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n = data[0]
    a = data[1:(n + 1)]
    b = data[(n + 1):]
    print(max_dot_product(a, b))
    
