# Uses python3
import sys

def binary_search(a, key, low=None, high=None):
    n = len(a) - 1
    
    if low == None and high == None:
        low = 0
        high = n

    mid = (low + high) // 2
    if low > high:
        return -1

    if key == a[mid]:
        return mid
    elif key < a[mid]:
        return binary_search(a, key, low, mid-1)
    else:
        return binary_search(a, key, mid+1, high)

if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n = data[0]
    m = data[n + 1]
    a = data[1 : n + 1]
    for x in data[n + 2:]:
        print(binary_search(a, x), end = ' ')
