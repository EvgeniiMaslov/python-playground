# Uses python3
import sys

def get_change(m):
    #write your code here
    min_num_coins = [0] * (m+1)
    coins = [1, 3, 4]
    num = 0
    for j in range(1, m+1):
        min_num_coins[j] = 99999
        for i in range(len(coins)):
            if j >= coins[i]:
                num = min_num_coins[j - coins[i]] + 1
                if num < min_num_coins[j]:
                    min_num_coins[j] = num
    return min_num_coins[m]


if __name__ == '__main__':
    m = int(sys.stdin.read())
    print(get_change(m))
