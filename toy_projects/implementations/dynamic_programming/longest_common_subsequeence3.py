#Uses python3

import sys

def lcs3(a, b, c):
    #write your code here
    m, n, d = len(a), len(b), len(c)
    dp = [[[0 for x in range(d+1)] for x in range(n + 1)] for x in range(m + 1)]

    for i in range(m+1):
        for j in range(n+1):
            for k in range(d+1):
                if i == 0 or j == 0 or k == 0:
                    dp[i][j][k] == 0
                elif a[i-1] == b[j-1] == c[k-1]:
                    dp[i][j][k] = dp[i-1][j-1][k-1]+1
                else:
                    dp[i][j][k] = max(dp[i-1][j][k], dp[i][j-1][k], dp[i][j][k-1])

    return dp[m][n][k]

if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    an = data[0]
    data = data[1:]
    a = data[:an]
    data = data[an:]
    bn = data[0]
    data = data[1:]
    b = data[:bn]
    data = data[bn:]
    cn = data[0]
    data = data[1:]
    c = data[:cn]
    print(lcs3(a, b, c))
