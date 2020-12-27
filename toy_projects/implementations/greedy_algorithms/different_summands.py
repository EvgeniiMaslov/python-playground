# Uses python3
import sys

def optimal_summands(n):
    summands = []
    
    remain = n
    i = 0
    while remain > 0:
        if remain - (i + i + 3) < 0:
            summands.append(remain)
            break
        else:
            summands.append(i + 1)
            remain -= (i+1)
            i += 1        
    return summands

if __name__ == '__main__':
    input = sys.stdin.read()
    n = int(input)
    summands = optimal_summands(n)
    print(len(summands))
    for x in summands:
        print(x, end=' ')
