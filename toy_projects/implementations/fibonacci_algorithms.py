import math
# return list of fibonacci values from 0 to n
def fib_list(n):
    lst = [0, 1]
    for i in range(2, n+1):
        lst.append(lst[i-1] + lst[i-2])
    return lst


def last_digit_period():
    period = 60
    lst = fib_list(period)
    remainders = [i%10 for i in lst]
    return remainders

# return last digit of n fibonacci value
def get_fibonacci_last_digit(n):
    remainders = last_digit_period()
    return remainders[n%60]


# the sequence fibonacci mod m is periodic,
# the period always starts with 01 and is known as Pisano period
def pisano_period(m):
    if m <= 1:
        return m
    pisano = [0, 1]
    n0, n1= 0, 1
    for _ in range(m*6):
        n0, n1 = n1, (n0+n1)%m
        pisano.append(n1 % m)
        if pisano[-1] == 1 and pisano[-2] == 0:
            break
    return pisano

# find fib_i mod m 
def fib_mod_m(n, m):
    if n <= 1:
        return n
    
    pisano = pisano_period(m)
    return pisano[n  % (len(pisano)-2)]

# since fib[0] + fib[1] + ... + fib[n] = fib[n+2] - 1
# we can compute last number of fib[n+2] - 1
def fib_sum_last_digit(n):
    if n <= 1:
        return max(0,n)
    sum_last = get_fibonacci_last_digit(n+2)
    if sum_last == 0:
        sum_last = 9
    else:
        sum_last -= 1
    return sum_last

def fib_partial_sum_last_digit(m, n):
    if n <= 1:
        return n
    left = fib_sum_last_digit(m-1)
    right = fib_sum_last_digit(n)

    result = right - left
    if result < 0:
        result += 10
    return result

def fib_sum_of_squares_last_digit(n):
    if n <= 1:
        return n
    left = get_fibonacci_last_digit(n)
    right = get_fibonacci_last_digit(n+1)
    result = left * right
    if result >= 10:
        result %= 10
    return result


# greatest common divisor
def gcd(a, b):
    if b == 0:
        return a
    a_prime = a - int(a/b)*b
    return gcd(b, a_prime)

# least common multiple
def lcm(a, b):
    return int(abs(a*b) / gcd(a, b))

print(fib_sum_of_squares_last_digit(2))