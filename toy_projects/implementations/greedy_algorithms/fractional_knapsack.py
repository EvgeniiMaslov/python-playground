# Uses python3
import sys

def get_optimal_value(capacity, weights, values):
    value = 0.
    # write your code here
    value_per_weight = [values[ind] / weights[ind] for ind in range(len(weights))]
    cap = capacity
    while cap > 0:
        if len(weights) == 0:
            break
        ind = value_per_weight.index(max(value_per_weight))
        mn = min(cap, weights[ind])
        if mn == cap:
            value += value_per_weight[ind] * cap
            cap = 0
        elif mn == weights[ind]:
            value += value_per_weight[ind] * weights[ind]
            cap -= weights[ind]
            values.pop(ind)
            weights.pop(ind)
            value_per_weight.pop(ind)
    return value


if __name__ == "__main__":
    data = list(map(int, sys.stdin.read().split()))
    n, capacity = data[0:2]
    values = data[2:(2 * n + 2):2]
    weights = data[3:(2 * n + 2):2]
    opt_value = get_optimal_value(capacity, weights, values)

    
    
    print("{:.10f}".format(opt_value))
