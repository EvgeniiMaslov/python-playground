# python3
import sys


def compute_min_refills(distance, tank, stops):
    # write your code here
    
    current_tank = tank
    count = 0
    stops = [0] + stops + [distance]
    for i in range(len(stops) - 1):
        dist = stops[i+1] - stops[i]
        if current_tank < dist:
            current_tank = tank
            count += 1
            if current_tank < dist:
                return -1  
        current_tank -= dist
        
    return count

if __name__ == '__main__':
    d, m, _, *stops = map(int, sys.stdin.read().split())
    print(compute_min_refills(d, m, stops))
