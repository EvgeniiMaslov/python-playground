#Uses python3

import sys

def is_cyclic(v, visited, recStack): 
        visited[v] = True
        recStack[v] = True

        for neighbour in adj[v]: 
            if visited[neighbour] == False: 
                if is_cyclic(neighbour, visited, recStack) == True: 
                    return True
            elif recStack[neighbour] == True: 
                return True 
        recStack[v] = False
        return False

def acyclic(adj):
    visited = [False] * len(adj)
    recStack = [False] * len(adj)
    for node in range(len(adj)): 
        if visited[node] == False: 
            if is_cyclic(node,visited,recStack) == True: 
                return 1
    return 0

if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n, m = data[0:2]
    data = data[2:]
    edges = list(zip(data[0:(2 * m):2], data[1:(2 * m):2]))
    adj = [[] for _ in range(n)]
    for (a, b) in edges:
        adj[a - 1].append(b - 1)
    print(acyclic(adj))
