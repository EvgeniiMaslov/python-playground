import sys
import threading

sys.setrecursionlimit(10 ** 6)  # max depth of recursion
threading.stack_size(2 ** 27)  # new thread will get stack of such size


class TreeOrders:
    def read(self):
        self.n = int(sys.stdin.readline())
        self.key = [0 for _ in range(self.n)]
        self.left = [0 for _ in range(self.n)]
        self.right = [0 for _ in range(self.n)]
        for i in range(self.n):
            self.key[i], self.left[i], self.right[i] = map(
                int, sys.stdin.readline().split()
            )

    def in_order(self):
        current_id = 0
        stack = []

        while True:
            if current_id != -1:
                stack.append(current_id)
                current_id = self.left[current_id]
            elif stack:
                current_id = stack.pop()
                yield self.key[current_id]
                current_id = self.right[current_id]
            else:
                break

    def pre_order(self):
        current_id = 0
        stack = []

        while True:
            if current_id != -1:
                yield self.key[current_id]
                stack.append(current_id)
                current_id = self.left[current_id]
            elif stack:
                current_id = stack.pop()
                current_id = self.right[current_id]
            else:
                break

    def post_order(self):
        stack1 = [0]
        stack2 = []

        while stack1:
            current_id = stack1.pop()
            stack2.append(self.key[current_id])

            left_id = self.left[current_id]
            right_id = self.right[current_id]
            if left_id != -1:
                stack1.append(left_id)
            if right_id != -1:
                stack1.append(right_id)

        while stack2:
            yield stack2.pop()


def main():
    tree = TreeOrders()
    tree.read()
    print(" ".join(str(x) for x in tree.in_order()))
    print(" ".join(str(x) for x in tree.pre_order()))
    print(" ".join(str(x) for x in tree.post_order()))


threading.Thread(target=main).start()