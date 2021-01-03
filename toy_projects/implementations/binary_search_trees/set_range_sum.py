import sys


class Vertex:
    def __init__(self, key, sum, left, right, parent):
        self.key, self.sum, self.left, self.right, self.parent = \
            key, sum, left, right, parent


class SplayTree:
    @staticmethod
    def update(v):
        if v is None:
            return
        v.sum = v.key + (v.left.sum if v.left is not None else 0) + (
            v.right.sum if v.right is not None else 0)
        if v.left is not None:
            v.left.parent = v
        if v.right is not None:
            v.right.parent = v

    @classmethod
    def _small_rotation(cls, v):
        parent = v.parent
        if parent is None:
            return
        grandparent = v.parent.parent
        if parent.left == v:
            m = v.right
            v.right = parent
            parent.left = m
        else:
            m = v.left
            v.left = parent
            parent.right = m
        cls.update(parent)
        cls.update(v)
        v.parent = grandparent
        if grandparent is not None:
            if grandparent.left == parent:
                grandparent.left = v
            else:
                grandparent.right = v

    @classmethod
    def _big_rotation(cls, v):
        if v.parent.left == v and v.parent.parent.left == v.parent:
            cls._small_rotation(v.parent)
            cls._small_rotation(v)
        elif v.parent.right == v and v.parent.parent.right == v.parent:
            cls._small_rotation(v.parent)
            cls._small_rotation(v)
        else:
            cls._small_rotation(v)
            cls._small_rotation(v)

    @classmethod
    def splay(cls, v):
        if v is None:
            return None
        while v.parent is not None:
            if v.parent.parent is None:
                cls._small_rotation(v)
                break
            cls._big_rotation(v)
        return v

    @classmethod
    def find(cls, root, key):
        v = root
        last = root
        next_ = None
        while v is not None:
            if v.key >= key and (next_ is None or v.key < next_.key):
                next_ = v
            last = v
            if v.key == key:
                break
            if v.key < key:
                v = v.right
            else:
                v = v.left
        root = cls.splay(last)
        return next_, root

    @classmethod
    def split(cls, root, key):
        result, root = SplayTree.find(root, key)
        if result is None:
            return root, None
        right = cls.splay(result)
        left = right.left
        right.left = None
        if left is not None:
            left.parent = None
        cls.update(left)
        cls.update(right)
        return left, right

    @classmethod
    def merge(cls, left, right):
        if left is None:
            return right
        if right is None:
            return left
        while right.left is not None:
            right = right.left
        right = cls.splay(right)
        right.left = left
        cls.update(right)
        return right


class Set:
    root = None

    def insert(self, key):
        left, right = SplayTree.split(self.root, key)
        new_vertex = None
        if right is None or right.key != key:
            new_vertex = Vertex(key, key, None, None, None)
        self.root = SplayTree.merge(SplayTree.merge(left, new_vertex), right)

    def erase(self, key):
        if self.search(key) is None:
            return

        SplayTree.splay(self.root)
        self.root = SplayTree.merge(self.root.left, self.root.right)
        if self.root is not None:
            self.root.parent = None

    def search(self, key):
        result, self.root = SplayTree.find(self.root, key)
        if result is None or result.key != key:
            return None
        return result.key

    def sum(self, fr, to):
        left, middle = SplayTree.split(self.root, fr)
        middle, right = SplayTree.split(middle, to + 1)

        if middle is None:
            ans = 0
            self.root = SplayTree.merge(left, right)
        else:
            ans = middle.sum
            self.root = SplayTree.merge(SplayTree.merge(left, middle), right)

        return ans


if __name__ == "__main__":
    n = int(sys.stdin.readline())

    last_sum_result = 0
    MODULO = 1000000001

    s = Set()
    for i in range(n):
        line = sys.stdin.readline().split()
        if line[0] == "+":
            x = int(line[1])
            s.insert((x + last_sum_result) % MODULO)
        elif line[0] == "-":
            x = int(line[1])
            s.erase((x + last_sum_result) % MODULO)
        elif line[0] == "?":
            x = int(line[1])
            print(
                "Found" if s.search(
                    (x + last_sum_result) % MODULO) is not None else "Not found"
            )
        elif line[0] == "s":
            l = int(line[1])
            r = int(line[2])
            res = s.sum((l + last_sum_result) % MODULO,
                        (r + last_sum_result) % MODULO)
            print(res)
            last_sum_result = res % MODULO