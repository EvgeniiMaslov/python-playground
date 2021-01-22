# python3


class Heap:
    def __init__(self):
        self.data = None
        self.size = None
        self.swaps_id = []
    
    def read_data(self):
        self.size = int(input())
        self.data = list(map(int, input().split()))
        assert len(self.data) == self.size

    def parent(self, i):
        return i // 2
    
    def left_child(self, i):
        return 2*i+1
        
    def right_child(self, i):
        return 2*i+2
    
    def sift_up(self, i):
        parent_index = self.parent(i)
        while i > 0 and self.data[parent_index] < self.data[i]:
            self.data[parent_index], self.data[i] = self.data[i], self.data[parent_index]
            i = parent_index
        
    def sift_down(self, i):
        min_index = i
        l = self.left_child(i)
        if l < self.size and self.data[l] < self.data[min_index]:
            min_index = l

        r = self.right_child(i)
        if r < self.size and self.data[r] < self.data[min_index]:
            min_index = r
    
        if i != min_index:
            self.swaps_id.append([i, min_index])
            self.data[i], self.data[min_index] = self.data[min_index], self.data[i]
            self.sift_down(min_index)
            
    def build_heap(self):
        for i in reversed(range(self.size//2)):
            self.sift_down(i)

    def write_response(self):
        print(len(self.swaps_id))
        for i, j in self.swaps_id:
            print(i, j)

    def solve(self):
        self.read_data()
        self.build_heap()
        self.write_response()

    

if __name__ == "__main__":
    heap = Heap()
    heap.solve()
