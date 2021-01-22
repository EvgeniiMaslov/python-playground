#Uses python3
import sys
import queue
class TrieNode():
    def __init__(self, char, cnt):
        self.char = char
        self.children = dict()
        self.counter = cnt
        self.is_end = False

class Trie():
    def __init__(self):
        self.cnt = 0
        self.root = TrieNode("", self.cnt)
        
    
    def insert(self, word):
        node = self.root
        for ch in word:
            if ch in node.children:
                node = node.children[ch]
            else:
                self.cnt += 1
                new_node = TrieNode(ch, self.cnt)
                node.children[ch] = new_node
                node = new_node
        node.is_end = True

    def print_trie(self):
        node = self.root
        q = queue.Queue()
        q.put(node)
        while not q.empty():
            u = q.get()
            for v in u.children:
                child = u.children[v]
                q.put(child)
                print("{}->{}:{}".format(u.counter, child.counter, child.char))
            
    


if __name__ == '__main__':
    patterns = sys.stdin.read().split()[1:]
    trie = Trie()
    for word in patterns:
        trie.insert(word)
    trie.print_trie()
    
