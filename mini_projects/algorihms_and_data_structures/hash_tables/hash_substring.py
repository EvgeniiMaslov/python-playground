class TextSearch:
    def __init__(self, pattern, text):
        self._pattern = pattern
        self._text = text
        self._window = len(pattern)
        self._scan_bound = len(text) - len(pattern) + 1
        self._checksums = []

    def checksum(self, string):
        return sum([ord(string[i]) for i in range(len(string))])

    def precompute_hashes(self):
        self._checksums = [self.checksum(self._text[:self._window])]

        for i in range(1, self._scan_bound):
            old_hash = self._checksums[i - 1]
            left_l_hash = ord(self._text[i - 1])
            right_l_hash = ord(self._text[i + self._window - 1])

            ith_hash = old_hash - left_l_hash + right_l_hash
            self._checksums.append(ith_hash)

    def find(self):
        pattern_checksum = self.checksum(self._pattern)
        self.precompute_hashes()

        results = []
        for i in range(self._scan_bound):
            if pattern_checksum == self._checksums[i]:
                if self._pattern == self._text[i:i + self._window]:
                    results.append(i)
        return results


if __name__ == "__main__":
    pattern, text = input().rstrip(), input().rstrip()

    ts = TextSearch(pattern, text)
    result = ts.find()

    print(" ".join(map(str, result)))