from collections import namedtuple

Bracket = namedtuple("Bracket", ["char", "position"])


def are_matching(left, right):
    return (left + right) in ["()", "[]", "{}"]


def find_mismatch(text):
    opening_brackets_stack = []
    indexes = []
    for i, next in enumerate(text):
        if next in "([{":
            # Process opening bracket, write your code here
            opening_brackets_stack.append(next)
            indexes.append(i+1)
        if next in ")]}":
            # Process closing bracket, write your code here
            if len(opening_brackets_stack) == 0:
                return i+1
            top = opening_brackets_stack.pop()
            indexes.pop()
            if (top == "[" and next != "]") or (top == "(" and next != ")") or (top == "{" and next != "}"):
                return i+1

    if opening_brackets_stack:
        return indexes[-1]
    
    return "Success"


def main():
    text = input()
    mismatch = find_mismatch(text)
    print(mismatch)
    # Printing answer, write your code here


if __name__ == "__main__":
    main()
