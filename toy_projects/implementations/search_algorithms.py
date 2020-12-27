# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:11:34 2020

@author: sqrte
"""

def binary_search(a, key, low=None, high=None):
    n = len(a) - 1
    
    if low == None and high == None:
        low = 0
        high = n

    mid = (low + high) // 2
    if low > high:
        return -1

    if key == a[mid]:
        return mid
    elif key < a[mid]:
        return binary_search(a, key, low, mid-1)
    else:
        return binary_search(a, key, mid+1, high)


def linear_search(arr, element):
    for index, value in enumerate(arr):
        if value == element:
            return index
    return 'Value not in list'