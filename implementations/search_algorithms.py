# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:11:34 2020

@author: sqrte
"""

def binary_search(arr, element, left=None, right=None):
    n = len(arr)
    
    if left == None or right == None:
        left = 0
        right = n

    middle = (left + right) // 2
    
    if arr[middle] == element:
        return middle
    elif arr[middle] < element:
        return binary_search(arr, element, middle, right)
    elif arr[middle] > element:
        return binary_search(arr, element, left, middle)

def binary_search_while(arr, element):

    left = 0
    right = len(arr)
    middle = right // 2
    
    while arr[middle] != element and left != right:
        if arr[middle] < element:
            left = middle + 1
        elif arr[middle] > element:
            right = middle - 1
        middle = (left + right) // 2
    
    if arr[middle] != element:
        return 'Value not in list'
    return middle

def linear_search(arr, element):
    for index, value in enumerate(arr):
        if value == element:
            return index
    return 'Value not in list'