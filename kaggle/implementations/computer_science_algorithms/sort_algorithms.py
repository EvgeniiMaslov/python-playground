import random

def bubble_sort(arr):
    for iteration in range(len(arr) - 1):
        for i in range(len(arr)-1):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
    return arr

def insertion_sort(arr):
    for j in range(1, len(arr)):
        key = arr[j]
        i = j-1
        while i >= 0 and arr[i] > key:
            arr[i+1] = arr[i]
            i -= 1
        arr[i+1] = key
    return arr

def selection_sort(arr):
    for i, element in enumerate(arr):
        mn = min(range(i, len(arr)), key=arr.__getitem__)
        arr[i], arr[mn] = arr[mn], element
    return arr

def merge_sort(arr):
    count = len(arr)
    if count > 2:
        part_1 = merge_sort(arr[:count // 2])
        part_2 = merge_sort(arr[count // 2:])
        arr = part_1 + part_2
        last_index = len(arr) - 1

        for i in range(last_index):
            min_value = arr[i]
            min_index = i

            for j in range(i + 1, last_index + 1):
                if min_value > arr[j]:
                    min_value = arr[j]
                    min_index = j

            if min_index != i:
                arr[i], arr[min_index] = arr[min_index], arr[i]

    elif len(arr) > 1 and arr[0] > arr[1]:
        arr[0], arr[1] = arr[1], arr[0]

    return arr

def quicksort(arr):
   if len(arr) <= 1:
       return arr
   else:
       q = random.choice(arr)
   l_arr = [n for n in arr if n < q]
 
   e_arr = [q] * arr.count(q)
   b_arr = [n for n in arr if n > q]
   return quicksort(l_arr) + e_arr + quicksort(b_arr)

arr = [9, 8, 7, 6, 5, 4, 3, 2, 1]
sorted_arr = quicksort(arr)

print('Array: {}'.format(arr))
print('Sorted array: {}'.format(sorted_arr))