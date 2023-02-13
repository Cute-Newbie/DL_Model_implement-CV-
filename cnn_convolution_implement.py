# -*- coding: utf-8 -*-
"""CNN_convolution_implement.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uCGXkZK8U5yZ5xySOEjP5Zd62Sy7y5mf
"""

#Reference: https://ndb796.tistory.com/
#CNN study using numpy and python
import numpy as np

"""Simple Calculation

"""

arr1 = np.array([[1,2,3],
                 [4,5,6]])

arr2 = np.array([[1,0,-1],
                 [1,0,-1]])
# 요소별 곱
print(arr1*arr2)
print()
# 요소별 곱의 합
print(np.sum(arr1*arr2))

"""Padding

"""

# padding - 1d

array = [1,2,3]

print(np.pad(array,(2,3),'constant',constant_values = 0))
print(np.pad(array,(0,3),'constant',constant_values = 0))

#padding - 2d

array2 = [[1,2,3],
          [4,5,6]]


print(np.pad(array2,((2,2),(2,2)),'constant',constant_values = 0))
print()
# 위쪽 1개행, 아래쪽 2개행, 왼쪽 3개열, 오른족 4개열 0으로 패딩
print(np.pad(array2, ((1,2),(3,4)), 'constant', constant_values=0))

#padding for unenven lengths of list
array3 = [[1,2],
          [4,5,6],
          [7]]

def smoother(array,fixed_length,padding_value = 0):
    answer = []
    for i in array:
        answer.append(np.pad(i,(0,fixed_length),'constant',constant_values=padding_value)[:fixed_length])

    return np.concatenate(answer,axis=0).reshape(-1,fixed_length)
        

smoother(array3,
         fixed_length=5,
         
         padding_value = 0)

"""Convolution layer"""

def conv(image, filters, stride=1, pad=0):
    n, c, h, w = image.shape
    n_f, _, filter_height, filter_width = filters.shape

    out_height = (h + 2 * pad - filter_height) // stride + 1
    out_width = (w + 2 * pad - filter_width) // stride + 1

    
    padded_image = np.pad(X, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    output = np.zeros((n, n_f, out_height, out_width))

    for i in range(n): 
        for c in range(n_f): 
            for h in range(out_height): 
                h_start = h * stride
                h_end = h_start + filter_height
                for w in range(out_width):
                    w_start = w * stride
                    w_end = w_start + filter_width
                    
                    output[i, c, h, w] = np.sum(padded_image[i, :, h_start:h_end, w_start:w_end] * filters[c])

    return output

X = np.asarray([
# image 1
[
    [[1, 2, 9, 2, 7],
    [5, 0, 3, 1, 8],
    [4, 1, 3, 0, 6],
    [2, 5, 2, 9, 5],
    [6, 5, 1, 3, 2]],

    [[4, 5, 7, 0, 8],
    [5, 8, 5, 3, 5],
    [4, 2, 1, 6, 5],
    [7, 3, 2, 1, 0],
    [6, 1, 2, 2, 6]],

    [[3, 7, 4, 5, 0],
    [5, 4, 6, 8, 9],
    [6, 1, 9, 1, 6],
    [9, 3, 0, 2, 4],
    [1, 2, 5, 5, 2]]
],
# image 2
[
    [[7, 2, 1, 4, 2],
    [5, 4, 6, 5, 0],
    [1, 2, 4, 2, 8],
    [5, 9, 0, 5, 1],
    [7, 6, 2, 4, 6]],

    [[5, 4, 2, 5, 7],
    [6, 1, 4, 0, 5],
    [8, 9, 4, 7, 6],
    [4, 5, 5, 6, 7],
    [1, 2, 7, 4, 1]],

    [[7, 4, 8, 9, 7],
    [5, 5, 8, 1, 4],
    [3, 2, 2, 5, 2],
    [1, 0, 3, 7, 6],
    [4, 5, 4, 5, 5]]
]
])

filters = np.asarray([
# kernel 1
[
    [[1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]],

    [[3, 1, 3],
    [1, 3, 1],
    [3, 1, 3]],

    [[1, 2, 1],
    [2, 2, 2],
    [1, 2, 1]]
],
# kernel 2
[
    [[5, 1, 5],
    [2, 1, 2],
    [5, 1, 5]],

    [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]],

    [[2, 0, 2],
    [0, 2, 0],
    [2, 0, 2]],
],
# kernel 3
[
    [[5, 1, 5],
    [2, 1, 2],
    [5, 1, 5]],

    [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]],

    [[2, 0, 2],
    [0, 2, 0],
    [2, 0, 2]],
]
])

filters.shape

out = conv(X, filters, stride=2, pad=0)
print('Output:', out.shape)
print(out)
