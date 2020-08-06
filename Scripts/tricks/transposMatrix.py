'''
使用推导式的嵌套列表实现一个矩阵的转置。
https://www.runoob.com/python3/python3-data-structure.html
'''

matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
        ]

trans_matrix = [[row[i] for row in matrix] for i in range(len(matrix[1]))]
print(trans_matrix)
