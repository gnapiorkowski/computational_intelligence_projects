matrix = [[1, 0.7, 0.7, 1, 0.7, 0.7, 0.5, 0.2],
[0.7, 1, 1, 0.7, 0.5, 0.5, 0.7, 0.5],
[0.5, 0.7, 0.7, 0.5, 0, 0.2, 0.5, 0.7],
[0.2, 0.5, 0, 0, 0, 0.2, 0.7, 0.5],
[0, 0.2, 0, 0.2, 0.2, 0.2, 0.7, 0.7],
[0, 0, 0, 0.2, 0.5, 0.5, 0.7, 0.7],
[0, 0, 0, 0.2, 0.5, 0.7, 0.7, 1],
[0, 0.2, 0.2, 0.5, 0.5, 0.7, 1, 1]]
filter1 = [[1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]]
filter2 = [[1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]]
filter3 = [[1, 0, 0.5],
        [0, 0, 0],
        [0, -0.5, 0]]
filter4 = [[1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]]

def multiply(a, b):
    if len(a) != len(b): return False
    y = []
    for i in range(len(a)):
        y.append([])
        for j in range(len(a[i])):
            y[i].append(a[i][j]*b[i][j])
        y[i] = sum(y[i])
    return max([0, sum(y)])

def convolute(a, b):
    if len(a) == len(b): return False
    convoluted = []
    #cycle rows
    for i in range(len(a)-len(b)+1):
        convoluted.append([])
        #cycle columns
        for j in range(len(a[i])-len(b)+1):
            tmp = []
            lenB = (len(b))
            for col in range(len(b)):
                tmp.append([])
                for row in range(len(b)):
                    tmp[col].append(a[col+i][row+j])
            assert len(tmp) == len(b), 'length of tmp equals that of b'
            convoluted[i].append(multiply(tmp, b))
    return convoluted

def maxPooling(matrix):
    pooled = []
    assert len(matrix)%2 == 0, 'matrix does not have even number of rows'
    for i in range(0, int(len(matrix)/2)+2, 2):
        assert len(matrix) == len(matrix[i]), 'pooled matrix is not a square'
        pooled.append([])
        for j in range(0, int(len(matrix)/2)+2, 2):
            tmp = matrix[i][j:j+2]
            for m1 in matrix[i+1][j:j+2]:
                tmp.append(m1)
            tmp = max(tmp)
            pooled[int(i/2)].append(tmp)
    return pooled
def main():
    x1 = multiply(filter3, maxPooling(convolute(matrix, filter1)))
    x2 = multiply(filter4, maxPooling(convolute(matrix, filter2)))
    x3 = x1*0.5 + x2*0.7
    x4 = x1*0.1 + x2*0.2
    return[x3, x4]
print(main())