from math import pow
import random
from matplotlib import pyplot as plt
import numpy as np

# def f(x):
#     return pow(x, 3)+3*pow(x, 2)-5*x+7


# def rSeq(length):
#     seq = []
#     for i in range(-length, length):
#         seq.append([i, f(i)])
#     return seq
def f(x):
    return pow(x, 3)+3*pow(x, 2)-5*x+7

def fRes(B):
    x, y = [], []
    for i in range(-100, 100):
        if (i == 0):
            x.append(0)
            y.append(f(0))
        else:
            x.append(i*B/100)
            y.append(f(i*B/100))
    return [x, y]

x = fRes(5)
y = np.array(x[1])
x = np.array(x[0])
print(y, y*2, sep='\n')
plt.plot(x, y, 'ro', x, y*2, 'b--')
plt.show()