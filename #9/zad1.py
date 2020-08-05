import math


def fact(x):
    return 1/(1+(math.exp(-x)))

def forwardPass(i1, i2, i3):
    w = [0, -0.46122, 0.78548, 0.97314, 2.10584, -0.39203, -0.57847, -0.81546, 1.03775]
    wb = [0, 0.80109, 0.43529, -0.2368]
    h1 = i1*w[1] + i2*w[3] + i3*w[5] + wb[1]
    h2 = i1*w[2] + i2*w[4] + i3*w[6] + wb[2]
    h1 = fact(h1)
    h2 = fact(h2)
    output = h1*w[7] + h2*w[8] + wb[3]
    return output


print(forwardPass(23, 75, 176))