import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg  as lalg


def Farr(x, n):
    F = np.ones((x.size, n))
    for i in range(x.size):
        for j in range(n):
            F[i][j] = x[i] ** j
    return F

dots=1000
x = np.linspace(0, 1, dots)
y = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
_err = 10 * np.random.rand(dots)
train = y + _err
el=12
Err = np.zeros(el)
k_er = np.linspace(0, el, el)
for i in range(12):#minimal error
    F = Farr(x, i)
    w = ((lalg.inv(F.transpose().dot(F))).dot(F.transpose())).dot(train)
    new_y = F.dot(w)
    for j in range(train.size):
        Err[i] += (train[j] - new_y[j]) ** 2
    Err[i] = Err[i] / 2
print(Err)

plt.figure()
plt.subplot(1,2,1)
plt.plot(x, train, '.g')
plt.plot(x, new_y, 'r')
plt.subplot(1,2,2)
plt.plot(k_er, Err, 'g')
plt.show()