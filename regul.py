import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg  as lalg
import random as rnd


def Farr(x, n):
    F = np.ones((x.size, n))
    for i in range(x.size):
        for j in range(n):
            F[i][j] = x[i] ** j
    return F

def Fsin(x,n):#надо бы сделать матрицу sin и cos
    F2 = np.ones((x.size, n))
    for i in range(x.size):
        for j in range(n):
            F2[i/2][j/2] = np.sin(x)
            F2[i/2+1][j/2+1] = np.cos(x)
    return F2

dots=1000
x = np.linspace(0, 1, dots)

y = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
_err = 10 * np.random.rand(dots)
train = y + _err
el=21

ind = list(range(1000))
rnd.shuffle(ind)

train_x = x[ind[:600]]
train_t = train[ind[:600]]


valid_x = x[ind[600:800]]
valid_t = train[ind[600:800]]

test_x = x[ind[800:]]
test_t = train[ind[800:]]


Err = np.zeros(el)
k_er = np.linspace(0, el, el)
for i in range(12):
    F = Farr(x, i)
    w = ((lalg.inv(F.transpose().dot(F))).dot(F.transpose())).dot(train)
    new_y = F.dot(w)
    for j in range(train_t.size):
        Err[i] += (train_t[j] - new_y[j]) ** 2
    Err[i] = Err[i] / 2
print(Err)

#for i in range(0,x.size):
 #   new_x=rnd.shuffle(train)

#train_set = new_x[0:600]
#valid_set = new_x[600:800]
#test_set = new_x[800:]


plt.figure()
plt.subplot(1,2,1)
plt.plot(train_x, train_t, '.g')
plt.plot(valid_x, valid_t, '.b')
plt.plot(test_x, test_t, '.y')
plt.plot(x, new_y, 'r')
plt.subplot(1,2,2)
plt.plot(k_er, Err, 'g')
plt.show()