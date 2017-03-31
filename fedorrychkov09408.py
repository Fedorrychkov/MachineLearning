import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg  as lalg
import random as rnd


def Farr(x, n):
    FT = np.ones((x.size, n))
    for i in range(x.size):
        for j in range(n):
            FT[i][j] = x[i] ** j

    F = np.column_stack((FT, np.sin(x)))
    F = np.column_stack((F, np.cos(x)))
    F = np.column_stack((F, np.exp(x)))
    #print (FT)
    return F



x = np.linspace(0, 1, 1000)

y = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
_err = 10 * np.random.rand(1000)
train = y + _err

ind = list(range(1000))
rnd.shuffle(ind)

train_x = x[ind[:600]]
train_t = train[ind[:600]]


valid_x = x[ind[600:800]]
valid_t = train[ind[600:800]]

test_x = x[ind[800:]]
test_t = train[ind[800:]]

step=21
Err = np.zeros(step)
k_er = np.linspace(0, step, step)


def calc_w(F):
    lam = 0  #
    I = np.identity(F.shape[1])  #
    w = ((lalg.inv(F.transpose().dot(F)) + I * lam).dot(F.transpose())).dot(train_t)  # train_t
    return w

def calc_tr_x(step, i):
    F = Farr(train_x, i)
    w = calc_w(F)
    return w
def calc_valid(step, i, w):
    F = Farr(valid_x, i)
    new_y = w.dot(F.transpose())
    return new_y

def calc_test_last(step):
    F = Farr(test_x, step)
    lam = 0  #
    I = np.identity(F.shape[1])  #
    w = ((lalg.inv(F.transpose().dot(F)) + I * lam).dot(F.transpose())).dot(test_t)  # train_x или test_x? При построении тестовой выборки
    for i in range (step): #возможно степ не надо, я не понял.
        predict_y = w.dot(F.transpose())
    print ("Minimal Error index:  ", step)

    #print (F)

    #print (w)

    #print (predict_y)
    return predict_y
def main():
    for i in range(step):
        w = calc_tr_x(step, i)
        new_y = calc_valid(step, i, w)
        for j in range(w.size):
            Err[i] += (valid_t[j] - new_y[j]) ** 2
            Err[i] = Err[i] / 2

    min_step = np.argmin(Err)
    print("Minimal Error from Valid", np.min(Err))
    predict_y = calc_test_last(min_step)
    for j in range(w.size):
        Err[i] += (test_t[j] - predict_y[j]) ** 2
        Err[i] = Err[i] / 2
    print ("Minimal Error from Test: ", np.min(Err))

    plt.figure('Предсказываем график слева, график ошибки справа')
    plt.subplot(1, 2, 1)
    #plt.plot(train_x, train_t, '.g')
    plt.plot(test_x, test_t, '.y')
    plt.plot(test_x, predict_y, '.r') #Я пытаюсь строить новый график предсказаний по test, но как то не очень получается..

    #plt.plot(valid_x, valid_t, '.b')
    #plt.plot(valid_x, new_y, 'r')  # Я пытаюсь строить график из valid set, там творится какой то ужас. Не понимаю что не так.


    plt.subplot(1, 2, 2)
    plt.plot(k_er, Err, 'g')
    plt.show()
if __name__ == "__main__":
    main()
