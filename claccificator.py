import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg  as lalg
import random as rnd

m1= np.random.randint(5,10)
m2= np.random.randint(5,10)
sgm1x1 = np.random.randint(1,8)
sgm1y1 = np.random.randint(1,8)
x1=np.random.randn(100)*sgm1x1+m1
y1=np.random.randn(100)*sgm1y1+m2

m3= np.random.randint(5,50)
m4= np.random.randint(5,60)
sgm1x2 = np.random.randint(1,8)
sgm1y2 = np.random.randint(1,8)
x2=np.random.randn(100)*sgm1x2+m3
y2=np.random.randn(100)*sgm1y2+m4


natural_t_C0 = np.zeros(100)
natural_t_C1 = np.ones(100)

ind = list(range(100))
rnd.shuffle(ind)

new_x1 = x1[ind[:]]
new_y1 = y1[ind[:]]

new_x2 = x2[ind[:]]
new_y2 = y2[ind[:]]

new_C0 = natural_t_C0[ind[:]]
new_C1 = natural_t_C1[ind[:]]

new_t = np.random.randint(0,2,200)
print (new_t)

#Досчитать все True Positive True Negative True Negative False Positive, метрики все остальные
plt.figure()
plt.plot(new_x1,new_y1, '.r')
plt.plot(new_x2,new_y2, '.b')
plt.show()

#print (x1)
#for i in range(50):
