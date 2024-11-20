import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from net.layers import Fc
from net.actvs import *
from net.loss import categorical_cross_entropy,categorical_cross_entropy_prime
from net.model import train,predict

# OXR 函数
X = np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
Y = np.reshape([[0,1],[1,0],[1,0],[0,1]],(4,2,1))

network = [
    Fc(2,5),
    Tanh(),
    Fc(5,2),
    Softmax(),
]

train(network,categorical_cross_entropy,categorical_cross_entropy_prime,X,Y,epoches=10000,learning_rate=0.1)

points1 = []
points2 = []
for x in np.linspace(0,1,20):
    for y in np.linspace(0,1,20):
        z = predict(network,[[x],[y]])
        points1.append([x,y,z[0,0]])
        points2.append([x,y,z[1,0]])

points1 = np.array(points1)
points2 = np.array(points2)

fig1 =  plt.figure(1)
ax = fig1.add_subplot(111,projection="3d")
ax.scatter(points1[:,0],points1[:,1],points1[:,2],c=points1[:,2],cmap="winter")

fig2 =  plt.figure(2)
ax = fig2.add_subplot(111,projection="3d")
ax.scatter(points2[:,0],points2[:,1],points2[:,2],c=points2[:,2],cmap="winter")
plt.show()