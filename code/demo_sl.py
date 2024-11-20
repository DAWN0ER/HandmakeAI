import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

from net.layers import Fc
from net.actvs import *
from net.loss import mse,mse_prime
from net.model import train,predict,save

# OXR 函数
X = np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
Y = np.reshape([[0],[1],[1],[0]],(4,1,1))

network = [
    Fc(2,5),
    Tanh(),
    Fc(5,1),
    Tanh(),
]

train(network,mse,mse_prime,X,Y,epoches=10000,learning_rate=0.1)

save(network,'./demo.j')

points = []
for x in np.linspace(0,1,20):
    for y in np.linspace(0,1,20):
        z = predict(network,[[x],[y]])
        points.append([x,y,z[0,0]])

points = np.array(points)

fig =  plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(points[:,0],points[:,1],points[:,2],c=points[:,2],cmap="winter")

#################################
path = './demo.j'
with open(path) as f:
    obj = json.load(f)
np_load = np.load(path + '.npz')
print(np_load.files)
networks = []
for layer in obj:
    t = layer['type']
    if t == str(type(Fc)):
        l = Fc(**layer['init'])
        l.weights = np_load[layer['weights']]
        l.bias = np_load[layer['bias']]
        networks.append(l)

    elif t == str(type(Tanh)):
        l = Tanh()
        networks.append(l)

points = []
for x in np.linspace(0,1,20):
    for y in np.linspace(0,1,20):
        z = np.array(predict(networks,[[x],[y]]))
        z = z[0,0]
        points.append([x,y,z])

points = np.array(points)

fig =  plt.figure(3)
ax = fig.add_subplot(111,projection="3d")
ax.scatter(points[:,0],points[:,1],points[:,2],c=points[:,2],cmap="winter")

#################################
print(np.array_equal(network[0].weights,networks[0].weights))
print(np.array_equal(network[0].bias,networks[0].bias))

plt.show()