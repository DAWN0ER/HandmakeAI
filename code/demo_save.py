import json
import numpy as np
from net.layers import *
from net.actvs import *
from net.model import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = './demo.j'
with open(path) as f:
    obj = json.load(f)
np_load = np.load(path + '.npz')
print(np_load.files)
networks = []
for layer in obj:
    t = layer['type']
    print(t)
    if t == str(Fc):
        print('FC!')
        l = Fc(**layer['init'])
        l.weights = np_load[layer['weights']]
        l.bias = np_load[layer['bias']]
        networks.append(l)

    elif t == str(Tanh):
        print('Tanh')
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
plt.show()