import net.model as model
import numpy as np
from net.layers import *
from net.actvs import *
from net.model import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

networks = model.load('./demo.j')

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