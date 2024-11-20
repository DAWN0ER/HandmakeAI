
fig =  plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(points[:,0],points[:,1],points[:,2],c=points[:,2],cmap="winter")
plt.show()