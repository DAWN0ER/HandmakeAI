import numpy as np
from net.loss import categorical_cross_entropy_prime,categorical_cross_entropy

y = np.reshape([1,0,0,0,0],(5,1))
yy = np.reshape([0.2,0.1,0.3,0.2,0.2],(5,1))
grad = categorical_cross_entropy_prime(y,yy)
print(categorical_cross_entropy(y,yy))

print(grad.shape)
# X = np.reshape([[0,0],[0,1],[1,0],[1,1],[2,3]],(5,2,1))
# Y = np.reshape([[0],[1],[1],[0],[9]],(5,1,1))

# c = np.reshape(X,(X.shape[0],1,*X.shape[1:]))
# print(c.shape)

# batch = 2
# num = len(X) // batch
# x_batch = [X[i*batch:(i+1)*batch] for i in range(num)]
# y_batch = [Y[i*batch:(i+1)*batch] for i in range(num)]
# remaining = X[num*batch:]
# if remaining.size:
#     x_batch.append(remaining)
# remaining = Y[num*batch:]
# if remaining.size:
#     y_batch.append(remaining)

# print(x_batch)
# print(y_batch)

# print('--------------')
# for x,y in zip(x_batch,y_batch):
#     print(x)
#     print(y)
#     print('=========')
#     for xi,yi in zip(x,y):
#         print(xi)
#         print(yi)
#         print('...........')