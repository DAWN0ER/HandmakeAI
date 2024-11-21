import numpy as np
from keras._tf_keras.keras.datasets import mnist
import keras._tf_keras.keras.utils as util

from net.layers import Fc,Conv,Reshape
from net.actvs import Sigmoid,Softmax
from net.loss import categorical_cross_entropy,categorical_cross_entropy_prime
import net.model as model

# 主要用于复现实验

def process_data(x,y,limit):
    x = x[:limit]
    x = x.reshape(len(x),1,28,28)
    x = x.astype("float32")/255

    y = y[:limit]
    y = util.to_categorical(y)
    y = y.reshape(len(y),10,1)
    
    return x,y

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train,y_train = process_data(x_train,y_train,100)
x_test,y_test = process_data(x_test,y_test,300)

network = [
    Conv((1,28,28),3,5),
    Sigmoid(),
    Reshape((5,26,26),(5*26*26,1)),
    Fc(5*26*26,124),
    Sigmoid(),
    Fc(124,10),
    Softmax(),
]

model.train(
    network=network,
    loss=categorical_cross_entropy,
    loss_prime=categorical_cross_entropy_prime,
    x_train=x_train,
    y_train=y_train,
    epoches=100,
    learning_rate=0.005
)

model.save(network=network,path='./save/mnist.j')

acc = 0
for x,y in zip(x_test,y_test):
    pred = model.predict(network,x)
    idx = np.argmax(pred)
    if idx == np.argmax(y):
        acc += 1
    print(f'pred:{idx},ground true:{np.argmax(y)}. probability:{pred[idx,0]}')
print(f'acc={acc/len(x_test)}')