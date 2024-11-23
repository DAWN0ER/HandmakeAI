import numpy as np
from keras._tf_keras.keras.datasets import mnist
import keras._tf_keras.keras.utils as util

import net.model as model

def process_data(x,y,limit):
    x = x[:limit]
    x = x.reshape(len(x),1,28,28)
    x = x.astype("float32")/255

    y = y[:limit]
    y = util.to_categorical(y)
    y = y.reshape(len(y),10,1)
    
    return x,y

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# acc=0.39666666666666667
x_train,y_train = process_data(x_train,y_train,100)
x_test,y_test = process_data(x_test,y_test,300)

network = model.load(path='./save/s_m.j')

acc = 0
for x,y in zip(x_test,y_test):
    pred = model.predict(network,x)
    idx = np.argmax(pred)
    if idx == np.argmax(y):
        acc += 1
    print(f'pred:{idx},ground true:{np.argmax(y)}. probability:{pred[idx,0]}')
print(f'acc={acc/len(x_test)}')