import numpy as np
from keras._tf_keras.keras.datasets import mnist
import keras._tf_keras.keras.utils as util

from net.layers import Fc,Conv,Reshape
from net.actvs import Sigmoid,Softmax
from net.loss import categorical_cross_entropy,categorical_cross_entropy_prime
import net.model as model
from tqdm import tqdm

def process_data(x,y,limit):
    x = x[:limit]
    x = x.reshape(len(x),1,28,28)
    x = x.astype("float32")/255

    y = y[:limit]
    y = util.to_categorical(y)
    y = y.reshape(len(y),10,1)
    
    return x,y

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train,y_train = process_data(x_train,y_train,800)
x_test,y_test = process_data(x_test,y_test,200)

network = [
    Conv((1,28,28),3,12),
    Sigmoid(),
    Reshape((12,26,26),(12*26*26,1)),
    Fc(12*26*26,256),
    Sigmoid(),
    Fc(256,10),
    Softmax(),
]

best_acc = 0.0
epoches = 150
for epoch in range(epoches) :
    model.batch_train(
        network=network,
        loss=categorical_cross_entropy,
        loss_prime=categorical_cross_entropy_prime,
        x_train=x_train,
        y_train=y_train,
        epoches=1,
        learning_rate=0.005,
        batch_size=10,
        delta=0.2,
        print_turn=100,
        learning_decay=1
    )
    acc = 0
    for x,y in tqdm(zip(x_test,y_test)):
        pred = model.predict(network,x)
        idx = np.argmax(pred)
        if idx == np.argmax(y):
            acc += 1
    acc = acc/len(x_test)
    print(f'Epoch:{epoch+1}:acc={acc}')
    if acc > best_acc:
        best_acc = acc
        model.save(network,'./save/s_m.j')