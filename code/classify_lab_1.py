import numpy as np

from net.actvs import *
from net.layers import *
import net.dataloader as loader
import net.model as mm
import net.loss as loss
from tqdm import tqdm

'''
(6750, 224, 224, 1)
(300, 224, 224, 1)
(6750, 6, 1)
(300, 6, 1)
'''

def load_data():
    path = './works/dataset_npz/signal.npz'
    try:
        npload = np.load(path)
        x_train = npload['x_train']
        y_train  = npload['y_train']
        x_test= npload['x_test']
        y_test= npload['y_test']
        return (x_train,y_train),(x_test,y_test)
    except Exception as e:
        print(f'fail load:{e}')
        (x_train,y_train),(x_test,y_test) = loader.load_dataset()
        np.savez('./works/dataset_npz/signal.npz',x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
        return (x_train,y_train),(x_test,y_test)

network = [
    Conv((1,224,224),9,16),         #=> (16,216,216)
    Relu(),
    Pool(2,2),                      #=> (16,108,108)
    Conv((16,108,108),3,32),        #=> (32,106,106)
    Relu(),
    Pool(2,2),                      #=> (32,53,53)
    Reshape((32,53,53),(32*53*53,1)),
    Fc(32*53*53,256),
    Relu(),
    Fc(256,256),
    Tanh(),
    Fc(256,6),
    Softmax(),
]

if __name__ == '__main__':
    (x_train,y_train),(x_test,y_test) = load_data()
    x_train = x_train[0:6750:2]
    y_train = y_train[0:6750:2]
    indices = np.random.permutation(x_train.shape[0])
    x_train = x_train[indices]
    y_train = y_train[indices]
    x_test = x_test[0:300:5]
    y_test = y_test[0:300:5]

    print(x_train.shape)

    epoches = 2
    batch_size = 32
    best_acc = 0.0
    for epoch in range(epoches):

        mm.batch_train(
            network=network,
            loss=loss.categorical_cross_entropy,
            loss_prime=loss.categorical_cross_entropy_prime,
            x_train=x_train,
            y_train=y_train,
            epoches=1,
            learning_rate=0.0001/batch_size,
            batch_size=batch_size,
            print_turn=1,
            shuffle=True,
            delta=0.75,
            learning_decay=1
        )

        acc = 0
        for x,y in tqdm(zip(x_test,y_test)):
            pred = mm.predict(network,x)
            idx = np.argmax(pred)
            if idx == np.argmax(y):
                acc += 1
        acc = acc/len(x_test)
        print(f'test acc={acc}')
        if (acc > best_acc):
            best_acc = acc
            mm.save(network=network, path='./save/clsf2.j')
