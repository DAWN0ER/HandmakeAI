import numpy as np

from net.actvs import *
from net.layers import *
import net.dataloader as loader
import net.model as mm
import net.loss as loss

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
    Conv((1,224,224),3,8), #=> (4,222,222)
    Relu(),
    Pool(2,2), #=> (4,111,111)
    Conv((8,111,111),3,12), #=> (16,108,108)
    Relu(),
    Pool(2,2), #=> (12,54,54)
    Conv((12,54,54),3,16), #=> (4,52,52) 
    Tanh(),
    Pool(2,2), #=> (16,26,26)
    Reshape((16,26,26),(16*26*26,1)),
    Fc(16*26*26,124),
    Tanh(),
    Fc(124,6),
    Softmax(),
]

if __name__ == '__main__':
    (x_train,y_train),(x_test,y_test) = load_data()
    indices = np.random.permutation(x_train.shape[0])
    x_train = x_train[0:6750:25]
    y_train = y_train[0:6750:25]
    x_test = x_test[0:300:10]
    y_test = y_test[0:300:10]

    print(x_train.shape)

    batch_size = 10
    mm.batch_train(
        network=network,
        loss=loss.categorical_cross_entropy,
        loss_prime=loss.categorical_cross_entropy_prime,
        x_train=x_train,
        y_train=y_train,
        epoches=5,
        learning_rate=0.00005/batch_size,
        batch_size=batch_size,
        print_turn=1,
        shuffle=True,
        delta=0.9,
        learning_decay=0.8 
    )

    acc = 0
    for x,y in zip(x_test,y_test):
        pred = mm.predict(network,x)
        idx = np.argmax(pred)
        if idx == np.argmax(y):
            acc += 1
        print(f'pred={idx}:{np.argmax(y)}(T) probability:{pred[idx,0]:.4f}')
    print(f'acc={acc/len(x_test)}')

    mm.save(network=network, path='./save/clsf.j')
