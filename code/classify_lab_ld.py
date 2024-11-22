import net.model as mm
import numpy as np
import net.dataloader as loader

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


(x_train,y_train),(x_test,y_test) = load_data()
indices = np.random.permutation(x_test.shape[0])
x_test = x_test[indices]
y_test = y_test[indices]
network = mm.load('./save/clsf2.j')

acc = 0
for x,y in zip(x_test,y_test):
    pred = mm.predict(network,x)
    idx = np.argmax(pred)
    if idx == np.argmax(y):
        acc += 1
    print(f'pred:{pred}|{idx}=={np.argmax(y)}. probability:{pred[idx,0]:.5f}:')
print(f'acc={acc/len(x_test)}')