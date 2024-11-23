from net.layers import *
from net.actvs import *
import json,os
import numpy as np
from tqdm import tqdm

def predict(nwtwork, input):
    output = input
    for layer in nwtwork:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train,y_train,epoches = 1000, learning_rate = 0.002):
    for epoch in range(epoches):
        err = 0
        for x,y in tqdm(zip(x_train,y_train)):
            # forward
            output = predict(network,x)
            err += loss(y, output)
            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                layer.zero_grad()
                grad = layer.backward(grad)
                layer.optimize(learning_rate)

        err /= len(x_train)
        print(f'Epoch [{epoch+1}/{epoches}], Loss: {err:.5f}')

def batch_train(network, loss, loss_prime, x_train,y_train, 
                batch_size = 25, learning_rate = 0.002,delta = 0.9,
                shuffle = True, print_turn = 20):
    x = np.copy(x_train)
    y = np.copy(y_train)
    if shuffle:
        indices = np.random.permutation(x.shape[0])
        x = x[indices]
        y = y[indices]
    # 提取 batch 数据
    x_batches = batch_data(x,batch_size)
    y_batches = batch_data(y,batch_size)
    num_batches = len(x_batches)
    ## 梯度清零
    for layer in (network):
        layer.zero_grad()
    for batch_idx, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
    # for batch_idx, (x_batch, y_batch) in tqdm(enumerate(zip(x_batches, y_batches))):
        err = 0.0
        ## SDG
        for layer in (network):
            layer.SDG_grad(delta)
        for xi,yi in tqdm(zip(x_batch,y_batch)):
        # for xi,yi in zip(x_batch,y_batch):
            output = predict(network,xi)
            err += loss(yi, output)
            # 累加梯度
            grad = loss_prime(yi, output)
            for layer in reversed(network):
                grad = layer.backward(grad)

        # 更新梯度
        for layer in network:
            layer.optimize(learning_rate)
        err /= batch_size
        # 每 print_turn 个 batch 输出一次
        if (batch_idx+1) % print_turn == 0:
            print(f'Batch [{batch_idx+1}/{num_batches}], Loss: {err:.5f}, batch-size: {batch_size}')

def batch_data(data,batch_size):
    if batch_size <=0:
        return np.reshape(data,(data.shape[0],1,*data.shape[1:]))
    num_batches = data.shape[0] // batch_size
    batches = [data[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    remaining = data[num_batches*batch_size:]
    if remaining.size:
        batches.append(remaining)
    return batches

def save(network,path):
    if path == None:
        print('Wrong dir!')
        return
    dir = os.path.dirname(path)
    os.makedirs(dir,exist_ok=True)
    save = []
    idx = 0
    idx2np = dict() #存储参数用的
    for layer in network:
        layer_dict = dict() # 存储基本信息用的
        layer_dict['type'] = str(type(layer))

        if isinstance(layer,Fc):
            layer_dict['init'] = dict()
            layer_dict['init']['input_size'] = layer.weights.shape[1]
            layer_dict['init']['output_size'] = layer.bias.shape[0]
            layer_dict['weights'] = f'param_{idx}'
            idx2np[layer_dict['weights']] = layer.weights
            idx += 1
            layer_dict['bias'] = f'param_{idx}'
            idx2np[layer_dict['bias']] = layer.bias
            idx += 1
        
        elif isinstance(layer,Reshape):
            layer_dict['init'] = dict()
            layer_dict['init']['input_shape'] = layer.in_shape
            layer_dict['init']['output_shape'] = layer.out_shape

        elif isinstance(layer,Conv):
            layer_dict['init'] = dict()
            layer_dict['init']['input_shape'] = layer.input_shape
            d,_,k,_ = layer.kernels_shape
            layer_dict['init']['kernel_size'] = k
            layer_dict['init']['depth'] = d
            layer_dict['kernels'] = f'param_{idx}'
            idx2np[layer_dict['kernels']] = layer.kernels
            idx += 1
            layer_dict['biases'] = f'param_{idx}'
            idx2np[layer_dict['biases']] = layer.biases
            idx += 1

        elif isinstance(layer,Pool):
            layer_dict['init'] = dict()
            layer_dict['init']['pool_size'] = layer.pool_size
            layer_dict['init']['stride'] = layer.stride

        save.append(layer_dict)
    
    with open(path,'w') as file:
        json.dump(save,fp=file,indent=4)
    np.savez(path+'.npz',**idx2np)

def load(path):
    with open(path) as f:
        json_dict = json.load(f)
    np_load = np.load(path + '.npz')
    network = []
    for layer_dict in json_dict:
        clazz_name = layer_dict['type']
        layer = None
        # Fc
        if clazz_name == str(Fc):
            layer = Fc(**layer_dict['init'])
            layer.weights = np_load[layer_dict['weights']]
            layer.bias = np_load[layer_dict['bias']]
        # Conv
        elif clazz_name == str(Conv):
            layer = Conv(**layer_dict['init'])
            layer.kernels = np_load[layer_dict['kernels']]
            layer.biases = np_load[layer_dict['biases']]
        # Reshape
        elif clazz_name == str(Reshape):
            layer = Reshape(**layer_dict['init'])
        # Pool
        elif clazz_name == str(Pool):
            layer = Pool(**layer_dict['init'])
        # Actv
        elif clazz_name == str(Tanh):
            layer = Tanh()
        elif clazz_name == str(Sigmoid):
            layer = Sigmoid()
        elif clazz_name == str(Softmax):
            layer = Softmax()
        elif clazz_name == str(Relu):
            layer = Relu()
        
        if layer == None:
            print(f'{clazz_name}=加载失败!')
            return
        network.append(layer)
    return network
