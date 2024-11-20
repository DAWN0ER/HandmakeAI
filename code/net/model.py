from net.layers import *
from net.actvs import *
import json,os
import numpy as np

def predict(nwtwork, input):
    output = input
    for layer in nwtwork:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train,y_train,epoches = 1000, learning_rate = 0.002):
    for i in range(epoches):
        err = 0
        for x,y in zip(x_train,y_train):
            # forward
            output = predict(network,x)
            err += loss(y, output)
            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad,learning_rate) 
        err /= len(x_train)
        print(f"epoch:{i+1},err={err}")

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
        sd = dict() # 存储基本信息用的
        sd['type'] = str(type(layer))

        if isinstance(layer,Fc):
            sd['init'] = dict()
            sd['init']['input_size'] = layer.weights.shape[1]
            sd['init']['output_size'] = layer.bias.shape[0]
            sd['weights'] = f'param_{idx}'
            idx2np[sd['weights']] = layer.weights
            idx += 1
            sd['bias'] = f'param_{idx}'
            idx2np[sd['bias']] = layer.bias
            idx += 1

        elif isinstance(layer,Conv):
            sd['init'] = dict()
            sd['init']['input_shape'] = layer.input_shape
            d,_,k,_ = layer.kernels_shape
            sd['init']['kernel_size'] = k
            sd['init']['depth'] = d
            sd['kernels'] = f'param_{idx}'
            idx2np[sd['kernels']] = layer.kernels
            idx += 1
            sd['biases'] = f'param_{idx}'
            idx2np[sd['biases']] = layer.biases
            idx += 1

        save.append(sd)
    
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
            print('加载失败!')
            return
        network.append(layer)
    return network