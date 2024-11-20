import numpy as np
from net.actv import Actv
from net.layer import Layer

class Tanh(Actv):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x)**2

        super().__init__(tanh,tanh_prime)

class Sigmoid(Actv):
    def __init__(self):
        def sigmoid(x):
            return 1/(1+np.exp(-x))

        def sigmoid_prime(x):
            return sigmoid(x) * (1 - sigmoid(x))

        super().__init__(sigmoid, sigmoid_prime)

class Relu(Actv):
    def __init__(self):
        def relu(x):
            return np.maximum(0,x)

        def relu_prime(x):
            return np.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime)

class Softmax(Layer):

    def forward(self, input):
        temp = np.exp(input)
        self.output = temp/np.sum(temp)
        return self.output
    
    def backward(self, output_grad, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output,output_grad)
    