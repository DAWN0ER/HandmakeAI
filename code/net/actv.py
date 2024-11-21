import numpy as np
from net.layer import Layer

class Actv(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input

        return self.activation(input)
    
    def backward(self, output_grad):
        return np.multiply(output_grad,self.activation_prime(self.input))