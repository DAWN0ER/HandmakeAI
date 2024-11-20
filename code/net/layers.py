import numpy as np
from scipy import signal
from net.layer import Layer

class Fc(Layer):
    def __init__(self,input_size,output_size):
        self.weights = np.random.randn(output_size,input_size)
        self.bias = np.random.randn(output_size,1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights,self.input) + self.bias
    
    def backward(self, output_grad, learning_rate):
        w_grad = np.dot(output_grad, self.input.T)
        input_grad = np.dot(self.weights.T, output_grad)

        self.weights -= learning_rate * w_grad
        self.bias -= learning_rate * output_grad

        return input_grad

# 没有实现 padding 和 stride
class Conv(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        in_channel,in_height,in_width =  input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.in_channel = in_channel
        # TODO 考虑 padding 和 stride
        self.output_shape = (depth, in_height - kernel_size + 1, in_width - kernel_size + 1)
        self.kernels_shape = (depth, in_channel, kernel_size, kernel_size)

        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
    
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.in_channel):
                self.output[i] += signal.correlate2d(self.input[j],self.kernels[i,j], "valid")

        return self.output

    def backward(self, output_grad, learning_rate):
        kernels_grad = np.zeros(self.kernels_shape)
        input_grad = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.in_channel):
                kernels_grad[i,j] = signal.correlate2d(self.input[j],output_grad[i],"valid")
                input_grad[j] += signal.correlate2d(output_grad[i],self.kernels[i,j],'full')
        
        self.kernels -= learning_rate * kernels_grad
        self.biases -= learning_rate * output_grad
        
        return input_grad

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.in_shape = input_shape
        self.out_shape = output_shape
    
    def forward(self, input):
        return input.reshape(self.out_shape)
    
    def backward(self, output_grad, learning_rate):
        return output_grad.reshape(self.in_shape)