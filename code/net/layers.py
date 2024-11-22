import numpy as np
from scipy import signal
from net.layer import Layer

class Fc(Layer):
    def __init__(self,input_size,output_size):
        self.weights = np.random.randn(output_size,input_size)
        self.bias = np.random.randn(output_size,1)
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros(self.bias.shape)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights,self.input) + self.bias
    
    def backward(self, output_grad):
        w_grad = np.dot(output_grad, self.input.T)
        input_grad = np.dot(self.weights.T, output_grad)

        self.weights_grad += w_grad
        self.bias_grad += output_grad

        return input_grad
    
    def zero_grad(self):
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros(self.bias.shape)
    
    def optimize(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad

    def SDG_grad(self, delta):
        self.weights_grad *= delta
        self.bias_grad *= delta

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
        self.kernels_grad = np.zeros(self.kernels.shape)
        self.biases_grad = np.zeros(self.biases.shape)
    
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.in_channel):
                self.output[i] += signal.correlate2d(self.input[j],self.kernels[i,j], "valid")

        return self.output

    def backward(self, output_grad):
        input_grad = np.zeros(self.input_shape)
        now_kernels_grad = np.zeros(self.kernels_shape)

        for i in range(self.depth):
            for j in range(self.in_channel):
                now_kernels_grad[i,j] = signal.correlate2d(self.input[j],output_grad[i],"valid")
                input_grad[j] += signal.correlate2d(output_grad[i],self.kernels[i,j],'full')
        
        self.kernels_grad += now_kernels_grad
        self.biases_grad += output_grad
        
        return input_grad
    
    def zero_grad(self):
        self.kernels_grad = np.zeros(self.kernels.shape)
        self.biases_grad = np.zeros(self.biases.shape)

    def optimize(self, learning_rate):
        self.kernels -= learning_rate * self.kernels_grad
        self.biases -= learning_rate * self.biases_grad

    def SDG_grad(self, delta):
        self.biases_grad *= delta
        self.kernels_grad *= delta

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.in_shape = input_shape
        self.out_shape = output_shape
    
    def forward(self, input):
        return input.reshape(self.out_shape)
    
    def backward(self, output_grad):
        return output_grad.reshape(self.in_shape)
    
# MAX 池化
class Pool(Layer):
    def __init__(self,pool_size,stride):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, input):
        self.in_shape = input.shape
        d,h,w = self.in_shape
        o_h = (h-self.pool_size) // self.stride + 1
        o_w = (w-self.pool_size) // self.stride + 1
        output =  np.zeros((d,o_h,o_w))
        self.input_idx = []

        for idx_d in range(d):
            for i in range(o_h):
                for j in range(o_w):
                    start_i = i * self.stride
                    start_j = j * self.stride
                    pool = input[idx_d, start_i:start_i+self.pool_size, start_j:start_j+self.pool_size]
                    # 最大值
                    output[idx_d][i][j] = np.max(pool)
                    idx_h,idx_w = np.unravel_index(np.argmax(pool),pool.shape)
                    # 记录 idx
                    self.input_idx.append((idx_d,i * self.stride + idx_h,j * self.stride + idx_w))
        return output
    
    def backward(self, output_grad):
        input_grad = np.zeros(self.in_shape)
        for (d,w,h),v in zip(self.input_idx,output_grad.flatten()):
            input_grad[d,w,h] += v
        return input_grad
        