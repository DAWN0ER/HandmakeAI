class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass

    def backward(self, output_grad):
        pass

    def zero_grad(self):
        pass

    def SDG_grad(self,delta):
        pass

    def optimize(self,learning_rate):
        pass