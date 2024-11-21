import numpy as np

# output = np.random.randn(9,9)
# print(output)
# pool = output[3:6,0:3]
# idx = np.argmax(pool)
# print(idx)
# print(pool.flat[idx])
# print(np.unravel_index(idx,pool.shape))
# idx = np.argmax(output[3:6,0:3])
# print(idx)
# print(output.flat.index)

# input_shape = (16,224,224)
pool_size = 3
stride = 2
def forward(input):
    in_shape = input.shape ## 存储
    d,h,w = in_shape
    o_h = (h-pool_size) // stride + 1
    o_w = (w-pool_size) // stride + 1
    output =  np.zeros((d,o_h,o_w))
    input_grad_idx = np.zeros(in_shape)
    input_idx = []

    for idx_d in range(d):
        for i in range(o_h):
            for j in range(o_w):
                start_i = i * stride
                start_j = j * stride
                pool = input[idx_d, start_i:start_i+pool_size, start_j:start_j+pool_size]
                output[idx_d][i][j] = np.max(pool)
                idx_h,idx_w = np.unravel_index(np.argmax(pool),pool.shape)
                input_grad_idx[idx_d][i * stride + idx_h][j * stride + idx_w] = 1
                input_idx.append((idx_d,i * stride + idx_h,j * stride + idx_w))

    return output,input_grad_idx,input_idx

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    inputs =  np.random.randn(2,7,7)
    print(inputs)
    outputs,input_grad_idx,input_idx = forward(inputs)
    print(outputs.flatten())
    # print(outputs)
    print(input_grad_idx * inputs)
    print(input_idx)
    for (d,w,h),v in zip(input_idx,outputs.flatten()):
        i = inputs[d,w,h]
        print(f'[{i==v}]d={d},w={w},h={h},input={i}, output={v}')