import numpy as np

# 랜덤값으로 체워진 Weight 행렬 생성, xavier/he initialization
def init_weights(shape):
    return np.random.randn(*shape).astype(np.float32) / np.sqrt(sum(shape))

def softmax(x):
    if len(x.shape) > 1:
        tmp = np.max(x, axis = 1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis = 1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    return x