import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist():
    """加载并预处理 MNIST 数据集"""
    print("正在下载/加载 MNIST 数据集")
    #  MNIST 数据 
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    
    # 像素值归一化 
    X = X / 255.0
    
    y = y.astype(int)
    
    # 对标签进行 One-hot 编码 
    num_classes = 10
    y_onehot = np.zeros((y.size, num_classes))
    y_onehot[np.arange(y.size), y] = 1
    
    # 划分训练集和验证集 
    X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# --- 激活函数及其导数  ---
def relu(Z):
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid(Z):
    Z = np.clip(Z, -500, 500)
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_backward(dA, Z):
    s = sigmoid(Z)
    return dA * s * (1 - s)

def tanh(Z):
    return np.tanh(Z)

def tanh_backward(dA, Z):
    return dA * (1 - np.square(np.tanh(Z)))

def softmax(Z):
    # 保证数值稳定性
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# --- 交叉熵损失函数 ---
def compute_loss(Y_pred, Y_true, weights, l2_lambda):
    m = Y_true.shape[0]
    # 添加epsilon防止 log(0)
    epsilon = 1e-15
    core_loss = -np.sum(Y_true * np.log(Y_pred + epsilon)) / m
    
    # 计算 L2 正则化项 
    l2_loss = 0
    for W in weights:
        l2_loss += np.sum(np.square(W))
    l2_loss = (l2_lambda / (2 * m)) * l2_loss
    
    return core_loss + l2_loss

def leaky_relu(Z, alpha=0.01):
    return np.where(Z > 0, Z, Z * alpha)

def leaky_relu_backward(dA, Z, alpha=0.01):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = dZ[Z <= 0] * alpha
    return dZ