import numpy as np
from utils import relu, relu_backward, sigmoid, sigmoid_backward, tanh, tanh_backward, softmax

class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        # 进阶优化：使用 He 初始化代替 0.01 的随机初始化 
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros((1, output_dim))
        self.activation = activation
        
        # 缓存前向传播的数据用于反向传播计算 
        self.X = None
        self.Z = None

    def forward(self, X):
        """实现前向传播 """
        self.X = X
        self.Z = np.dot(X, self.W) + self.b
        
        if self.activation == 'relu':
            return relu(self.Z)
        elif self.activation == 'sigmoid':
            return sigmoid(self.Z)
        elif self.activation == 'tanh':
            return tanh(self.Z)
        elif self.activation == 'softmax':
            return softmax(self.Z)
        elif self.activation == 'leaky_relu':
            from utils import leaky_relu
            return leaky_relu(self.Z)
        return self.Z

    def backward(self, dA, l2_lambda=0.0):
        """实现反向传播 """
        """参数说明：
        dA: 后一层传递过来的误差梯度 (对于输出层，配合交叉熵，这里直接传入的是 dZ)
        dX: 传递给前一层的误差梯度
        dW: 当前层权重矩阵的梯度
        db: 当前层偏置的梯度
        l2_lambda: L2正则化系数，用于在权重梯度上施加惩罚
        """
        m = self.X.shape[0]# 获取当前 mini-batch 的样本数量，用于求梯度的平均值
        
        # 1. 计算激活函数的导数 dZ 
        if self.activation == 'relu':
            dZ = relu_backward(dA, self.Z)
        elif self.activation == 'sigmoid':
            dZ = sigmoid_backward(dA, self.Z)
        elif self.activation == 'tanh':
            dZ = tanh_backward(dA, self.Z)
        elif self.activation == 'leaky_relu':
            from utils import leaky_relu_backward
            dZ = leaky_relu_backward(dA, self.Z)
        else:
            dZ = dA 

        # 2. 计算权重的梯度，包含 L2 正则化项的导数，有效避免过拟合
        dW = np.dot(self.X.T, dZ) / m + (l2_lambda / m) * self.W
        
        # 3. 计算偏置的梯度 此时的梯度应该等于所有样本的dZ沿批次维度的平均和
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        # 4. 计算传递给上一层的误差梯度 
        dX = np.dot(dZ, self.W.T)

        return dX, dW, db

class MLP:
    def __init__(self, layer_dims, activations, l2_lambda=0.0):
        self.layers = []
        self.l2_lambda = l2_lambda
        # 根据结构列表动态生成隐藏层和输出层 [cite: 1]
        for i in range(len(layer_dims) - 1):
            self.layers.append(FullyConnectedLayer(layer_dims[i], layer_dims[i+1], activations[i]))

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, Y_pred, Y_true):
        # 交叉熵损失函数配合 Softmax 输出层的梯度直接化简为：Y_pred - Y_true
        dA = Y_pred - Y_true
        grads = []
        
        # 逆序遍历每一层进行反向传播
        for layer in reversed(self.layers):
            dA, dW, db = layer.backward(dA, self.l2_lambda)
            grads.insert(0, (dW, db))
        return grads

    def update_parameters(self, grads, learning_rate):
        """Mini-batch 梯度下降参数更新 """
        for layer, (dW, db) in zip(self.layers, grads):
            layer.W -= learning_rate * dW
            layer.b -= learning_rate * db