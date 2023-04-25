import numpy as np
import jax.numpy as jnp
import jax
from sklearn import datasets
from jax import grad
from tqdm import tqdm



# NLP 感知机
# MLP 多层感知机
def mlp(X, W0, b0, W1, b1):
    z = perceptron(X, W0, b0)
    z = jax.nn.tanh(z)#tanh创建一个非线性层，非线性变换
    y = perceptron(z, W1, b1)
    return y

# 感知机 perceptron
def perceptron(X, W, b):
    # X:input, [samples, input_dim]
    # W:weight, [inpyt_dim, output_dim]
    # b:bias, [1, output_dim]
    y = X @ W + b #@ 是矩阵乘法，*是矩阵按元素相乘
    return y

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X.shape)
print(y.shape)
Y = jax.nn.one_hot(y, 3)
print(Y[-3:])
#print(Y)

print(Y.shape)

input_dim = 4 #输入维度为4
hidden_dim = 100 # 隐藏层维度
output_dim = 3 #输出维度

W0 = np.random.randn(input_dim, hidden_dim)
b0 = np.random.randn(1, hidden_dim)
W1 = np.random.randn(hidden_dim, output_dim)
b1 = np.random.randn(1, output_dim)

weights = [
    W0, b0, W1, b1
]


#前向传播
#pyhon中在数组名前面带*表示将数组解开
#比如这里的*weight 其实就是W0, b0, W1, b1
#计算交叉熵，将交叉熵作为loss函数

# priceptron
def loss_func(weights, X , Y):
    y_hat = mlp(X, *weights)
    #使用交叉熵来当做loss
    loss = -jnp.mean(Y*jax.nn.log_softmax(y_hat))
    return loss

#使用grad对loss自动求梯度
grad_func = grad(loss_func)

lerning_rate = 0.01
pbar = tqdm(range(1000))
for i in pbar:
    loss = loss_func(weights, X, Y)
    grads = grad_func(weights, X, Y)
    for j in range(len(weights)):
        weights[j] = weights[j] - lerning_rate *grads[j]
    pbar.set_description(f'{loss:4f}')

y_hat = mlp(X, *weights)
print(y_hat)
preds = y_hat.argmax(-1)#argmx返回数组中最大值的索引
#其中，-1表示沿着数组的最后一个轴进行操作,就是找到每一子数组中的最大值的下标

print(preds)
from sklearn.metrics import accuracy_score
print(accuracy_score(y, preds))
