import numpy as np

def func(x, a, b):
    y = x * a+ b
    return y

def loss_func(weight, x, y):
    a, b = weight
    # y'
    y_hat = func(x, a, b)
    #这里loss是均方差
    return np.mean((y_hat - y) ** 2)

from jax import grad
#jax主要就是用来自动求偏导的
'''
def f(x):
    # f(x) = x**2
    # f'(x) = 2 *x
    return x ** 2

#df = f'(x) = 2*x
df = grad(f)

print(df(3.0))
'''
a = np.random.random()
b = np.random.random()
weight = [a, b]
#生成关于xy的随机数
x = np.array([np.random.random() for _ in range(1000)])
#这里目标函数是 y = 3*x + 4
#最终就是看我们的a和b是否能学习的接近3, 4
y = np.array([3*xx + 4 for xx in x])

grad_func = grad(loss_func)

learning_rate = 0.01
for i in range(1000):
    loss = loss_func(weight, x, y)
    print(loss)
    da, db = grad_func(weight, x, y)
    a = a - learning_rate * da
    b = b - learning_rate * db
    weight = [a, b] #更新参数a,b的值

print(a)
print(b)

# NLP 感知机
# MLP 多层感知机