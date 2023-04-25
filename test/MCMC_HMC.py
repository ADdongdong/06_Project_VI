import torch
import numpy as np

# 定义目标分布函数
def target_dist(x):
    return torch.exp(-(x ** 2) / 2) / torch.sqrt(2 * torch.tensor(np.pi))

# 定义计算梯度的函数
def compute_grad(x):
    return torch.autograd.grad(target_dist(x), x, create_graph=True)[0]

# 初始化参数
x_init = torch.tensor(0.0, requires_grad=True)
n_steps = 100
step_size = 0.1

# 设置步数和burn-in期
n_steps = 10000
burn_in = 1000

# 初始化样本
samples = torch.zeros(n_steps)

# 进行采样
for i in range(n_steps):
    # 从标准正态分布中采样一个动量
    p_init = torch.randn(1)

    # 计算初始的Hamiltonian
    H_init = -target_dist(x_init) + 0.5 * (p_init ** 2)

    # 计算梯度
    grad = compute_grad(x_init)

    # 进行一步Leapfrog更新
    p = p_init - 0.5 * step_size * grad
    x = x_init + step_size * p
    grad = compute_grad(x)
    p = p - 0.5 * step_size * grad

    # 计算新的Hamiltonian
    H = -target_dist(x) + 0.5 * (p ** 2)

    # 计算接受率
    p_accept = min(1.0, torch.exp(H_init - H))

    # 接受或拒绝新的样本
    u = torch.rand(1)
    if u < p_accept:
        x_init = x
    samples[i] = x_init

# 删除burn-in期
samples = samples[burn_in:]

# 计算期望值和方差
mean = torch.mean(samples)
var = torch.var(samples)

# 输出结果
print(f"Mean: {mean}")
print(f"Variance: {var}")
