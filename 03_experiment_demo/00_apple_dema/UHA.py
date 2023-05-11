import torch
import hvae 



class UHA:
    '''
        参数：
          f 能量函数，也就是用UHA来优化和采样的函数
          dim 采样空间的维度
          L_m 每次采样时，内部运行的leapforg算法的步数
          eps_m 每次Leapfrog步长的大小
    '''
    def __init__(self, f, dim, L_m=10, eps_m=0.1):
        self.dim = dim
        self.L_m = L_m
        self.eps_m = eps_m
        self.E = f 

    #定义能量函数E
    #def E(self, x):
    #    return torch.sum(x ** 2) / 2

    #定义能量函数的梯度
    def hamiltonian_dynamics(self, mu, logvar):
        # Compute the gradients of the target function with respect to mu and logvar
        grad_mu, grad_logvar = torch.autograd.grad(self.E(mu, logvar), [mu, logvar])
        return grad_mu, grad_logvar


    #HMC步骤
    def HMC_step(self, x, p, eps):
        # 按照哈密顿动力学仿鲿进行一半时间不长的动量更新
        p = p - eps * self.grad_E(x) / 2
        # 按照哈密顿动力学仿鲿进行一整个时间步长的坐标更新
        x = x + eps * p
        # 按照哈密顿动力学方程进行一班时间步长的动量更新
        p = p - eps * self.grad_E(x) / 2
        return x, p
    
    #UHA步骤，未校正哈密顿
    def UHA_step(self, x):
        p = torch.randn_like(x)
        for i in range(self.L_m):
            #使用HMC步骤进行L_m次采样，并更新x和p
            x, p = self.HMC_step(x, p, self.eps_m)
        return x

    #采样函数
    def sample(self, num_samples=1):
        samples = []
        for i in range(num_samples):
            # 初始化当前状态
            x = torch.randn(self.dim).requires_grad_(True)
            # 运行未校正哈密顿算法
            for m in range(1000):
                # 从未校正的状态转移矩阵中采样(从转移核中采样)
                y = self.UHA_step(x)

            # 保存在理目标分布的最终样本(在UHA中没有使用)
            samples.append(y.detach().numpy())

        return samples
    @staticmethod
    def target_function_z(mu, logvar):
        #重参数化z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    #定义函数从q(z|x)中采样
    def q_z(self, x:list, mu, logvar, num:int)->list:
        '''
            参数x: 要采样的样本个数
            参数func: 对那个分布(函数)进行采样
                mu: 编码器学习到的mu
                logvar: 编码器学习到的logvar
            返回值：采样成功以后的样本列表
        '''
        self.E = UHA.target_function_z(mu, logvar)

        #传入进来的是空列表
        sample = self.sample(num)
        return sample

    #定义函数丛p(z|x)中采样
    def p_z(x:list)->list:
        '''
            函数返回值为采样的列表
        '''
        pass


def E(x):
    return torch.sum(x * 2) / 2

uha = UHA(f=E, dim =1)
sample = uha.sample(10)
print(sample)
result = []
mu = torch.tensor(1.)
logvar = torch.tensor(0.)
sample = uha.q_z(result, mu, logvar, 10)