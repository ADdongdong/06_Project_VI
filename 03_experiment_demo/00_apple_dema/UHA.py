import torch
from tqdm import tqdm


class UHA:
    '''
        参数：
          f 能量函数，也就是用UHA来优化和采样的函数
          dim 采样空间的维度
          L_m 每次采样时，内部运行的leapforg算法的步数
          step.size 每次Leapfrog步长的大小
    '''
    def __init__(self, dim, f = None, L_m=10, step_size=0.1):
        self.dim = dim
        self.L_m = L_m
        self.step_size = step_size
        self.E = f 

    
    
    # Define the target function(能量函数E)
    # -logq(z1)
    def target_function(self, mu, logvar):
        # Compute the log probability of a normal distribution with mean mu and variance exp(logvar)
        return -0.5 * torch.sum(mu**2 + torch.exp(logvar) - logvar - 1)



    #定义能量函数的梯度（the Hamiltonian dynamics)
    def hamiltonian_dynamics(self, mu, logvar):
        # Compute the gradients of the target function with respect to mu and logvar
        grad_mu, grad_logvar = torch.autograd.grad(self.target_function(mu, logvar), [mu, logvar])
        return grad_mu, grad_logvar


    #HMC步骤 Define the Hamiltonian Monte Carlo sampler
    '''
        mu 编码器得到的mu,在哈密顿系统中当前的更新值
        logvar 编码器得到的方差logvar,在哈密顿系统中当前的更新值
        momentum 动量,在哈密顿系统中当前的更新值
        self.step_size 步长
    '''
    def HMC_step(self, mu_current, logvar_current, momentum_current):
        # Compute the gradients of the Hamiltonian with respect to mu and logvar
        grad_mu, grad_logvar = self.hamiltonian_dynamics(mu_current, logvar_current)
        # Update the momentum
        #按照哈密顿动力学进行一半时间步长动量更新
        momentum_current = momentum_current - 0.5 * self.step_size * grad_logvar
        # 按照哈密顿动力学进行一整个时间步长坐标更新
        mu_current = mu_current + self.step_size * momentum_current
        logvar_current = logvar_current + self.step_size * grad_mu
        # 按照哈密顿动力学方程进行一半时间步长的动量更新
        momentum_current = momentum_current - 0.5 * self.step_size * grad_logvar

        return mu_current, logvar_current, momentum_current

    
    #UHA步骤，未校正哈密顿
    '''
        参数
            mu_init 从编码器学到的mu,即送入系统的初始值
            logvar_init 从编码器学到的logvar,即送入系统的初始值
    '''
    def UHA_step(self, mu_init, logvar_init):
        #初始化mu_current,logvar_current
        mu_current = mu_init
        logvar_current = logvar_init
        #初始化动量momentum_current 
        momentum_current = torch.randn_like(mu_init).requires_grad_(True)

        #使用HMC步骤进行L_m次采样，并更新mu, logvar, momentum
        for i in range(self.L_m):
            mu_current, logvar_current, momentum_current = self.HMC_step(mu_current, logvar_current, momentum_current)

        #UHA算法去掉了拒绝-接受步骤，直接进行采样
        return mu_current, logvar_current

        

    #采样函数
    def sample(self, mu, logvar, num_samples=1):
        samples = []
        #添加采样进度条
        #qbar = tqdm(range(num_samples))
        for i in range(num_samples):
            # 初始化当前状态
            #mu = torch.randn(1).requires_grad_(True)
            #logvar = torch.randn(1).requires_grad_(True)
            # 运行未校正哈密顿算法, 这里循环的次数表示经过很多次后，
            # MCMC会趋近平稳，达到目标函数
            for m in range(1000):
                # 从未校正的状态转移矩阵中采样(从转移核中采样)
                mu, logvar = self.UHA_step(mu, logvar)

            # 保存在理目标分布的最终样本
            samples.append([mu.detach().numpy(), logvar.detach().numpy()])

        return samples
    

#测试函数测试优化mu和logvar的UHA
def test1():
    #这里的mu和logvar模拟的是变分自编码器生成的mu和logvar
    mu = torch.tensor(1.).requires_grad_(True)
    logvar = torch.tensor(0.).requires_grad_(True)
    uha = UHA(2, None, L_m=10, step_size=0.1)
    result = uha.sample(mu, logvar, 1)
    print(result)

if __name__ == 'main':
    test1()
