import torch
import torch.nn as nn

class IAF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(IAF, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义变换函数f和反变换函数f_inv
        self.f = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ) for _ in range(num_layers)],
            nn.Linear(hidden_size, input_size)
        )
        
        self.f_inv = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ) for _ in range(num_layers)],
            nn.Linear(hidden_size, input_size)
        )
        
    def forward(self, x):
        # 前向传播算法
        z = self.f(x)
        log_det_jacobian = torch.sum(torch.log(torch.abs(torch.diag_embed(torch.autograd.functional.jacobian(self.f, x)[0]))), dim=1)
        
        return z, log_det_jacobian
    
    def inverse(self, z):
        # 反向传播算法
        x = self.f_inv(z)
        
        return x
    
# 测试代码
input_dim = 10
hidden_dim = 32
num_layers = 2

iaf = IAF(input_dim, hidden_dim, num_layers)
x = torch.randn(32, input_dim)
z, log_det_jacobian = iaf(x)
x_recon = iaf.inverse(z)

print("Input shape:", x.shape)
print("Output shape (z):", z.shape)
print("Output shape (log_det_jacobian):", log_det_jacobian.shape)
print("Output shape (x_recon):", x_recon.shape)