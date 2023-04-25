import torch
import nflows
import nflows.distributions
import nflows.transforms

# 定义流模型
class FlowModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        transforms = []
        for i in range(num_layers):
            transforms.append(nflows.transforms.MaskedAffineAutoregressiveTransform(input_size, hidden_size))
            transforms.append(nflows.transforms.AffineCouplingTransform(input_size, hidden_size))
        self.flow = nflows.transforms.flowSequential(*transforms)

    def forward(self, x):
        z, log_det = self.flow(x)
        return z, log_det


# 定义数据集和数据预处理方法
class SineDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.x = torch.linspace(-10, 10, self.size).reshape(-1, 1)
        self.y = torch.sin(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.size


def normalize_data(x):
    return (x - torch.mean(x)) / torch.std(x)

# 定义训练过程
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x = normalize_data(x.to(device))
        z, log_det = model(x)
        loss = -torch.mean(nflows.distributions.Normal(0, 1).log_prob(z) + log_det)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

# 定义数据加载器
train_dataset = SineDataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlowModel(input_size=1, hidden_size=32, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

import matplotlib.pyplot as plt

# 可视化数据分布
x = train_dataset.x.numpy()
y = train_dataset.y.numpy()
plt.scatter(x, y, s=5)
plt.title("Sine Data Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 可视化模型学习到的分布
z, _ = model(normalize_data(torch.from_numpy(x).to(device)))
z = z.cpu().detach().numpy()
plt.scatter(z, y, s=5)
plt.title("Learned Distribution")
plt.xlabel("z")
plt.ylabel("y")
plt.show()
