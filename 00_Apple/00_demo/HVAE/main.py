import  torch
from    torch.utils.data import DataLoader
from    torch import nn, optim
from    torchvision import transforms, datasets


from    hvae import HVAE

import  visdom

def main():
    #读取数据集
    mnist_train = datasets.MNIST('mnist', False, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)


    mnist_test = datasets.MNIST('mnist', True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)

    #迭代
    x, _ = next(iter(mnist_test))
    print('x:', x.shape)
    #device转化到cuda上面
    device = torch.device('cuda')
    model = HVAE().to(device)
    #定义损失函数
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)
    #visdom可视化
    viz = visdom.Visdom()

    for epoch in range(1000):


        for batchidx, (x, _) in enumerate(mnist_test):
            # [b, 1, 28, 28]
            x = x.to(device)

            x_hat, kld2 = model(x)
            #计算损失函数
            loss = criteon(x_hat, x)
            #如果kld2不为空
            if kld2 is not None:
                elbo = - loss - 1.0 * kld2
                loss = - elbo
            # backprop
            #梯度清0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(epoch, 'loss:', loss.item(), 'kld2:', kld2.item())

        x, _ = next(iter(mnist_test))
        #对x进行重构
        x = x.to(device)
        with torch.no_grad():
            x_hat, kld = model(x)
        # 进行可视化        一行8张图片      标题
        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))


if __name__ == '__main__':
    main()