import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True  # 定义w是需要计算梯度的


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# Tensor在做运算的时候会构建计算图，所以下面在更新权重的时候用.data(可读写)或.item()（只读）将值取出不会构建计算图
print("Predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):  # 随机梯度下降
        l = loss(x, y)  # 前馈
        l.backward()  # 反向传播，并释放计算图
        print('\tgrad: ', x, y, w.grad.item())  # 将w的梯度值拿出来，.item()拿出的是一个标量
        w.data -= 0.01 * w.grad.data

        w.grad.data.zero_()  # 每次更新完w后将w的梯度清零
    print("progress:", epoch, w.item(), l.item())

print("predict (after training)", 4, forward(4).item())
