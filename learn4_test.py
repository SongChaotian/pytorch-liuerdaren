import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([1.0])
w2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])
w1.requires_grad = True  # 定义w是需要计算梯度的
w2.requires_grad = True  # 定义w是需要计算梯度的
b.requires_grad = True  # 定义w是需要计算梯度的


def forward(x):
    return w1 * x * x + w2 * x + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# Tensor在做运算的时候会构建计算图，所以下面在更新权重的时候用.data(可读写)或.item()（只读）将值取出不会构建计算图
print("Predict (before training)", 4, forward(4).item())

for epoch in range(10000):
    for x, y in zip(x_data, y_data):  # 随机梯度下降
        l = loss(x, y)  # 前馈
        l.backward()  # 反向传播，并释放计算图
        print(f"\tgrad: x={x}, y={y}, "
              f"w1.grad={w1.grad.item()}, "
              f"w2.grad={w2.grad.item()}, "
              f"b.grad={b.grad.item()}")  # 将w的梯度值拿出来，.item()拿出的是一个标量
        # 学习率：0.2
        w1.data -= 0.02 * w1.grad.data
        w2.data -= 0.02 * w2.grad.data
        b.data -= 0.02 * b.grad.data

        w1.grad.data.zero_()  # 每次更新完w后将w的梯度清零
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("progress:", epoch, "w1=", w1.item(), "w2=", w2.item(), "b=", b.item(), "loss=", l.item())

print("predict (after training)", 4, forward(4).item())
