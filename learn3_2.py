import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # 初始权重猜测


def forword(x):
    return x * w


def loss(x, y):  # MSE
    y_pred = forword(x)
    return (y_pred - y) ** 2


def gradient(x, y):  # 求随机梯度
    return 2 * x * (x * w - y)


print('Predict (before training)', 4, forword(4))
epoc_list = []
loss_list = []
for epoch in range(100):  # 训练100轮
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= 0.05 * grad  # 随机梯度下降， 学习率 0.05
        print('\tgrad: ', x, y, grad)
        l = loss(x, y)
    print("progress:", epoch, "w=", w, "loss=", l)
    epoc_list.append(epoch)
    loss_list.append(l)
print('Predoct (after training)', 4, forword(4))

# 画图
plt.plot(epoc_list, loss_list)
plt.ylabel('Cost')
plt.xlabel('Epoc')
plt.show()
