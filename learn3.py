import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # 初始权重猜测


def forword(x):
    return x * w


def cost(xs, ys):  # MSE
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forword(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


def gradient(xs, ys):  # 求梯度
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print('Predict (before training)', 4, forword(4))
epoc_list = []
cost_list = []
for epoch in range(100):  # 训练100轮
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.05 * grad_val  # 梯度下降， 学习率 0.05
    print("Epoch:", epoch, 'w=', w, 'loss=', cost_val)
    epoc_list.append(epoch)
    cost_list.append(cost_val)
print('Predoct (after training)', 4, forword(4))

# 画图
plt.plot(epoc_list, cost_list)
plt.ylabel('Cost')
plt.xlabel('Epoc')
plt.show()
