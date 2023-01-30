import torch
import torch.nn as nn

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()
criterion = nn.MSELoss(size_average=False)  # 损失函数
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)  # 优化器

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()  # 将所有梯度归零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

# Output weight and bias
print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

# 训练好后来预测
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print(f"输入的值为{x_test.item()}， 预测的输出为{y_test.item()}")
