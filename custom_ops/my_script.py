import torch
import custom_add

# 创建两个张量
input1 = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
input2 = torch.tensor([4, 5, 6], dtype=torch.float32, requires_grad=True)

# 使用自定义加法算子
output = custom_add.custom_add(input1, input2)

# 打印输出张量
print("Output after custom add:", output)

# 计算梯度
output.sum().backward()

# 打印输入张量的梯度
print("Gradient of input1:", input1.grad)
print("Gradient of input2:", input2.grad)
