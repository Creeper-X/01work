import torch

# 1. 确认 CUDA 可用
print("CUDA 可用:", torch.cuda.is_available())  # 必须为 True
# 2. 确认显卡架构（9.0）
print("显卡架构:", torch.cuda.get_device_capability())  # 输出 (9, 0)
# 3. 测试 CUDA 内核执行（核心验证）
x = torch.randn(10, 10).cuda()
y = torch.matmul(x, x.T)  # 执行矩阵乘法（CUDA 内核操作）
print("CUDA 运算结果:", y.mean())  # 无报错则成功
# 4. 验证依赖兼容
import wilds
print("Wilds 导入正常:", wilds.__version__)  # 输出 1.2.2