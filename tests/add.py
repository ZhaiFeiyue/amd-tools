import torch
import numpy as np
import time

#[1,8192] x[8192,8192] = [1,8192]

m, k, n = 2**2, 1024, 8192

A = torch.randn(k, n, device='cuda').to(torch.bfloat16)  # 随机初始化矩阵A，直接放到指定设备
B = torch.randn(k, n, device='cuda').to(torch.bfloat16)  # 随机初始化矩阵B

C1 = A + B

torch.cuda.synchronize()

