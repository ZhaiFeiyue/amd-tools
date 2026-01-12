import torch

m, k, n = 2**0, 4096, 8192

A = torch.randn(m, k, device='cuda',dtype=torch.bfloat16)
B = torch.randn(k, n, device='cuda',dtype=torch.bfloat16)

C1 = torch.matmul(A, B)

torch.cuda.synchronize()

