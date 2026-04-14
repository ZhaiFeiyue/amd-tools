"""Reference for: solve(const float* A, float* B, int M, int N)"""
import torch

def reference(*, A, B, M, N, **kwargs):
    B[:M * N].reshape(N, M).copy_(A[:M * N].reshape(M, N).T)
