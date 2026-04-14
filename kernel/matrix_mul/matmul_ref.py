"""Reference for: solve(const float* A, const float* B, float* C, int M, int K, int N)"""
def reference(*, A, B, C, M, K, N, **kwargs):
    C[:M * N] = (A[:M * K].reshape(M, K) @ B[:K * N].reshape(K, N)).reshape(-1)
