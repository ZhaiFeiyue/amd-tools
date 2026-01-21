import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_ddp(rank, world_size):
    """初始化DDP分布式环境 (适配AMD GPU + spawn)"""
    # 基础配置
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'  # 使用未被占用的端口
    os.environ['NCCL_DEBUG'] = 'WARN'     # 减少日志干扰
    os.environ['TORCH_USE_RCCL'] = '1'    # 强制使用RCCL（AMD GPU）
    
    # 修复：先设置GPU设备，再初始化进程组
    torch.cuda.set_device(rank)
    
    try:
        # 初始化进程组（AMD用rccl，NVIDIA用nccl）
        backend = 'nccl' if torch.cuda.get_device_name().lower().find('nvidia') != -1 else 'rccl'
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.Timedelta(seconds=30)  # 增加超时时间
        )
        return rank
    except Exception as e:
        print(f"进程 {rank} 初始化失败: {e}")
        raise

def cleanup_ddp():
    """安全清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def calculate_gemm_tops(
    matrix_size: int = 2048, 
    warmup_steps: int = 5, 
    test_steps: int = 20
) -> float:
    """
    计算GEMM操作的TOPS（适配AMD GPU，降低默认参数避免OOM）
    
    Args:
        matrix_size: 方阵维度（AMD GPU建议从2048开始）
        warmup_steps: 预热步数
        test_steps: 测试步数
    
    Returns:
        单卡TOPS值
    """
    device = torch.cuda.current_device()
    
    # 修复：使用pin_memory和更小的初始矩阵，避免显存爆炸
    A = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32, pin_memory=True)
    B = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32, pin_memory=True)
    
    # 预热（增加同步，确保GPU稳定）
    torch.cuda.synchronize()
    for _ in range(warmup_steps):
        with torch.no_grad():  # 禁用梯度计算，节省显存
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
    
    # 测试阶段
    torch.cuda.synchronize()
    start_time = time.perf_counter()  # 更高精度的计时器
    
    with torch.no_grad():
        for _ in range(test_steps):
            C = torch.matmul(A, B)
            torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    # 计算TOPS
    total_time = end_time - start_time
    avg_time_per_step = total_time / test_steps
    flops_per_gemm = 2 * (matrix_size ** 3)
    tops = (flops_per_gemm / avg_time_per_step) / (10 ** 12)
    
    # 清理显存
    del A, B, C
    torch.cuda.empty_cache()
    
    return tops

def worker(rank, world_size, matrix_size, warmup_steps, test_steps):
    """每个进程的工作函数（增加异常捕获）"""
    try:
        # 初始化DDP
        setup_ddp(rank, world_size)
        
        # 计算TOPS
        single_gpu_tops = calculate_gemm_tops(matrix_size, warmup_steps, test_steps)
        
        # 收集结果（修复：使用CPU tensor避免GPU通信问题）
        single_tops_cpu = torch.tensor(single_gpu_tops, device='cpu')
        all_tops = [torch.tensor(0.0) for _ in range(world_size)]
        dist.all_gather(all_tops, single_tops_cpu)
        
        # 主进程输出
        if rank == 0:
            print("=" * 60)
            print(f"分布式GEMM性能测试 (AMD GPU + Spawn + DDP)")
            print(f"GPU数量: {world_size} | 矩阵尺寸: {matrix_size}×{matrix_size}")
            print("=" * 60)
            for i, tops in enumerate(all_tops):
                print(f"GPU {i} TOPS: {tops.item():.2f}")
            print("-" * 60)
            avg_tops = sum([t.item() for t in all_tops]) / world_size
            total_tops = sum([t.item() for t in all_tops])
            print(f"平均单卡TOPS: {avg_tops:.2f}")
            print(f"总TOPS: {total_tops:.2f}")
            print("=" * 60)
    
    except Exception as e:
        print(f"进程 {rank} 执行失败: {str(e)}")
        raise
    finally:
        # 安全清理
        cleanup_ddp()
        torch.cuda.empty_cache()

def main():
    """主函数（增加前置检查）"""
    # 基础检查
    if not torch.cuda.is_available():
        print("错误：未检测到CUDA设备")
        return
    
    # 配置参数（AMD GPU保守配置）
    WORLD_SIZE = torch.cuda.device_count()
    MATRIX_SIZE = 2048  # 可逐步增大：2048 → 4096 → 8192
    WARMUP_STEPS = 5
    TEST_STEPS = 20
    
    print(f"检测到 {WORLD_SIZE} 个AMD GPU，开始测试...")
    print(f"矩阵尺寸: {MATRIX_SIZE}×{MATRIX_SIZE} (可根据显存调整)")
    
    # 修复：设置多进程上下文
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 启动进程（增加异常捕获）
    try:
        mp.spawn(
            worker,
            args=(WORLD_SIZE, MATRIX_SIZE, WARMUP_STEPS, TEST_STEPS),
            nprocs=WORLD_SIZE,
            join=True,
            daemon=False  # 非守护进程，便于调试
        )
    except Exception as e:
        print(f"进程启动失败: {e}")
        # 强制清理GPU资源
        for i in range(WORLD_SIZE):
            torch.cuda.device(i)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # 限制线程数，避免资源竞争
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()