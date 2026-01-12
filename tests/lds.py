import triton
import triton.language as tl
import torch

# 适配AMD GPU的Load/Store Triton内核
@triton.jit
def load_store_kernel(
    input_ptr,    # 输入张量的指针（全局内存）
    output_ptr,   # 输出张量的指针（全局内存）
    n_elements,   # 张量总元素数
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译期常量）
):
    # 1. 计算当前线程块处理的起始索引
    pid = tl.program_id(axis=0)  # 获取当前程序ID（对应线程块ID）
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 块内索引偏移

    # 2. 边界检查：避免访问超出张量范围的内存
    mask = offsets < n_elements

    # 3. Load操作：从全局内存加载数据到寄存器（AMD平台优化：对齐访问）
    # 对于AMD GPU，推荐使用tl.load的align参数（按数据类型对齐，如float32=4字节）
    data = tl.load(input_ptr + offsets, mask=mask, align=4)

    # 可选：对加载的数据进行简单运算（示例：乘2）
    data = data * 2.0

    # 4. Store操作：将处理后的数据写回全局内存
    # AMD平台优化：使用cache参数适配AMD的缓存层级
    tl.store(output_ptr + offsets, data, mask=mask, cache='cg')

# 封装成可调用的函数
def amd_load_store_operation(input_tensor: torch.Tensor) -> torch.Tensor:
    # 确保输入张量在AMD GPU上（且为float32，适配基础示例）
    assert input_tensor.is_cuda and input_tensor.dtype == torch.float32
    output_tensor = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()

    # 配置块大小（AMD GPU推荐64/128/256，适配wave size）
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 启动内核（指定target为amdgcn）
    load_store_kernel[grid](
        input_tensor, output_tensor, n_elements,
        BLOCK_SIZE=BLOCK_SIZE, target='amdgcn'
    )

    return output_tensor

# 测试代码
if __name__ == "__main__":
    # 检查是否有AMD GPU
    if not torch.cuda.is_available() or 'amd' not in torch.cuda.get_device_name(0).lower():
        print("警告：未检测到AMD GPU，代码可能无法正常运行！")
    else:
        # 创建测试张量
        input_tensor = torch.randn(1024 * 1024, dtype=torch.float32, device='cuda')
        # 执行Load/Store操作
        output_tensor = amd_load_store_operation(input_tensor)
        # 验证结果（加载后乘2，存储结果应与原张量乘2一致）
        assert torch.allclose(output_tensor, input_tensor * 2.0)
        print("AMD平台Load/Store内核执行成功！")