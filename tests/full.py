import torch
import triton
import triton.language as tl

@triton.jit
def full_kernel(
    output_ptr,  # 输出张量指针
    value,       # 要填充的值
    n_elements,  # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 每个线程块处理的元素数量
):
    # 获取当前线程块的起始索引
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 创建掩码，防止越界
    mask = offsets < n_elements

    # 填充指定值
    tl.store(output_ptr + offsets, tl.where(mask, value, 0), mask=mask)

def triton_full(shape, value, dtype=torch.float32, device="cuda"):
    """
    Triton实现的 full 操作，类似 torch.full。
    :param shape: 输出张量形状
    :param value: 要填充的值
    :param dtype: 数据类型
    :param device: 设备
    :return: 填充好的张量
    """
    output = torch.empty(shape, dtype=dtype, device=device)
    n_elements = output.numel()

    # 计算线程块数量
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # 调用 kernel
    full_kernel[grid](
        output_ptr=output,
        value=value,
        n_elements=n_elements,
        BLOCK_SIZE=1024
    )

    return output

# 测试
if __name__ == "__main__":
    shape = (1024, 1024)
    value = 3.1415
    t = triton_full(shape, value)
    print(t)
    print(torch.all(t == value))  # 验证是否全为 value