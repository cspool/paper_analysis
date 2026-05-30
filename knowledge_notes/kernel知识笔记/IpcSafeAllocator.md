## IpcSafeAllocator

术语是什么？

IpcSafeAllocator 是 ElasticMoE 自定义的 PyTorch 内存分配器，覆盖 PyTorch 默认的 `TorchCachingAllocator`，确保所有模型权重分配使用 CANN 的 IPC 兼容内存标记。标准 PyTorch 分配器使用设备内存池，分配结果通常作为单一内存块管理，无法被跨进程 IPC 共享。IpcSafeAllocator 拦截 `torch.ones()`、`torch.empty()`、`torch.full()` 等核心分配函数，直接调用 IPC 兼容的 `aclrtMalloc`，使分配的张量可被 `rtIpcSetMemoryName`/`rtIpcOpenMemory` 跨进程共享。

从 kernel 调度角度拆解术语：

```
IpcSafeAllocator 伪代码：

class IpcSafeAllocator:
    def allocate(shape, dtype, device):
        size = shape.numel() * dtype.itemsize
        ptr = aclrtMalloc(size, ACL_MEM_MALLOC_HUGE_FIRST | IPC_COMPATIBLE_FLAG)
        return ptr
    def free(ptr):
        aclrtFree(ptr)

// 覆盖 PyTorch 默认分配
torch.ones = patched_ones   // 使用 IpcSafeAllocator 分配
torch.empty = patched_empty
torch.full = patched_full
```

禁用 IpcSafeAllocator 效果：scale-up 延迟 +29%（2.43→3.14s），peak memory +5.4%（275.2→290.0 GB）。延迟增加因回退到非 IPC 内存需额外拷贝步骤；内存增加因无法共享导致两份副本。

术语一般如何实现？如何使用？

基于 Ascend CANN `aclrtMalloc` + IPC 兼容 flag。PyBind11 暴露 C++ allocator。在 CUDA 生态中等效使用 `cudaMalloc` + `cudaIpcGetMemHandle` 兼容的分配策略。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models
