## Symmetric Heap (GPU IPC Memory Sharing)

术语是什么？
Symmetric Heap是OpenSHMEM规范中定义的内存抽象：多进程环境中，每个进程分配一块大小相同、虚拟地址布局对称的内存区域（symmetric heap），所有peer进程可通过相同的偏移量直接访问彼此的heap数据。在GPU场景中，Iris将symmetric heap概念适配到Triton的多GPU编程模型：通过HIP IPC (Inter-Process Communication)机制——hipIpcGetMemHandle导出本地GPU内存句柄、hipIpcOpenMemHandle导入peer句柄——建立跨GPU的对称地址空间。每个GPU rank分配相同大小的symmetric heap，所有tensor均从该heap分配。Iris的__translate函数通过偏移计算（`offset = ptr - from_heap_base`）将本地指针转换为任意peer GPU的远程指针（`remote_ptr = to_heap_base + offset`），实现跨GPU的transparent memory access。

从硬件架构角度拆解术语：
Symmetric Heap在8-GPU系统上的建立流程：
```
Host进程（每GPU一个独立进程）:
  Step 1: hipMalloc在本地GPU分配symmetric heap (e.g., 4 GiB per GPU)
  Step 2: hipIpcGetMemHandle导出IPC handle → PyTorch Distributed all_gather交换所有rank的handles
  Step 3: hipIpcOpenMemHandle打开所有peer handles → 建立所有跨GPU的虚拟地址映射
  Step 4: 构造heap_bases tensor: [base_rank0, base_rank1, ..., base_rank7]
  Step 5: Device kernel可访问heap_bases做指针翻译

GPU Device端 (Triton kernel内):
  @triton.jit
  def __translate(ptr, from_rank, to_rank, heap_bases):
      from_base = tl.load(heap_bases + from_rank)  # 本地GPU heap基址
      to_base = tl.load(heap_bases + to_rank)      # 目标GPU heap基址
      ptr_int = tl.cast(ptr, tl.uint64)            # 指针转整数
      offset = ptr_int - from_base                 # 在本地heap中的偏移
      to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
      translated_ptr = to_base_byte + offset       # 目标GPU上的绝对地址
      return translated_ptr
```

关键硬件条件：
- 需要GPU间直接P2P interconnect（Infinity Fabric或NVLink），不能经host CPU中转
- HIP IPC仅适用于单节点内(intra-node)，跨节点需RDMA（论文未实现）
- Heap bases数组（每GPU一个uint64，8 GPU仅64 bytes）常驻L1 cache，翻译开销近乎为零

术语一般如何实现？如何使用？
Iris提供PyTorch风格的tensor创建函数(zeros/ones/empty/rand等)，所有tensor从symmetric heap分配。Device-side通过iris.load(ptr, to_rank, from_rank, heap_bases)或iris.store/put/get执行跨GPU操作。开发者无需手动管理IPC handles或指针——Iris transparently handles translation。

涉及论文标题：
- Iris: First-Class Multi-GPU Programming Experience in Triton
