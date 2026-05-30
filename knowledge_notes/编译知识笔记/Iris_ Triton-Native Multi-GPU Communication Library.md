## Iris: Triton-Native Multi-GPU Communication Library

术语是什么？
Iris是第一个完全用Python和Triton实现的多GPU通信库（https://github.com/ROCm/iris），无需外部通信库依赖（如rocSHMEM/NVSHMEM）。它提供tile-based symmetric memory API，使开发者能在单个Triton kernel中无缝交织计算(tl.dot等)和通信(iris.load/store/get/put/copy/atomic_*)。核心设计理念：将通信原语提升为first-class Triton code而非opaque binary blob，使Triton编译器能同时看到计算和通信操作做co-optimization。Host端API提供PyTorch风格的symmetric heap tensor创建和初始化；Device端API提供值语义(load/store，register↔remote memory)和指针语义(get/put/copy，buffer↔buffer)两套通信原语。

从编译框架角度拆解术语：
Iris在Triton编译框架中的位置和作用：
```
┌──────────────────────────────────────────────────────────────┐
│ User Triton Kernel Code (.py)                                │
│   @triton.jit                                                │
│   def my_kernel(A, B, C, heap_bases, ...):                   │
│       c = gemm_loop(A, B)      # 计算                        │
│       iris.store(C+offset, c, ...)  # 通信 ← Iris API       │
│                                                              │
│ Triton Compiler (triton.compile)                             │
│   ↓ 看到 iris.store 的完整实现（非外部二进制）               │
│   ↓ 联合优化：统一register allocation、                       │
│   ↓          instruction scheduling、memory coalescing        │
│   ↓                                                          │
│ Generated Object (AMD GPU: HSACO / NVIDIA: CUBIN)            │
│   - 计算指令(MMA)和通信指令(global load/store with remote     │
│     address)在同一指令流中，编译器可重排指令pipeline           │
└──────────────────────────────────────────────────────────────┘
```

与传统wrapper-based方法的编译流程对比：
```
Wrapper-based (Triton-Distributed, PyTorch Symmetric Memory):
  Triton kernel ──→ Triton compiler ──→ compute IR
  xSHMEM library ──→ pre-compiled binary (opaque) ──→ linked as blob
  → 编译器看不到通信操作的IR → 无法co-optimize

Iris (Native Triton):
  Triton kernel + Iris device code ──→ Triton compiler ──→ unified IR
  → 所有代码都是Triton → 编译器全可见 → full co-optimization
```

术语一般如何实现？如何使用？
Host端：`iris.init()`初始化（PyTorch Distributed + HIP IPC handle交换）→ `iris.zeros/ones/rand()`创建symmetric heap tensor → `iris.get_heap_bases()`获取heap基址数组。Device端：在Triton kernel内调用`iris.load(ptr, to_rank, from_rank, heap_bases)`等API直接跨GPU操作。完整示例：
```
iris.init()
A = iris.randn((M, K), dtype=torch.float16)
B = iris.randn((K, N // world_size), dtype=torch.float16)
heap_bases = iris.get_heap_bases()

@triton.jit
def gemm_all_scatter(A, B, C, heap_bases, M, N, K, ...):
    pid = tl.program_id(0)
    for tile_id in range(pid, total_tiles, NUM_SMS):
        c = gemm_loop(A, B, C, M, N, K, ...)
        for r in range(world_size):
            iris.store(C + offset, c, cur_rank, r, heap_bases, mask=mask)

gemm_all_scatter[(304, 1, 1)](A, B, C, heap_bases, M, N, K, ...)
```
Iris也提供Gluon后端（使用Triton @gluon.jit和@aggregate decorators）封装heap_bases为context对象，改善ergonomics。

涉及论文标题：
- Iris: First-Class Multi-GPU Programming Experience in Triton
