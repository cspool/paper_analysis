## Triton Language (GPU Kernel Programming)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Triton 是 OpenAI 于 2019 年发布的 GPU kernel 编程语言和编译器（Tillet et al., 2019, MAPL），旨在降低 GPU 编程门槛同时保持接近手写 CUDA 的性能。Triton 的核心理念是 **block-level programming**：开发者以 Python 语法编写 tile-level 运算（如加载 block、执行 GEMM、写入结果），Triton 编译器自动处理 thread-level 调度、memory coalescing 和 warp-level 优化。

与 CUDA 的对比：
- CUDA：开发者手动管理 thread/warp/block，显式控制 shared memory allocation 和 synchronization
- Triton：开发者仅指定 block 大小和数据访问模式，编译器自动生成 optimized PTX/LLVM IR

Triton kernel 使用 `@triton.jit` 装饰器，通过 `tl.program_id()` 获取 block index，`tl.load/tl.store` 进行 memory 操作，`tl.dot` 执行 tensor core 矩阵乘法。

从编译框架角度拆解术语。

**Triton 编译流程**：

```
Python Code (triton.jit)
    ↓
Triton IR (block-level intermediate representation)
  - Block-level tiling info (BLOCK_M, BLOCK_N, BLOCK_K)
  - Memory access patterns (load, store, atomic)
  - Arithmetic operations (dot, add, mul, etc.)
    ↓
Triton Compiler Backend
  - Thread/warp mapping optimization
  - Shared memory allocation
  - Memory coalescing analysis
  - Loop unrolling & instruction scheduling
    ↓
LLVM IR / PTX
    ↓
GPU Driver (CUDA) → Hardware Execution
```

**Triton kernel 示例（简化版 Attention）**：

```python
@triton.jit
def attention_kernel(Q, K, V, O, stride_q_n, stride_k_n, ...,
                     BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    # Block index
    pid = tl.program_id(0)
    
    # Load Q block from HBM → SRAM
    q_ptrs = Q + pid * BLOCK_N * stride_q_n + ...
    q = tl.load(q_ptrs)  # [BLOCK_N, BLOCK_D]
    
    # Initialize running statistics (online softmax/entmax)
    m_i = tl.full([BLOCK_N], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    # Loop over K,V blocks
    for j in range(0, N, BLOCK_N):
        k = tl.load(k_ptrs + j * ...)  # [BLOCK_N, BLOCK_D]
        v = tl.load(v_ptrs + j * ...)
        
        # Compute scores on SRAM
        s = tl.dot(q, tl.trans(k))  # [BLOCK_N, BLOCK_N]
        
        # Sparse activation (α-entmax or softmax)
        p = sparse_activation(s, alpha, tau)  # block-wise
        
        # Accumulate output
        acc += tl.dot(p, v)
    
    # Write result back to HBM
    tl.store(o_ptrs, acc)
```

术语一般如何实现？如何使用？

1. **安装**：`pip install triton`
2. **基本用法**：定义 `@triton.jit` kernel 并调用 `kernel[(grid,)](args...)` 启动；grid 决定 block 并行度
3. **关键概念**：
   - **Program ID**：每个 block 的唯一标识，用于确定处理的数据范围
   - **tl.load/tl.store**：自动处理边界检查和 memory coalescing
   - **tl.dot**：触发 Tensor Core 进行快速矩阵乘法
   - **tl.constexpr**：编译时常量，用于 block 大小等形状参数
   - **Autotuning**：`@triton.autotune` 装饰器自动搜索最优配置
4. **在 AdaSplash 中的使用**：所有 attention kernel（前向 Halley-Bisection、前向 attention、反向 dQ/dK/dV）均用 Triton 实现，利用 Triton 的 block-level 编程模型精细控制 GPU memory hierarchy (HBM↔SRAM)
5. **生态**：Triton 被用于 vLLM、SGLang 等 serving 框架，以及 PyTorch `torch.compile` 的 Inductor backend

涉及论文标题：
- AdaSplash: Adaptive Sparse Flash Attention
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU (使用 Triton 实现参数化 Pruning Stage Kernel——单一 kernel 通过不同 (b_q, l_c, k) 参数支持所有 stage, key sequence dim 并行类似 FlashDecode split-KV；和 Block Sparse Attention Kernel——FlashAttention-style prefill + FlashDecoding-style decoding + PagedAttention block-based KV 管理)
