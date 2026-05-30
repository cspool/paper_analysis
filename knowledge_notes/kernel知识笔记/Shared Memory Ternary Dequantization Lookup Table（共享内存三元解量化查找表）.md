## Shared Memory Ternary Dequantization Lookup Table（共享内存三元解量化查找表）

术语是什么？
Shared Memory Ternary Dequantization Lookup Table 是 QMoE 在 Sub1MatVec kernel 中使用的一种 GPU shared memory 数据结构，用于将 2-bit 编码的三元值 {0, 1, 2} 快速转换为可计算的浮点权重 {0, w_min, w_max}。核心设计：deq[3][32*num_warps] 的二维 shared memory 数组，其中每个浮点值在列方向（warp dimension）复制 32 次以避免 bank conflict。原因：28 threads 同时执行 deq[ter][thread] 查找，若不复制，不同 threads 访问不同 ter 值（0/1/2）时 smem 的 bank 冲突（同一 bank 被多个 thread 同时访问）会频繁发生；复制 32× 后，thread i 永远访问 deq[ter][i]——每 thread 独占一列，列内连续 32 个 float 分布在 32 banks，无冲突。

从kernel调度角度拆解术语：
```
// 初始化（每 warp 独立，在 kernel 头部执行一次）
// 3 行 × (32 * num_warps) 列，每列 32 个 float 分布在 32 banks
deq[0][thread] = 0;                                        // ter=0 → 0
deq[1][thread] = __bfloat162float(ter_minmax[row].x);     // ter=1 → w_min
deq[2][thread] = __bfloat162float(ter_minmax[row].y);     // ter=2 → w_max

// 运行时查表（per decoded weight）
int ter = (wx14 >> (4 + 2 * (lane % 14))) & 0x3;  // 提取 2-bit ternary index
float w = deq[ter][thread];  // 查表；每 thread 列独立 → 0 bank conflict
```

为何不用 register 或 constant memory？
- Register: 每个 thread 需存 3 个 float = 太少，不影响；但同一 warp 内不同 threads 的 ter 值可能不同——register 无法"共享"访问
- Constant memory: 读延迟 ~constant cache hit (few cycles) / miss (~L1/L2); 但 28 threads 读同一地址 → broadcast 机制可用，但不如 smem 可控
- Shared memory: 1 cycle latency，无 bank conflict (复制 32× 后)，确定性最低延迟
- 代价：deq table 占用 3 × 32 × num_warps × 4 bytes = 384 × num_warps bytes smem——以 num_warps=4 为例仅 1.5KB，可忽略

术语一般如何实现？如何使用？
- QMoE CUDA kernel 源码中实现，配合 `__syncthreads()` 确保所有 warps 完成 deq 初始化后才开始解码
- 通用技术：任何需要在 GPU warp 内频繁执行 small lookup table 操作的 kernel 均可使用（如 low-bit dequantization）
- 限制：仅当 lookup table 极小（≤ few KB）时才实用——每复制 32× 内存乘 32

涉及论文标题：
- QMoE Sub-1-Bit Compression of Trillion-Parameter Models
