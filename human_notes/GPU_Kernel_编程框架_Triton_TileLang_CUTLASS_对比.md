# GPU Kernel 编程框架：Triton / TileLang / CUTLASS 对比 + Grouped GEMM 详解

- 导出时间：2026-06-02 (UTC+8)
- 来源：当前 Claude Code 可见对话上下文（两轮完整问答）
- 保存路径：human_notes/GPU_Kernel_编程框架_Triton_TileLang_CUTLASS_对比.md
- 说明：本对话无特定论文标题，系 GPU Kernel 编程框架的横向技术对比问答。使用描述性文件名。

## 对话记录

### User（第一轮）
triton, tilelang, cutlass更多的例子. groupedgemm 是什么?伪代码需要详细注释

### Claude（第一轮回复概要）
通过 WebSearch + WebFetch + Obsidian omnisearch 搜索了三类内容和 Grouped GEMM 定义，输出了六部分长篇回答：

1. **三者定位对比表**：从语言、抽象层级、后端、学习曲线、性能天花板等维度对比 Triton（Python DSL, Block级）、TileLang（Python DSL + TVM, Block→Warp三级）、CUTLASS（C++ 模板元编程, Warp→Thread 全显式）。

2. **Triton 详解**：
   - 核心编程模型：block-level，`tl.program_id` + `tl.load`/`tl.dot`/`tl.store`
   - 完整 GEMM kernel（逐行注释），含 L2 cache 优化的 GROUP_SIZE_M 分组调度
   - Persistent Grouped GEMM（MoE 场景，H100 TMA）：`grid=(NUM_SMS,1,1)` + device-side loop + 动态 TMA descriptor
   - 关键特性表：`@triton.autotune`, `tl.dot`, `num_stages`, `num_warps`, persistent kernel, TMA

3. **TileLang 详解**：
   - 设计哲学：三级编程模型（Beginner全自动 / Developer显式tile / Expert手写PTX）
   - Level 2 完整 GEMM 代码：`T.Kernel`→`T.alloc_shared`/`T.alloc_fragment`→`T.Pipelined`→`T.copy`→`T.gemm`→`T.Parallel`
   - Level 3 FP8 DeepGemm 模式：e4m3_float8 + 2× Accumulator 精度刷新 + scale fuse
   - 关键原语速查表

4. **CUTLASS 详解**：
   - 3.x 五层分层架构：Device→Kernel→Collective→TiledMMA/TiledCopy→Atom
   - Hopper Grouped GEMM 完整 C++ 代码（`GemmUniversalAdapter`, `GroupArrayProblemShape`, `CollectiveMma`, warp specialization pingpong）
   - Warp Specialization 流水线示意图（Producer-0/Producer-1 TMA load + Consumer WGMMA + 3-stage buffer）

5. **Grouped GEMM 深度解析**：
   - 定义：单次 kernel launch 完成多组不同 M 的矩阵乘法（MoE 核心需求）
   - 数据布局图示（x[M,K] 拼接 + w[G*N,K] 堆叠）
   - 两个完整伪代码算法（注释到每一行）：
     - 算法1：朴素分组调度（每 program 一个 group）
     - 算法2：Persistent 调度（SM 级别 Round-Robin 自调度）
   - MoE Forward + Backward 完整流程

6. **总结**：场景→工具推荐决策树

主要 Sources:
- Triton Matrix Multiplication Tutorial
- PyTorch Blog: Accelerating MoE with Persistent Cache-Aware Grouped GEMM
- TileLang DeepWiki & SegmentFault 三级编程模型
- CUTLASS 3.x GEMM API & NVIDIA Technical Blog
- pytorch-labs/applied-ai Grouped GEMM

### User（第二轮）
triton, tilelang, cutlass分别提供了什么编程接口,能够显式定义到硬件的什么层次(block任务和调度, warp任务和调度, 线程任务和调度..)

### Claude（第二轮回复概要）
进一步搜索了各框架的 API 原语→硬件层次映射，输出了六部分对比：

1. **GPU 硬件抽象层级定义**：Grid(Device)→Block(CTA)→Warp(Warpgroup)→Thread(Lane)→Register/Instruction 五层模型

2. **Triton API→硬件映射**：
   - Grid: 半显式（`grid=(x,y,z)`, `tl.program_id`, `tl.num_programs`）
   - Block: **显式**（`num_warps`, `num_stages`, `BLOCK_SIZE_*`, `tl.debug_barrier()`）
   - Warp: 全自动（实验性 `num_consumer_groups` + `num_buffers_warp_spec`）
   - Thread: 完全自动（无直接 API）
   - Register: 完全自动
   - 指令: 隐式触发（`tl.dot`→mma, `tl.load`→ld/cp/TMA）
   - 核心哲学："写 block 级逻辑，编译器搞定 warp/thread"

3. **TileLang API→硬件映射**（三级模型）：
   - Level 1 (Beginner): 全自动，只指定算法意图
   - Level 2 (Developer): Grid 显式(`T.Kernel`), Block 显式(`T.alloc_shared`, `T.Pipelined`), Warp 半显式(`T.Parallel` 自动 layout), Thread 半显式(`T.alloc_fragment`), 指令 自动选择(`T.gemm`, `T.copy`)
   - Level 3 (Expert): Warp/Thread/Register/指令 全线显式（`T.ldmatrix`, `T.mma`, `T.stmatrix`, `T.ldg128`, `T.stg256`, `T.annotate_layout`, `T.set_max_nreg`）
   - "可以像 Triton 写 block 代码，也可以下探到 warp/thread 细节"

4. **CUTLASS 3.x API→硬件映射**（五层正交抽象）：
   - Layer 5 (Device): Host 端，CUDA stream + workspace
   - Layer 4 (Kernel): Grid，Tile Scheduler (Persistent/Stream-K/Basic)
   - Layer 3 (Collective): Block/CTA，Pipeline+Warp Specialization+同步
   - Layer 2 (TiledMMA/TiledCopy): Warp/Warpgroup，CuTe Layout 代数精确到 lane-level
   - Layer 1 (Atom): Instruction，精确 MMA 变体 + Copy 指令
   - 每层独立可替换，全显式控制

5. **最终对比大表**：Grid/Block/Warp/Thread/Register/指令 六个层次在三个框架中的显式程度对比，标注每个框架最擅长的层次

6. **一句话总结**：
   - Triton: "我告诉 block 做什么，编译器决定 warp/thread"
   - TileLang: "可以 block 级也可以下探到 warp/thread"
   - CUTLASS: "每层每个 warp 每条指令我全管"

## 已产生的文件或修改

- 无文件修改。本对话为纯技术问答，仅产出 Obsidian vault 搜索结果缓存文件和 WebFetch 缓存（临时文件）。

## 后续待办

- 可考虑将 Triton/TileLang/CUTLASS 编程接口对比写入 `knowledge_notes/` 作为系统性知识点。
- Grouped GEMM 的伪代码可补充到 MoE 相关的 knowledge note 中。

---

---

## 对话记录补充：2026-06-03 17:29 CST

当前可见上下文包含较早的 SGLang 提问，但对应 Codex 最终回复在当前上下文中不可见；以下保存当前可见的 GPU Kernel / Triton / TileLang / CUTLASS / GroupedGEMM 相关用户输入与 Codex 最终输出。

### 001 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- SKILL.md: .codex/skills/obsidian-keyword-explainer/SKILL.md
- Q1.4_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.4_L1_answer.md
- Q1.2_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.2_L1_answer.md
- review_summary_skill.md: draft/review_summary_skill.md

## My request for Codex:
'### 方法5: Triton / CUTLASS / TileLang — Kernel 层编程与编译

**笔记证据**: `knowledge_notes/编译知识笔记/TileLang（可组合的 Tiled 编程模型）.md` (score: 672.8); `knowledge_notes/编译知识笔记/Tile-based Programming Language (TileLang DSL).md` (score: 2410.9)

**方法细节**（L1 粒度 —— 三层 Kernel 编程抽象对比）:

```
Kernel 编程框架的层次对比:
===========================

┌─────────────────────────────────────────────────────────┐
│ Layer 3: Python DSL (Triton / TileLang Beginner)        │
│   用户用 Python 描述 block-level 计算                    │
│   编译器自动管理 shared memory, thread binding           │
│   示例: @triton.jit def matmul_kernel(A, B, C, M, N, K) │
├─────────────────────────────────────────────────────────┤
│ Layer 2: Tile-Level 编程 (TileLang Developer)            │
│   用户显式描述 tile 数据流: T.load, T.gemm, T.reduce     │
│   Layout Inference: 自动推导内存布局 (row/col/swizzled)  │
│   Pipeline Inference: 自动插入 async copy + barrier     │
│   示例: ~50 行 TileLang 实现 DeepSeek MLA, 95%+ CUDA 性能│
├─────────────────────────────────────────────────────────┤
│ Layer 1: 细粒度控制 (CUTLASS / TileLang Expert)          │
│   C++ template: thread tile, warp tile, CTA tile 全暴露  │
│   CuTe (CUTLASS 3.x): 多维线程/数据布局抽象              │
│   支持: FP64/FP32/TF32/FP16/BF16/FP8/INT8/INT4/INT1     │
│   后端: CUDA (PTX→SASS), ROCm (HIP), NPU (Ascend C)     │
└─────────────────────────────────────────────────────────┘

// Triton 示例: Fused Online Softmax kernel
@triton.jit
def online_softmax(X, Y, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]  // [BLOCK_M, 1]
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    d_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for start_n in range(0, N, BLOCK_N):
        x = tl.load(X + rows*N + (tl.arange(0, BLOCK_N)[None,:] + start_n))
        m_prev = m_i
        m_i = tl.maximum(m_i, tl.max(x, axis=1))
        d_i = d_i * tl.exp(m_prev - m_i) + tl.sum(tl.exp(x - m_i[:, None]), axis=1)
    
    y = tl.exp(x - m_i[:, None]) / d_i[:, None]
    tl.store(Y + rows*N + tl.arange(0, BLOCK_N)[None,:], y)
```

**注解**:
- **编译流程**: Triton (Python→Triton-IR→MLIR→PTX→SASS)；TileLang (Python→TileLang IR→CUTE/PTX/HIP/NPU 指令)
- **TileLang 跨平台**: 支持 "CUDA GPUs (H100/A100/V100)、ROCm GPUs (MI300/MI250) 和国产加速器"
- **CUTLASS GroupedGEMM**: MoE 推理的关键 kernel——将不同 expert 的 GEMM 合并为一个 GroupedGEMM kernel，避免逐个 expert 的 kernel launch overhead

---'教我,解释伪代码

### 002 Codex Final
我用 `$obsidian-keyword-explainer` 的口径来讲：这段不是在讲“一个算法”，而是在讲 **怎么写高性能 GPU/NPU kernel**，从易用到可控分三层。

**语义拆分**
| Segment | Keywords | 作用 |
|---|---|---|
| S1 | Triton, TileLang, CUTLASS | 三种 kernel 编程抽象对比 |
| S2 | Online Softmax, tile, block | 示例 kernel 要计算什么 |
| S3 | GroupedGEMM, MoE expert | 为什么这些框架对 MoE 推理重要 |

**先抓主线**
Triton / TileLang / CUTLASS 都是为了同一件事：把 PyTorch 里多个零散算子，改写成一个更贴近硬件的数据搬运 + 计算流程。

区别是：

| 框架 | 你在控制什么 | 难度 | 适合场景 |
|---|---|---:|---|
| Triton | block/tile 级计算 | 中 | fused softmax、layernorm、matmul、小型自定义 kernel |
| TileLang | tile 数据流 + 自动 layout/pipeline | 中高 | attention、MLA、跨 CUDA/ROCm/NPU 的复杂 kernel |
| CUTLASS | CTA/warp/thread/MMA 全层细节 | 高 | 极致 GEMM、GroupedGEMM、Tensor Core 深度优化 |

一句话记：

> Triton 像“Python 写 GPU tile kernel”；TileLang 像“显式描述 tile 数据流，让编译器补 layout 和 pipeline”；CUTLASS 像“C++ 模板级别操控 Tensor Core GEMM”。

**伪代码在算什么**
这个 Triton 示例想实现的是 **按行 softmax**：

$$
Y_{i,j} = \frac{e^{X_{i,j} - m_i}}{\sum_k e^{X_{i,k} - m_i}}
$$

其中：

$$
m_i = \max_j X_{i,j}
$$

问题是，如果一行很长，不能一次把整行全部放进 SRAM/register，所以它把列维度 `N` 切成很多 `BLOCK_N` 小块，边读边更新 running max 和 running sum。这就是 **Online Softmax**。

核心递推是：

$$
m_{new}=\max(m_{old}, \max(x_{block}))
$$

$$
d_{new}=d_{old}\cdot e^{m_{old}-m_{new}}+\sum e^{x_{block}-m_{new}}
$$

为什么要乘 `exp(m_old - m_new)`？因为最大值基准变了。之前的和是以 `m_old` 为中心算的，现在全局最大值可能变成 `m_new`，旧的分母必须重新缩放到新坐标系里。

**逐行解释**
```python
@triton.jit
def online_softmax(X, Y, M, N, BLOCK_M, BLOCK_N):
```

`@triton.jit` 表示这不是普通 Python 函数，而是 Triton kernel，会被编译成 GPU 代码。`BLOCK_M`、`BLOCK_N` 是 tile 大小：一个 program 处理 `BLOCK_M` 行、每次读取 `BLOCK_N` 列。

```python
pid = tl.program_id(0)
```

当前 GPU program/block 的编号。可以理解成 CUDA 里的 block id。

```python
rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
```

这个 program 负责哪些行。  
如果 `BLOCK_M=4`，`pid=2`，那它负责第 `8,9,10,11` 行。`[:, None]` 是为了后面和列索引 broadcast 成 `[BLOCK_M, BLOCK_N]` 的二维 tile。

```python
m_i = tl.full((BLOCK_M,), -inf)
d_i = tl.zeros((BLOCK_M,))
```

每一行维护两个状态：

- `m_i`：当前已经看过的列块里的最大值
- `d_i`：当前 softmax 分母，也就是 shifted exp sum

```python
for start_n in range(0, N, BLOCK_N):
    x = tl.load(...)
```

沿列方向一块一块读取。`x` 的形状大概是：

```text
[BLOCK_M, BLOCK_N]
```

也就是多行、多列的一小块。

```python
m_prev = m_i
m_i = tl.maximum(m_i, tl.max(x, axis=1))
```

先保存旧最大值，再用当前 tile 的行最大值更新 running max。

```python
d_i = d_i * tl.exp(m_prev - m_i) + tl.sum(tl.exp(x - m_i[:, None]), axis=1)
```

这是最关键的一行。

它做两件事：

1. `d_i * exp(m_prev - m_i)`：把旧分母缩放到新的最大值基准。
2. `sum(exp(x - m_i[:, None]))`：加入当前 tile 的贡献。

`m_i[:, None]` 把 `[BLOCK_M]` 变成 `[BLOCK_M, 1]`，这样每一行都用自己的最大值去减。

**这个伪代码有个重要小坑**
原文最后：

```python
y = tl.exp(x - m_i[:, None]) / d_i[:, None]
tl.store(..., y)
```

这只会存 **最后一个 `BLOCK_N` 列块** 的 softmax 结果。因为循环结束后，变量 `x` 只保留了最后一次加载的 tile。

所以它表达了 Online Softmax 的核心思想，但不是完整可用的 row-wise softmax kernel。

完整写法通常要么两遍：

```python
# pass 1: online 计算每行 global max 和 denominator
m = -inf
d = 0
for block in columns:
    x = load(X_block)
    m_new = max(m, rowmax(x))
    d = d * exp(m - m_new) + rowsum(exp(x - m_new))
    m = m_new

# pass 2: 再次读取每个 block，写出 softmax
for block in columns:
    x = load(X_block)
    y = exp(x - m) / d
    store(Y_block, y)
```

Annotations:

- `m` 是每行全局最大值。
- `d` 是每行 softmax 分母。
- 第一遍只算归一化统计量。
- 第二遍才真正写出所有列的 `Y`。
- 真实 Triton 代码还需要 `mask=(rows < M) & (cols < N)`，避免越界读取。

如果是 FlashAttention，就不需要把整个 softmax 矩阵 `Y` 写出来，而是边算边乘 `V`：

```python
m = -inf
l = 0
acc = 0

for kv_block in KV_blocks:
    scores = Q @ K_block.T
    m_new = max(m, rowmax(scores))

    alpha = exp(m - m_new)
    p = exp(scores - m_new)

    acc = acc * alpha + p @ V_block
    l = l * alpha + rowsum(p)
    m = m_new

O = acc / l
```

Annotations:

- `acc` 是未归一化 attention 输出。
- `l` 是 softmax 分母。
- `alpha` 用来重缩放旧输出。
- 这就是 FlashAttention 能不 materialize `N×N` attention matrix 的原因。

**GroupedGEMM 为什么放在这里**
MoE 里每个 expert 都要做 FFN GEMM，但每个 expert 分到的 token 数不同：

```text
Expert 0: 128 tokens
Expert 1: 0 tokens
Expert 2: 43 tokens
Expert 3: 271 tokens
```

如果每个 expert 单独 launch 一个 GEMM kernel，开销很大。GroupedGEMM 把这些不同大小的 GEMM 打包成一次 kernel launch。CUTLASS 在这里很常用，因为它能精细控制 GEMM tile、warp、epilogue 和 Tensor Core。

**怎么选**
写自定义 fused kernel，先用 Triton。  
要表达复杂 attention/MLA tile 数据流，并希望跨硬件，考虑 TileLang。  
要 MoE GroupedGEMM 或极限 GEMM 性能，走 CUTLASS/cuBLASLt 这一类底层库。

笔记证据：`knowledge_notes/编译知识笔记/Triton (GPU Kernel Language and Compiler).md`、`knowledge_notes/编译知识笔记/CUTLASS.md`、`knowledge_notes/编译知识笔记/TileLang（可组合的 Tiled 编程模型）.md`、`knowledge_notes/算法知识笔记/Online Softmax _ Tiled Softmax Computation.md`、`knowledge_notes/kernel知识笔记/GroupedGEMM (Grouped General Matrix Multiplication).md`。

### 003 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- SKILL.md: .codex/skills/obsidian-keyword-explainer/SKILL.md
- Q1.4_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.4_L1_answer.md
- Q1.2_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.2_L1_answer.md
- review_summary_skill.md: draft/review_summary_skill.md

## My request for Codex:
triton, tilelang, cutlass更多的例子.  groupedgemm打包后4个expert的权重矩阵需要同时加载?

### 004 Codex Final
对，你这个问题抓到了 GroupedGEMM 的核心：**“打包”是调度打包，不是把 4 个 expert 的完整权重矩阵同时搬进 shared memory。**

**先给更多例子**
Triton 适合写这种“一个 PyTorch 算子太慢，想 fusion”的 kernel。

例子 1：fused bias + GELU。

```python
@triton.jit
def bias_gelu_kernel(X, B, Y, N, BLOCK):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    z = x + b

    y = 0.5 * z * (1.0 + tl.tanh(0.79788456 * (z + 0.044715 * z * z * z)))
    tl.store(Y + offs, y, mask=mask)
```

这里 Triton 的思维是：一个 program 处理一段连续元素，`tl.arange` 生成 tile 内索引，`tl.load/tl.store` 搬数据，elementwise 计算在寄存器里完成。适合 LayerNorm、RMSNorm、Softmax、RoPE、quant/dequant、small fused op。

TileLang 更像“我描述 tile 数据流，你帮我推 layout 和 pipeline”。

概念伪代码：

```python
for block_m, block_n in output_tiles:
    acc = zeros([BM, BN])

    for block_k in T.Pipelined(range(0, K, BK), num_stages=3):
        a_tile = T.load(A[block_m, block_k])   # global -> shared/register
        b_tile = T.load(B[block_k, block_n])
        acc += T.gemm(a_tile, b_tile)          # Tensor Core / Matrix Core

    T.store(C[block_m, block_n], acc)
```

TileLang 的价值在于：你显式说清楚 tile 级数据流，比如 `load -> gemm -> reduce/store`，但 layout swizzle、async copy、barrier、pipeline stage 可以让编译器推。它特别适合 MLA、attention、dequantized GEMM 这种比 Triton 更复杂、但又不想手写 CUTLASS/CUDA 的场景。

CUTLASS 更底层，适合“我要控制 GEMM 到 warp/MMA 级别”。

普通 GEMM 的 CUTLASS 心智模型：

```cpp
GemmConfig:
  CTA tile:        128 x 128 x 64
  Warp tile:        64 x  64 x 64
  MMA instruction:  16 x   8 x 16
  Epilogue: bias + activation

for each CTA output tile:
    for k_tile in K:
        load A tile to shared memory
        load B tile to shared memory
        warp-level MMA
    epilogue
    store C tile
```

它难，但强：GEMM、GroupedGEMM、Conv implicit GEMM、FP8/INT4/FP4、epilogue fusion 都是 CUTLASS 的主战场。

**GroupedGEMM 到底怎么打包**
MoE 里假设有 4 个 expert：

```text
Expert 0: 收到 128 tokens
Expert 1: 收到   0 tokens
Expert 2: 收到  43 tokens
Expert 3: 收到 271 tokens
```

每个 expert 都要算：

```text
A_e [M_e, d_model] x W_e [d_model, d_ff] -> C_e [M_e, d_ff]
```

GroupedGEMM 会构造一组问题：

```python
groups = [
    {expert: 0, A: tokens_e0, B: W0, M: 128, N: d_ff, K: d_model},
    # expert 1 没 token，通常跳过
    {expert: 2, A: tokens_e2, B: W2, M: 43,  N: d_ff, K: d_model},
    {expert: 3, A: tokens_e3, B: W3, M: 271, N: d_ff, K: d_model},
]

cutlass_grouped_gemm(groups)
```

重点来了：kernel launch 只有一次，但里面的 CTA/tile 调度会把不同 expert 的 GEMM tile 分配给不同 SM。

概念上像这样：

```text
CTA 0 -> Expert 0, output tile (m0, n0)
CTA 1 -> Expert 3, output tile (m0, n0)
CTA 2 -> Expert 2, output tile (m0, n1)
CTA 3 -> Expert 3, output tile (m1, n0)
...
```

每个 CTA 内部才做：

```python
for k_tile in K_tiles:
    load A_e[m_tile, k_tile] into shared/register
    load W_e[k_tile, n_tile] into shared/register
    acc += mma(A_tile, W_tile)
store C_e[m_tile, n_tile]
```

所以回答你的问题：

**不需要 4 个 expert 的完整权重矩阵同时加载。**

更准确地说：

| 层级 | 是否同时有 4 个 expert 权重？ |
|---|---|
| HBM/global memory | 权重本来都常驻 GPU 显存里 |
| L2 cache | 可能同时缓存多个 expert 的部分权重 tile |
| shared memory | 每个 CTA 只放自己正在算的那个 expert 的一小块 `W_e[k_tile, n_tile]` |
| registers | 每个线程只拿更小的 MMA fragment |
| kernel launch | 是一次 launch 管多个 expert 的 GEMM |

因此，“同时”只发生在 **tile 粒度**：多个 SM/CTA 可能同时处理不同 expert，于是同时从 HBM/L2 读取不同 expert 的权重切片。但不是把 `W0/W1/W2/W3` 整块矩阵一次性搬进片上内存。

一个形象版本：

```text
GroupedGEMM 打包的是:
  [问题列表 + 指针 + shape + 调度]

不是打包成:
  [W0 整块 + W1 整块 + W2 整块 + W3 整块] -> shared memory
```

这就是 GroupedGEMM 比“每个 expert 单独 GEMM”快的原因：它减少 kernel launch 开销，并让小 batch、不均匀 expert 的 tile 可以混在一个调度器里填满 SM。真正的数据加载仍然是按 tile、按 K-loop 分批进行。

### 005 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- SKILL.md: .codex/skills/obsidian-keyword-explainer/SKILL.md
- Q1.4_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.4_L1_answer.md
- Q1.2_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.2_L1_answer.md
- review_summary_skill.md: draft/review_summary_skill.md

## My request for Codex:
所有例子的伪代码都给出注释,并尽量完整

### 006 Codex Final
下面给你一版“尽量完整但不爆炸”的笔记版。先统一符号：

```text
M, N, K     : GEMM/矩阵维度
BM, BN, BK : tile/block 大小
CTA         : CUDA thread block / Triton program 的近似对应物
HBM         : GPU global memory
SMEM        : shared memory
REG         : registers
```

**1. Triton: Fused Bias + GELU**
```python
# Host side: 一个 Triton kernel launch
grid = (ceil_div(N, BLOCK),)
bias_gelu_kernel[grid](X, B, Y, N, BLOCK)

@triton.jit
def bias_gelu_kernel(X, B, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # 当前 program id，类似 CUDA blockIdx.x

    offs = pid * BLOCK + tl.arange(0, BLOCK)     # 当前 program 负责的一段连续元素
    mask = offs < N                              # 防止最后一个 block 越界

    x = tl.load(X + offs, mask=mask)             # 从 HBM 读 X tile 到寄存器
    b = tl.load(B + offs, mask=mask)             # 从 HBM 读 bias tile 到寄存器

    z = x + b                                    # 在寄存器里做 add

    # GELU 近似公式，仍然在寄存器中完成
    y = 0.5 * z * (1.0 + tl.tanh(0.79788456 * (z + 0.044715 * z * z * z)))

    tl.store(Y + offs, y, mask=mask)             # 写回 HBM
```

Annotations:

- 这个例子展示 Triton 最典型用法：`load -> elementwise compute -> store`。
- `BLOCK` 个元素由一个 program 处理，Triton 编译器再把它映射到线程/warp。
- 这里没有显式 shared memory，因为 elementwise 算子只需要寄存器即可。
- 它把 PyTorch 里的 `add` 和 `gelu` 两个算子融合成一个 kernel，减少 HBM 读写和 kernel launch。

**2. Triton: 完整 Two-Pass Online Softmax**
```python
@triton.jit
def softmax_kernel(X, Y, M, N, stride_xm, stride_ym,
                   BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)                         # 每个 program 负责 BM 行

    rows = pid_m * BM + tl.arange(0, BM)[:, None]    # [BM, 1]
    cols_base = tl.arange(0, BN)[None, :]            # [1, BN]

    m = tl.full((BM,), -float("inf"), tl.float32)     # 每行 running max
    d = tl.zeros((BM,), tl.float32)                  # 每行 running denominator

    # Pass 1: 扫完整行，计算全局 max 和 denominator
    for start_n in range(0, N, BN):
        cols = start_n + cols_base                   # 当前列 tile
        mask = (rows < M) & (cols < N)

        x = tl.load(X + rows * stride_xm + cols,
                    mask=mask, other=-float("inf"))  # [BM, BN]

        block_m = tl.max(x, axis=1)                  # 当前 tile 每行最大值
        m_new = tl.maximum(m, block_m)               # 更新 running max

        # 旧 denominator 需要缩放到新的 max 坐标系
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new[:, None]), axis=1)
        m = m_new

    # Pass 2: 再扫一遍，真正写出每个元素的 softmax
    for start_n in range(0, N, BN):
        cols = start_n + cols_base
        mask = (rows < M) & (cols < N)

        x = tl.load(X + rows * stride_xm + cols,
                    mask=mask, other=-float("inf"))

        y = tl.exp(x - m[:, None]) / d[:, None]      # 用 Pass 1 得到的 m,d 归一化

        tl.store(Y + rows * stride_ym + cols, y, mask=mask)
```

Annotations:

- 这是比你原文更完整的版本：原文只会写最后一个 `BN` tile，这里第二遍会写完整输出。
- `m` 和 `d` 都是 `[BM]`，即每一行维护一份状态。
- Online Softmax 的关键是这行：`d_old * exp(m_old - m_new)`，因为最大值基准变了，旧分母要重缩放。
- 如果是 attention，通常不显式存 softmax 矩阵，而是把 `exp(score)` 直接乘上 `V` 累加到输出。

**3. TileLang: Tiled GEMM 数据流**
```python
@tilelang.jit
def tiled_gemm(A, B, C, M, N, K):
    for bm, bn in T.grid(ceil_div(M, BM), ceil_div(N, BN)):
        acc = T.alloc_fragment([BM, BN], dtype="float32")  # REG accumulator
        T.clear(acc)

        # Pipelined 表示 K 维 tile 循环可做多 stage overlap
        for bk in T.Pipelined(range(0, K, BK), num_stages=3):
            As = T.alloc_shared([BM, BK], dtype="float16") # SMEM tile of A
            Bs = T.alloc_shared([BK, BN], dtype="float16") # SMEM tile of B

            T.copy(A[bm*BM : bm*BM+BM, bk : bk+BK], As)   # HBM -> SMEM
            T.copy(B[bk : bk+BK, bn*BN : bn*BN+BN], Bs)   # HBM -> SMEM

            T.gemm(As, Bs, acc)                           # Tensor Core MMA

        T.copy(acc, C[bm*BM : bm*BM+BM, bn*BN : bn*BN+BN]) # REG -> HBM
```

Annotations:

- TileLang 的重点是“数据流”：加载 tile、做 GEMM、累加、写回。
- `T.Pipelined(..., num_stages=3)` 让编译器插入 async copy、barrier、stage buffer。
- 用户不用像 CUTLASS 那样手写所有 warp/thread 细节，但比 Triton 更明确地描述 SMEM tile 和 pipeline。
- 片上只保存当前 `A/B` tile，不保存完整矩阵。

**4. CUTLASS: GEMM 主循环心智模型**
```cpp
// Host side: 用模板配置 GEMM kernel
GemmConfig:
    ElementA = half
    ElementB = half
    ElementC = half
    Accumulator = float

    ThreadblockShape = 128 x 128 x 64   // 一个 CTA 处理的 C tile
    WarpShape        =  64 x  64 x 64   // 一个 warp 处理的 tile
    InstructionShape =  16 x   8 x 16   // Tensor Core MMA 指令形状

launch_cutlass_gemm(A, B, C, M, N, K, config)

# Kernel internal pseudocode
for each CTA_tile_C in grid:
    acc = zeros([128, 128])                  # 每个 CTA 的 accumulator fragments

    for k0 in range(0, K, 64):
        async_load A_tile[128, 64] to SMEM   # producer warp / cp.async / TMA
        async_load B_tile[64, 128] to SMEM

        wait_until_tile_ready()              # barrier / pipeline sync

        for each warp_tile inside CTA:
            mma(acc_fragment, A_fragment, B_fragment)

    epilogue(acc)                            # bias / activation / cast / quant
    store C_tile to HBM
```

Annotations:

- CUTLASS 把 GEMM 拆成 CTA tile、warp tile、MMA instruction tile 三层。
- 你能控制 tile shape、数据类型、layout、pipeline、epilogue，所以性能上限高。
- 代价是复杂度高，C++ template 很重。
- MoE 的 GroupedGEMM、FP8/INT4 GEMM、fused epilogue 常用 CUTLASS 实现。

**5. GroupedGEMM: MoE 4 个 Expert 打包**
```python
# Step 1: Router 给每个 token 选择 expert
expert_ids = router(hidden_states)       # e.g. [0, 3, 3, 2, 0, ...]

# Step 2: 按 expert 对 token 分组
tokens_by_expert = {
    0: [token indices...],               # M_0 tokens
    1: [],                               # M_1 = 0，通常跳过
    2: [token indices...],               # M_2 tokens
    3: [token indices...],               # M_3 tokens
}

# Step 3: 构造 GroupedGEMM problem descriptors
groups = []
for e in [0, 1, 2, 3]:
    M_e = len(tokens_by_expert[e])
    if M_e == 0:
        continue

    groups.append({
        "A": packed_hidden_for_expert[e], # [M_e, d_model]
        "B": expert_weight[e],            # [d_model, d_ff]
        "C": expert_output[e],            # [M_e, d_ff]
        "M": M_e,
        "N": d_ff,
        "K": d_model,
        "expert_id": e,
    })

# Step 4: 一次 kernel launch 处理所有非空 expert GEMM
grouped_gemm(groups)
```

Annotations:

- `groups` 里放的是指针、shape、stride、expert id，不是把权重矩阵复制到一起。
- Expert 1 如果没有 token，可以不生成 GEMM problem。
- `packed_hidden_for_expert[e]` 通常来自 token permutation / dispatch，把同一 expert 的 token 放连续。
- 这一步减少的是 kernel launch 开销和小 GEMM 调度碎片。

**6. GroupedGEMM Kernel 内部调度**
```python
@kernel
def grouped_gemm_kernel(groups, tile_prefix_sum):
    global_tile_id = get_global_tile_id()          # 当前 CTA 要处理的全局 tile 编号

    # 根据 prefix sum 找到这个 tile 属于哪个 expert/problem
    problem_id = find_problem(tile_prefix_sum, global_tile_id)
    desc = groups[problem_id]

    e = desc.expert_id
    M_e, N_e, K_e = desc.M, desc.N, desc.K

    # 当前 CTA 在该 expert GEMM 内部负责哪个 C tile
    local_tile_id = global_tile_id - tile_prefix_sum[problem_id]
    tile_m, tile_n = map_tile_id_to_mn(local_tile_id, M_e, N_e, BM, BN)

    acc = zeros([BM, BN])

    for k0 in range(0, K_e, BK):
        # 注意：这里只加载当前 expert 的一个权重 tile
        A_tile = load(desc.A[tile_m*BM : tile_m*BM+BM, k0 : k0+BK])
        B_tile = load(desc.B[k0 : k0+BK, tile_n*BN : tile_n*BN+BN])

        acc += mma(A_tile, B_tile)

    store(desc.C[tile_m*BM : tile_m*BM+BM,
                 tile_n*BN : tile_n*BN+BN], acc)
```

Annotations:

- 一个 CTA 只处理某个 expert 的一个输出 tile。
- 它只加载 `B = W_e` 的一个 `[BK, BN]` 小块，不会加载该 expert 的完整权重。
- 多个 CTA 可以同时处理不同 expert，所以 L2/HBM 层面可能同时有多个 expert 的权重 tile 在流动。
- shared memory/register 层面只放当前 CTA 正在算的那个 expert 的小 tile。
- 所以答案仍然是：GroupedGEMM 后 **不需要 4 个 expert 完整权重同时加载**；只是一次 launch 里调度多个 expert 的 GEMM tile。

### 007 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- SKILL.md: .codex/skills/obsidian-keyword-explainer/SKILL.md
- Q1.4_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.4_L1_answer.md
- Q1.2_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.2_L1_answer.md
- review_summary_skill.md: draft/review_summary_skill.md

## My request for Codex:
每行都需要注释,函数调用的目的,等等,尽量详细

### 008 Codex Final
下面是“逐行注释版”。注意：有些是教学伪代码，函数名贴近真实 Triton / TileLang / CUTLASS 思维，但不是每段都能直接复制运行。

**1. Triton: Fused Bias + GELU**
```python
grid = (ceil_div(N, BLOCK),)                       # Host 端：计算需要多少个 Triton program；每个 program 处理 BLOCK 个元素

bias_gelu_kernel[grid](X, B, Y, N, BLOCK)          # Host 端：启动 kernel；把 X、bias B、输出 Y、长度 N、块大小 BLOCK 传给 GPU

@triton.jit                                        # 告诉 Triton：下面这个 Python 函数要 JIT 编译成 GPU kernel
def bias_gelu_kernel(X, B, Y, N, BLOCK: tl.constexpr): # 定义 kernel；BLOCK 是编译期常量，便于编译器展开和优化
    pid = tl.program_id(0)                         # 获取当前 program 在第 0 个 grid 维度上的编号，类似 CUDA blockIdx.x

    offs = pid * BLOCK + tl.arange(0, BLOCK)       # 生成当前 program 负责的全局元素下标，例如 [0..BLOCK-1] 或 [BLOCK..2BLOCK-1]

    mask = offs < N                                # 判断这些下标是否越界；最后一个 block 可能不满 BLOCK 个元素

    x = tl.load(X + offs, mask=mask, other=0.0)    # 从 HBM/global memory 读取 X[offs]；越界位置用 0.0 填充

    b = tl.load(B + offs, mask=mask, other=0.0)    # 从 HBM/global memory 读取 B[offs]；B 是 bias 或逐元素偏置

    z = x + b                                      # 在寄存器中执行逐元素加法；不会额外写回 HBM

    t = z + 0.044715 * z * z * z                   # GELU tanh 近似公式的一部分；仍在寄存器中计算

    u = 0.79788456 * t                             # 0.79788456 约等于 sqrt(2/pi)，用于 GELU 近似

    y = 0.5 * z * (1.0 + tl.tanh(u))               # 计算 GELU(z)；tl.tanh 是 Triton 的向量化 tanh

    tl.store(Y + offs, y, mask=mask)               # 把结果写回 Y[offs]；mask 防止越界写
```

Annotations:

- 这个 kernel 把 `bias add` 和 `GELU` 融合成一次 HBM 读取、一次 HBM 写回。
- `tl.arange` 生成的是 tile 内向量下标，所以 `x/b/y` 都是向量，不是单个标量。
- `tl.load` 的目的：从 global memory 把一段连续数据搬到寄存器。
- `tl.store` 的目的：把寄存器里的结果写回 global memory。
- 这个例子没有 shared memory，因为逐元素计算不需要跨元素复用。

**2. Triton: Two-Pass Online Softmax**
```python
@triton.jit                                             # 把函数编译为 GPU kernel
def softmax_kernel(X, Y, M, N, stride_xm, stride_ym, BM: tl.constexpr, BN: tl.constexpr): # X/Y 是矩阵，M 行 N 列，BM/BN 是 tile 大小
    pid_m = tl.program_id(0)                            # 当前 program 负责哪一个 row block

    row_ids = pid_m * BM + tl.arange(0, BM)             # 当前 program 负责的 BM 个行号，形状 [BM]

    col_offsets = tl.arange(0, BN)                      # 当前列 tile 内的 BN 个列偏移，形状 [BN]

    m = tl.full((BM,), -float("inf"), tl.float32)        # 每一行的 running max，初始为 -inf

    d = tl.zeros((BM,), tl.float32)                     # 每一行的 running denominator，初始为 0

    for start_n in range(0, N, BN):                     # 第一遍：沿列方向逐 tile 扫描整行

        cols = start_n + col_offsets                    # 当前 tile 的真实列号，形状 [BN]

        mask = (row_ids[:, None] < M) & (cols[None, :] < N) # 生成二维 mask，防止行或列越界

        ptrs = X + row_ids[:, None] * stride_xm + cols[None, :] # 计算 X[row, col] 的地址，形状 [BM, BN]

        x = tl.load(ptrs, mask=mask, other=-float("inf")) # 读取当前 [BM, BN] tile；越界位置设为 -inf，不影响 max

        block_m = tl.max(x, axis=1)                     # 对每一行求当前 tile 的最大值，结果形状 [BM]

        m_new = tl.maximum(m, block_m)                  # 更新每一行的 running max

        old_scale = tl.exp(m - m_new)                   # 如果 max 变大，旧 denominator 要按 exp(m_old - m_new) 缩放

        new_sum = tl.sum(tl.exp(x - m_new[:, None]), axis=1) # 计算当前 tile 对 denominator 的贡献

        d = d * old_scale + new_sum                     # 合并旧 denominator 和当前 tile denominator

        m = m_new                                       # 保存新的 running max，进入下一个列 tile

    for start_n in range(0, N, BN):                     # 第二遍：再次沿列扫描，真正写出 softmax 结果

        cols = start_n + col_offsets                    # 当前输出 tile 的列号

        mask = (row_ids[:, None] < M) & (cols[None, :] < N) # 生成二维 mask，防止越界读写

        ptrs_x = X + row_ids[:, None] * stride_xm + cols[None, :] # 计算输入地址

        ptrs_y = Y + row_ids[:, None] * stride_ym + cols[None, :] # 计算输出地址

        x = tl.load(ptrs_x, mask=mask, other=-float("inf")) # 再次读取当前 tile

        y = tl.exp(x - m[:, None]) / d[:, None]         # 用第一遍得到的全局 m 和 d 计算 softmax

        tl.store(ptrs_y, y, mask=mask)                  # 写回当前 tile 的 softmax 结果
```

Annotations:

- 第一遍只计算每行的 `m=max(x)` 和 `d=sum(exp(x-m))`。
- 第二遍才写出完整 `Y`，因为 softmax 每个元素都依赖整行 denominator。
- `m_new[:, None]` 把 `[BM]` 扩成 `[BM,1]`，用于和 `[BM,BN]` 的 `x` 广播相减。
- 你原文里的伪代码只在循环结束后 store，所以只会写最后一个列 tile；这个 two-pass 版本才完整。
- FlashAttention 会把第二遍换成 `p @ V` 累加，所以不需要显式保存 softmax 矩阵。

**3. TileLang: Tiled GEMM**
```python
@tilelang.jit                                           # 告诉 TileLang：下面是 tile-level kernel，要编译到目标后端
def tiled_gemm(A, B, C, M, N, K):                       # 计算 C[M,N] = A[M,K] x B[K,N]

    for bm, bn in T.grid(ceil_div(M, BM), ceil_div(N, BN)): # 遍历输出矩阵 C 的二维 tile grid

        acc = T.alloc_fragment([BM, BN], dtype="float32") # 在寄存器/fragment 中分配 accumulator tile

        T.clear(acc)                                     # 把 accumulator 清零，准备累加 K 维贡献

        for bk in T.Pipelined(range(0, K, BK), num_stages=3): # 沿 K 维循环；num_stages=3 表示尝试三阶段流水

            As = T.alloc_shared([BM, BK], dtype="float16") # 在 shared memory 分配 A 的当前 tile

            Bs = T.alloc_shared([BK, BN], dtype="float16") # 在 shared memory 分配 B 的当前 tile

            T.copy(A[bm*BM : bm*BM+BM, bk : bk+BK], As) # 把 A 的一个 tile 从 HBM 搬到 shared memory

            T.copy(B[bk : bk+BK, bn*BN : bn*BN+BN], Bs) # 把 B 的一个 tile 从 HBM 搬到 shared memory

            T.gemm(As, Bs, acc)                         # 用 Tensor Core/Matrix Core 做 tile GEMM，并累加到 acc

        T.copy(acc, C[bm*BM : bm*BM+BM, bn*BN : bn*BN+BN]) # 把 accumulator 写回 C 的对应输出 tile
```

Annotations:

- `T.grid` 的目的：定义输出 tile 的二维遍历空间。
- `T.alloc_fragment` 的目的：给 MMA accumulator 分配寄存器级 fragment。
- `T.alloc_shared` 的目的：给当前 tile 分配 shared memory 缓冲区。
- `T.Pipelined` 的目的：让编译器安排 `load next tile` 和 `compute current tile` 的重叠。
- TileLang 的关键优势是：你描述 tile 数据流，layout inference / pipeline inference 尽量自动处理底层细节。

**4. CUTLASS: GEMM 主循环心智模型**
```cpp
ElementA = half;                                      // A 矩阵元素类型，通常 FP16/BF16/FP8/INT8 等

ElementB = half;                                      // B 矩阵元素类型

ElementC = half;                                      // C 矩阵输出元素类型

Accumulator = float;                                  // 累加器类型，FP16 GEMM 通常用 FP32 累加

ThreadblockShape = Shape<128, 128, 64>;               // 一个 CTA 负责 C 的 128x128 tile，每次处理 K 的 64

WarpShape = Shape<64, 64, 64>;                        // 一个 warp/warpgroup 负责的子 tile 形状

InstructionShape = Shape<16, 8, 16>;                  // 单条 Tensor Core MMA 指令处理的矩阵片段形状

for each CTA_tile_C in grid:                          // 每个 CTA 领取一个输出 C tile

    acc = zeros([128, 128]);                           // 为该 CTA 的输出 tile 初始化 accumulator

    for k0 in range(0, K, 64):                         // 沿 K 维分块循环，逐块累加 A[:,k] x B[k,:]

        async_load_A_to_smem(A, CTA_tile_C, k0);        // 异步把 A 的当前 tile 从 HBM 搬到 shared memory

        async_load_B_to_smem(B, CTA_tile_C, k0);        // 异步把 B 的当前 tile 从 HBM 搬到 shared memory

        wait_until_smem_ready();                       // 等待异步加载完成，保证后续 MMA 能读到正确数据

        for each warp_tile inside CTA:                 // CTA 内部再分给多个 warp/warpgroup

            load_fragments_from_smem();                // 每个线程/warp 从 shared memory 取自己的 MMA fragment

            mma(acc_fragment, A_fragment, B_fragment); // 调用 Tensor Core MMA 指令，更新 accumulator fragment

    epilogue(acc);                                     // 对 accumulator 做输出后处理，如 bias、GELU、类型转换

    store_C_tile_to_hbm(C, acc);                       // 把最终 C tile 写回 HBM
```

Annotations:

- CUTLASS 的核心是把 GEMM 映射到三层：CTA tile、warp tile、MMA instruction tile。
- `async_load_*` 的目的：隐藏 HBM 到 shared memory 的加载延迟。
- `wait_until_smem_ready` 的目的：保证计算不会读到未完成的数据。
- `epilogue` 的目的：把 GEMM 后处理融合进同一个 kernel，减少额外 kernel launch。
- 真实 CUTLASS 是大量 C++ template，这里是心智模型伪代码。

**5. GroupedGEMM: Host 端打包 4 个 Expert**
```python
expert_ids = router(hidden_states)                    # Router 根据 token hidden state 选择 expert，例如 top-k expert

tokens_by_expert = group_tokens(expert_ids)            # 把 token 按 expert id 重新分组，便于每个 expert 做连续 GEMM

groups = []                                            # 初始化 GroupedGEMM 的 problem descriptor 列表

for e in range(4):                                     # 假设这里只有 4 个 expert：0、1、2、3

    token_ids = tokens_by_expert[e]                    # 取出分给 expert e 的 token 下标

    M_e = len(token_ids)                               # M_e 是 expert e 收到的 token 数，也就是该 GEMM 的 M 维度

    if M_e == 0:                                       # 如果该 expert 没收到 token

        continue                                       # 跳过这个 expert，不生成空 GEMM problem

    A_e = pack_hidden(hidden_states, token_ids)        # 把 expert e 的 token hidden state gather 成连续矩阵 A_e[M_e,d_model]

    B_e = expert_weight[e]                             # 取 expert e 的权重矩阵 W_e[d_model,d_ff]，只是取指针，不复制完整权重

    C_e = allocate_output(M_e, d_ff)                   # 为 expert e 的输出分配连续空间 C_e[M_e,d_ff]

    groups.append((A_e, B_e, C_e, M_e, d_ff, d_model, e)) # 把指针、shape、expert id 加入 problem descriptor

grouped_gemm(groups)                                   # 一次 kernel launch 处理所有非空 expert 的 GEMM
```

Annotations:

- `groups` 打包的是 descriptor：指针、shape、stride、expert id。
- 它不是把 4 个 expert 的完整权重矩阵复制成一个大矩阵。
- `B_e = expert_weight[e]` 通常只是拿到该 expert 权重在 HBM 中的地址。
- Expert token 数不同，所以每个 group 的 `M_e` 可以不同。
- 这解决的是“小 GEMM 多、launch 多、SM 利用率差”的问题。

**6. GroupedGEMM: Kernel 内部如何加载权重**
```python
@kernel                                                   # 表示下面是 GPU kernel 内部逻辑

def grouped_gemm_kernel(groups, tile_prefix_sum):         # groups 是 problem descriptors，tile_prefix_sum 用来定位 tile 属于哪个 expert

    global_tile_id = get_global_tile_id()                 # 当前 CTA 的全局 tile 编号

    problem_id = find_problem(tile_prefix_sum, global_tile_id) # 根据 prefix sum 判断当前 tile 属于哪个 expert/GEMM

    desc = groups[problem_id]                             # 取出当前 expert 的 descriptor，包括 A/B/C 指针和 shape

    e = desc.expert_id                                    # 当前 CTA 正在处理的 expert id

    M_e = desc.M                                          # 当前 expert 的 token 数

    N_e = desc.N                                          # 当前 expert 输出维度，通常是 d_ff

    K_e = desc.K                                          # 当前 expert 输入维度，通常是 d_model

    local_tile_id = global_tile_id - tile_prefix_sum[problem_id] # 把全局 tile id 转成当前 expert 内部的 tile id

    tile_m, tile_n = map_tile_id_to_mn(local_tile_id, M_e, N_e, BM, BN) # 计算当前 CTA 负责 C_e 的哪个 [BM,BN] tile

    acc = zeros([BM, BN])                                 # 初始化当前输出 tile 的 accumulator

    for k0 in range(0, K_e, BK):                          # 沿 K 维分块累加

        A_tile = load(desc.A[tile_m*BM : tile_m*BM+BM, k0 : k0+BK]) # 加载当前 expert 的 token activation tile

        B_tile = load(desc.B[k0 : k0+BK, tile_n*BN : tile_n*BN+BN]) # 加载当前 expert 的权重 tile，不是完整权重矩阵

        acc += mma(A_tile, B_tile)                        # 用 Tensor Core 对当前 tile 做矩阵乘并累加

    store(desc.C[tile_m*BM : tile_m*BM+BM, tile_n*BN : tile_n*BN+BN], acc) # 写回当前 expert 的输出 tile
```

Annotations:

- 这个代码回答你的关键问题：GroupedGEMM 后，**一个 CTA 只加载一个 expert 的一个权重 tile**。
- 多个 CTA 可以同时处理不同 expert，所以 HBM/L2 层面可能同时流入多个 expert 的权重切片。
- shared memory/register 层面不会同时放 4 个完整 expert 权重矩阵。
- “Grouped” 的含义是：多个 GEMM problem 共享一次 kernel launch 和一套 tile scheduler。
- 真正的数据移动仍然是 `A_tile/B_tile` 级别，按 `K` 循环逐块加载。

### 009 User
# Context from my IDE setup:

## Active file: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- SKILL.md: .codex/skills/obsidian-keyword-explainer/SKILL.md
- Q1.4_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.4_L1_answer.md
- Q1.2_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.2_L1_answer.md
- review_summary_skill.md: draft/review_summary_skill.md

## My request for Codex:
增加每一个变量的含义和作用

### 010 Codex Final
可以。下面是上一版伪代码的“变量字典”。形状是教学抽象，真实代码里还会有 layout、stride、alignment、dtype 细节。

**0. 通用变量**
| 变量 | 含义 | 作用 |
|---|---|---|
| `M` | 矩阵行数 | 通常表示 token 数、batch 行数、输出行数 |
| `N` | 矩阵列数 | softmax 中表示每行长度；GEMM 中表示输出列数 |
| `K` | GEMM 规约维度 | `A[M,K] x B[K,N]` 中间相乘的维度 |
| `BM` | row tile 大小 | 一个 program/CTA 一次处理多少行 |
| `BN` | column tile 大小 | 一个 program/CTA 一次处理多少列 |
| `BK` | K tile 大小 | GEMM 沿 K 维每次加载多少列/行 |
| `CTA` | thread block | GPU 上一个协作执行单位 |
| `HBM` | global memory | 显存，容量大但慢 |
| `SMEM` | shared memory | SM 内共享内存，容量小但快 |
| `REG` | registers | 线程私有寄存器，最快但最少 |

**1. Triton Bias + GELU**
| 变量 | 含义 | 作用 |
|---|---|---|
| `grid` | kernel launch 的 program 网格 | 决定启动多少个 Triton program |
| `X` | 输入向量/张量 | 被加 bias 并做 GELU |
| `B` | bias 向量 | 与 `X` 逐元素相加 |
| `Y` | 输出向量/张量 | 保存 GELU 结果 |
| `N` | 元素总数 | 判断边界，避免越界 |
| `BLOCK` | 每个 program 处理的元素数 | 控制 tile 粒度 |
| `pid` | 当前 program id | 决定当前 program 负责哪段数据 |
| `offs` | 当前 program 的全局元素下标 | 用于访问 `X/B/Y` |
| `mask` | 越界保护布尔向量 | 防止最后一个 block 读写越界 |
| `x` | 从 `X` 读出的 tile | 存在寄存器中 |
| `b` | 从 `B` 读出的 bias tile | 存在寄存器中 |
| `z` | `x + b` | bias add 后的中间值 |
| `t` | GELU 近似公式中间项 | 计算 `z + 0.044715 z^3` |
| `u` | tanh 输入 | 计算 `sqrt(2/pi) * t` |
| `y` | GELU 输出 tile | 最后写回 `Y` |

**2. Triton Online Softmax**
| 变量 | 含义 | 作用 |
|---|---|---|
| `X` | 输入矩阵 `[M,N]` | 每行做 softmax |
| `Y` | 输出矩阵 `[M,N]` | 保存 softmax 结果 |
| `stride_xm` | `X` 的行 stride | 用 `row * stride_xm + col` 算地址 |
| `stride_ym` | `Y` 的行 stride | 用 `row * stride_ym + col` 算输出地址 |
| `pid_m` | 当前 row-block id | 决定当前 program 处理哪几行 |
| `row_ids` | 当前 tile 的行号 `[BM]` | 访问多行 |
| `col_offsets` | tile 内列偏移 `[BN]` | 生成当前列 tile |
| `start_n` | 当前列 tile 起点 | 沿 N 维循环扫描 |
| `cols` | 当前 tile 的列号 `[BN]` | 和 `row_ids` 组合成二维地址 |
| `mask` | `[BM,BN]` 越界 mask | 防止读写超过 `M,N` |
| `ptrs` | 输入地址矩阵 | 指向 `X[row, col]` |
| `ptrs_x` | 输入地址矩阵 | 第二遍读取 `X` |
| `ptrs_y` | 输出地址矩阵 | 第二遍写入 `Y` |
| `x` | 当前 `[BM,BN]` 输入 tile | 当前列块的数据 |
| `m` | running max `[BM]` | 每行当前见过的最大值 |
| `d` | running denominator `[BM]` | 每行当前 softmax 分母 |
| `block_m` | 当前 tile 每行最大值 | 用于更新 `m` |
| `m_new` | 更新后的 running max | 新旧 max 的较大者 |
| `old_scale` | `exp(m_old - m_new)` | 把旧分母缩放到新 max 坐标系 |
| `new_sum` | 当前 tile 的 exp sum | 当前 tile 对分母的贡献 |
| `y` | 当前 tile softmax 结果 | `exp(x-m)/d` |

核心关系：

```text
m_new = max(m, block_m)
d_new = d_old * exp(m_old - m_new) + sum(exp(x - m_new))
```

**3. TileLang Tiled GEMM**
| 变量 | 含义 | 作用 |
|---|---|---|
| `A` | 左矩阵 `[M,K]` | GEMM 输入 |
| `B` | 右矩阵 `[K,N]` | GEMM 权重/输入 |
| `C` | 输出矩阵 `[M,N]` | 保存 `A @ B` |
| `bm` | 输出 tile 的行块 id | 定位 `C` 的行 tile |
| `bn` | 输出 tile 的列块 id | 定位 `C` 的列 tile |
| `bk` | K 维 tile 起点 | 沿 K 维分块累加 |
| `acc` | accumulator `[BM,BN]` | 保存当前输出 tile 的累加结果 |
| `As` | A 的 shared-memory tile `[BM,BK]` | 从 HBM 搬到 SMEM 以复用 |
| `Bs` | B 的 shared-memory tile `[BK,BN]` | 从 HBM 搬到 SMEM 以复用 |
| `num_stages` | pipeline stage 数 | 控制加载和计算重叠程度 |

数据流：

```text
A/B in HBM -> As/Bs in SMEM -> MMA -> acc in REG -> C in HBM
```

**4. CUTLASS GEMM**
| 变量 | 含义 | 作用 |
|---|---|---|
| `ElementA` | A 元素类型 | 例如 FP16/BF16/INT8 |
| `ElementB` | B 元素类型 | 例如 FP16/BF16/INT8 |
| `ElementC` | C 输出类型 | 输出保存精度 |
| `Accumulator` | 累加器类型 | 通常 FP32，保证精度 |
| `ThreadblockShape` | CTA tile 形状 | 一个 CTA 处理多大的输出 tile |
| `WarpShape` | warp tile 形状 | CTA 内每个 warp 处理的子块 |
| `InstructionShape` | MMA 指令形状 | Tensor Core 单次处理的 fragment |
| `CTA_tile_C` | 当前 CTA 的输出 tile | 当前 block 负责的 `C` 区域 |
| `k0` | K 维当前分块起点 | 控制 mainloop 的 K tile |
| `acc` | 累加器 fragment | 保存 MMA 累加结果 |
| `A_fragment` | A 的 MMA 小片段 | 从 SMEM 取到寄存器 |
| `B_fragment` | B 的 MMA 小片段 | 从 SMEM 取到寄存器 |
| `acc_fragment` | 当前线程/warp 的累加片段 | Tensor Core 写入的位置 |

CUTLASS 的重点是把一个 GEMM 拆成：

```text
Grid -> CTA tile -> Warp tile -> MMA instruction tile
```

**5. GroupedGEMM Host 端打包**
| 变量 | 含义 | 作用 |
|---|---|---|
| `hidden_states` | token hidden states | MoE expert 的输入 |
| `expert_ids` | router 输出的 expert id | 决定每个 token 送给哪个 expert |
| `tokens_by_expert` | 按 expert 分组的 token 列表 | 让同一 expert 的 token 连续处理 |
| `groups` | GEMM problem descriptor 列表 | 存每个 expert GEMM 的指针和 shape |
| `e` | expert id | 当前正在构造哪个 expert 的问题 |
| `token_ids` | expert `e` 收到的 token 下标 | 用于 gather hidden states |
| `M_e` | expert `e` 的 token 数 | 当前 GEMM 的 M 维 |
| `d_model` | hidden size | 当前 GEMM 的 K 维 |
| `d_ff` | FFN hidden size | 当前 GEMM 的 N 维 |
| `A_e` | expert `e` 的输入矩阵 `[M_e,d_model]` | token activation |
| `B_e` | expert `e` 的权重 `[d_model,d_ff]` | expert weight 指针 |
| `C_e` | expert `e` 的输出 `[M_e,d_ff]` | GEMM 输出 |
| `expert_weight[e]` | 第 `e` 个 expert 的权重 | 只是取地址，不复制完整矩阵 |

这里的关键点：

```text
groups 打包的是 descriptor，不是完整权重矩阵。
```

**6. GroupedGEMM Kernel 内部**
| 变量 | 含义 | 作用 |
|---|---|---|
| `groups` | problem descriptor 数组 | 每个元素描述一个 expert GEMM |
| `tile_prefix_sum` | 每个 problem 的 tile 前缀和 | 用全局 tile id 找到所属 expert |
| `global_tile_id` | 当前 CTA 的全局 tile 编号 | 表示当前 CTA 在所有 expert tile 中排第几个 |
| `problem_id` | 当前 tile 属于哪个 GEMM problem | 找到对应 expert |
| `desc` | 当前 problem descriptor | 取出 A/B/C 指针和 M/N/K |
| `e` | 当前 expert id | 当前 CTA 处理哪个 expert |
| `M_e` | 当前 expert token 数 | 当前 GEMM 的 M 维 |
| `N_e` | 当前 expert 输出维度 | 通常等于 `d_ff` |
| `K_e` | 当前 expert 输入维度 | 通常等于 `d_model` |
| `local_tile_id` | 当前 expert 内部 tile id | 从全局 tile 编号转成本地编号 |
| `tile_m` | 当前输出 tile 的 M 坐标 | 决定处理哪些 token 行 |
| `tile_n` | 当前输出 tile 的 N 坐标 | 决定处理哪些输出列 |
| `acc` | 当前 CTA 的 `[BM,BN]` 累加器 | 保存输出 tile 的累加结果 |
| `k0` | 当前 K tile 起点 | 沿 K 维循环 |
| `A_tile` | 当前 activation tile `[BM,BK]` | 从 `desc.A` 加载 |
| `B_tile` | 当前 expert 权重 tile `[BK,BN]` | 从 `desc.B` 加载 |
| `desc.C` | 当前 expert 输出地址 | 写回 `acc` |

回答你之前的问题可以精确写成：

```text
GroupedGEMM 一次 launch 处理多个 expert。
但每个 CTA 只处理一个 expert 的一个 C tile。
每个 CTA 每次只加载该 expert 的一个 B_tile = W_e[k0:k0+BK, n:n+BN]。
不会把 4 个 expert 的完整权重同时加载到 shared memory。
```

**函数调用目的速查**
| 调用 | 目的 |
|---|---|
| `tl.program_id(0)` | 获取当前 Triton program 的编号 |
| `tl.arange(a,b)` | 生成 tile 内向量索引 |
| `tl.load(ptrs, mask, other)` | 从 HBM 读取 tile，mask 越界保护 |
| `tl.store(ptrs, val, mask)` | 把 tile 写回 HBM |
| `tl.max(x, axis=1)` | 对 tile 每一行求最大值 |
| `tl.sum(x, axis=1)` | 对 tile 每一行求和 |
| `T.grid(...)` | 定义 TileLang tile 遍历空间 |
| `T.alloc_fragment(...)` | 分配寄存器级 accumulator |
| `T.alloc_shared(...)` | 分配 shared memory tile |
| `T.copy(src,dst)` | 在 HBM/SMEM/REG 间搬运 tile |
| `T.gemm(A,B,acc)` | 调用 Tensor Core/Matrix Core 做 tile GEMM |
| `T.Pipelined(...)` | 让加载和计算形成流水重叠 |
| `grouped_gemm(groups)` | 一次 kernel launch 处理多个 GEMM problem |

### 011 User
<environment_context>
  <current_date>2026-06-03</current_date>
  <timezone>Asia/Shanghai</timezone>
  <filesystem><workspace_roots><root>/data3/paper_analysis</root></workspace_roots><permission_profile type="managed"><file_system type="restricted"><entry access="read"><special>:root</special></entry><entry access="write"><path>/data3/paper_analysis</path></entry><entry access="write"><special>:slash_tmp</special></entry><entry access="write"><special>:tmpdir</special></entry><entry access="read"><path>/data3/paper_analysis/.git</path></entry><entry access="read"><path>/data3/paper_analysis/.agents</path></entry><entry access="read"><path>/data3/paper_analysis/.codex</path></entry></file_system></permission_profile></filesystem>
</environment_context>

### 012 User
# Context from my IDE setup:

## Open tabs:
- Q1.3_L1_answer.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/Q1.3_L1_answer.md
- L1_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md
- SKILL.md: .codex/skills/export-conversation-notes/SKILL.md
- Accelerating Sparse Transformer Inference on GPU (STOF).md: human_notes/Accelerating Sparse Transformer Inference on GPU (STOF).md
- Adaptive Draft Sequence Length_ Enhancing Speculative Decoding Throughput on PIM-Enabled Systems.md: human_notes/Adaptive Draft Sequence Length_ Enhancing Speculative Decoding Throughput on PIM-Enabled Systems.md

## My request for Codex:
[$export-conversation-notes](/data3/paper_analysis/.codex/skills/export-conversation-notes/SKILL.md) 

### 013 User
<skill>
<name>export-conversation-notes</name>
<path>/data3/paper_analysis/.codex/skills/export-conversation-notes/SKILL.md</path>
---
name: export-conversation-notes
description: Export the currently available Codex/chat session record into a target Markdown file, defaulting to human_notes/. Save only user inputs and Codex final outputs. Use when the user asks to save, archive, output, or record the current conversation, dialogue history, or analysis session in the paper_analysis workspace. If the target Markdown file already exists and is non-empty, append incrementally without modifying, deleting, reformatting, or replacing any existing content.
---

# Export Conversation Notes

## Overview

Save the current session's available user inputs and Codex final outputs to the requested target location. Default to `human_notes/` when no explicit directory is provided. Default to Chinese for headings and status text. Do not save process notes, status updates, tool calls, command outputs, hidden instructions, or intermediate work logs.

## Workflow

1. Identify the target path.
   - save under `human_notes/`.
   - If no filename can be identified, use `<session-name>.md` and <session-name> is the name of the current session.

2. Prepare the target file.
   - Work relative to the current workspace root.
   - Ensure the destination directory exists.
   - Sanitize only the filename portion by replacing filesystem-hostile characters such as `/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, and `|` with safe separators, then trim whitespace.
   - Save to the resolved target path.

3. Capture the available conversation record.
   - Output only user inputs and Codex final outputs in chronological order with speaker labels.
   - Include all available user messages exactly as written.
   - Include only Codex final responses that answer the user, not interim progress updates or tool-facing content.
   - Do not include tool calls, shell commands, command outputs, file edit logs, errors from tools, status updates, planning chatter, hidden system/developer/policy/runtime instructions, or intermediate reasoning.
   - Do not summarize, compress, paraphrase, normalize, or reorganize the saved user inputs and final outputs.
   - If earlier user inputs or final outputs are unavailable because context was compacted or not exposed to Codex, state this limitation briefly before the saved conversation.

4. Write the Markdown file.
   - If the file does not exist, create it with the new-file template.
   - If the file exists but is empty, write the new-file template.
   - If the file exists and is non-empty, enter incremental mode: append a new dated section only at the end of the file.
   - In incremental mode, never modify, delete, reorder, summarize, normalize, reformat, or replace any existing content, even if the existing note has typos, duplicate headings, stale metadata, or inconsistent formatting.
   - In incremental mode, use an append-only edit. With `apply_patch`, add only new lines after the existing final line.
   - Keep the saved record readable, but preserve the available user inputs and final outputs over brevity.
   - If the record is too long for one edit, append it in multiple consecutive chunks until all currently available user inputs and final outputs are saved.

## Markdown Template

For a new file, use this structure:

```md
# <session-or-paper-title>

- 导出时间：<YYYY-MM-DD HH:MM TZ>
- 来源：当前 Codex session 可用对话上下文
- 保存路径：<resolved-target-path>
- 范围：仅用户输入与 Codex 最终输出

## 对话记录

### 001 User
<用户消息原文>

### 002 Codex Final
<Codex 最终回复原文>
```

For appending to an existing non-empty file, add this block at the end of the file without changing earlier content:

```md
---

## 对话记录补充：<YYYY-MM-DD HH:MM TZ>

<continue the same chronological format, saving only user inputs and Codex final outputs>
```

## Completion Response

After saving, respond briefly with the output path and whether the file was created or appended. Mention any uncertainty about target inference or incomplete available conversation context.

</skill>
