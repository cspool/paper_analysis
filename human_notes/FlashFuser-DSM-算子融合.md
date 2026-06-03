# FlashFuser DSM 算子融合

- 导出时间：2026-06-03
- 来源：当前 Claude Code session 可用对话上下文
- 保存路径：human_notes/FlashFuser-DSM-算子融合.md
- 范围：仅用户输入与 Claude 最终输出

## 对话记录

### 001 User
（用户选中 L1_horizon_summary.md 第 115 行关于 FlashFuser DSM Fusion 的内容）

教我

### 002 Claude

好的，我来从你选的这条 **FlashFuser DSM Fusion** 出发，系统讲解算子融合这一整个分类。这将是一篇完整的技术教学。

---

# 算子融合（Operator Fusion）：从 HBM 瓶颈到 FlashFuser 跨 SM 核间融合

## 一、先理解问题：为什么需要算子融合？

### 1.1 GPU 存储金字塔的核心矛盾

现代 GPU（以 H100 为例）存储层次的速度差距呈数量级：

```
┌─────────────────────────────────────────────────────────┐
│  H100 存储层次                   容量        带宽       延迟  │
│  ┌──────────┐                                             │
│  │ Register │              256KB/SM     ~100 TB/s   ~0ns │
│  │  SMEM    │              227KB/SM     ~10 TB/s    ~10ns│
│  │  DSM  ★  │          跨SM共享      4-8 TB/s     ~20ns│  ← FlashFuser 的新战场
│  │  L2      │              50MB          ~7 TB/s   ~100ns│
│  │  HBM     │              80GB          3.35 TB/s ~280ns│  ← 传统融合主战场
│  └──────────┘                                             │
└─────────────────────────────────────────────────────────┘
```

**核心矛盾：算力增长 >> 显存带宽增长。**

H100 的 BF16 Tensor Core 算力是 **989 TFLOPS**，但 HBM 带宽只有 **3.35 TB/s**。这意味着：

- 每从 HBM 读一个 FP16 数值（2 bytes），Tensor Core 可以做 989 TFLOPS / (3.35 TB/s / 2B) ≈ **590 次浮点运算**
- 绝大多数 DL 算子实际上是 **memory-bound**（数据搬运时间 >> 计算时间）

这就是 **Roofline 模型**告诉我们的：不减少数据搬运，算力再多也用不上。

### 1.2 传统执行模式：Kernel-by-Kernel 的浪费

以最简单的 `Linear → ReLU → Linear` 为例：

```
传统执行（PyTorch eager）：
  Time ──────────────────────────────────────►
  
  Kernel 1 (Linear1 GEMM):
    HBM→SMEM: 加载 W1, X  ────────
    Tensor Core: A×B           ████
    SMEM→HBM: 写回中间结果 Y   ────
  
  Kernel 2 (ReLU):                     ← 全新 kernel launch (~100μs)
    HBM→SMEM: 重新加载 Y      ────────  ← Y 刚写回 HBM, 又立刻读出来!
    CUDA Core: max(0, y)      ██
    SMEM→HBM: 写回 Y'         ────
  
  Kernel 3 (Linear2 GEMM):              ← 又一个 kernel launch
    HBM→SMEM: 加载 W2, Y'    ────────  ← Y' 又是刚写就读!
    Tensor Core: C×D          ████
    SMEM→HBM: 写回输出        ────

问题:
  ❌ 中间数据 Y, Y' 来回穿越 HBM（HBM access 量 = 输入 + W1 + Y + Y + W2 + Y' + 输出）
  ❌ 3 次 kernel launch overhead（~3×100μs）
  ❌ SMEM 在每个 kernel 结束后全部丢弃
```

### 1.3 算子融合的直观思想

**一句话：让中间数据留在片上存储（SMEM/DSM/L2），别写回 HBM。**

```
算子融合后（Fused Kernel）：
  Time ──────────────────────────────────────►
  
  Single Fused Kernel:
    HBM→SMEM: 加载 W1, X  ────────
    Tensor Core: GEMM1      ████████
    ┌─ SMEM 中的 Y 直接传给 ReLU ─┐  ← 不写 HBM！
    CUDA Core: ReLU         ████
    ┌─ SMEM 中的 Y' 直接传给 GEMM2┐  ← 还是不写 HBM！
    Tensor Core: GEMM2      ████████
    SMEM→HBM: 写回最终输出  ────

收益:
  ✅ HBM access 减少 (消除了 Y 和 Y' 的写+读)
  ✅ 单次 kernel launch
  ✅ 减少总执行时间
```

---

## 二、融合的分级：从寄存器到跨 SM 集群

算子融合并非二元的"能融/不能融"，而是沿着存储层次有明确的**融合等级**：

```
融合等级          存储介质      容量限制      代表方法
─────────────────────────────────────────────────────
L0 融合 (寄存器级)  Register     256 KB/SM     Ansor, TVM element-wise
L1 融合 (SMEM级)    Shared Mem   227 KB/SM     TensorRT, Welder
L2 融合 (DSM级)  ★  DSM          跨 SM 共享    FlashFuser (本文重点)
L3 融合 (L2级)      L2 Cache     50 MB         Kitsune (L2 ring buffer)
L4 融合 (Streaming) 全片上 SRAM   ~520 MB       SN40L 硬件原生融合
```

**关键瓶颈：SMEM 只有 227KB/SM。**

对于 LLM 推理中典型的 GEMM（例如 M=128, N=16384, K=4096），中间结果 C matrix = 128 × 16384 × 2 bytes ≈ **4.2 MB**，远超单 SM 的 227KB。传统方法只能放弃融合，让 C 溢出到 HBM；FlashFuser 的价值在于**让 C 越过 SMEM 但留在 DSM 中**，从而突破单 SM 的融合规模上限。

---

## 三、FlashFuser 核心机制详解

### 3.1 什么是 DSM（Distributed Shared Memory）？

Hopper 架构（H100）引入了 **SM Cluster** 概念：多个 SM 组成一个 cluster，cluster 内的 SM 可以通过 **DSM** 直接访问彼此的 shared memory，无需经过 L2 或 HBM。

```
┌─────────── SM Cluster (可达 16 SM) ───────────┐
│                                                │
│  ┌─ SM0 ────┐  ┌─ SM1 ────┐      ┌─ SM15 ───┐│
│  │ SMEM     │  │ SMEM     │      │ SMEM      ││
│  │ 227KB    │  │ 227KB    │ ...  │ 227KB     ││
│  └──┬───────┘  └──┬───────┘      └──┬────────┘│
│     │             │                 │          │
│     └─────────────┼─────────────────┘          │
│                   │ DSM Interconnect           │
│              BW: ~4-8 TB/s                     │
│              Latency: ~20ns (vs HBM ~280ns)    │
└────────────────────────────────────────────────┘
```

**DSM 的关键参数：**
- Cluster=2 SM: ~8 TB/s 带宽，~20ns 延迟
- Cluster=16 SM: ~4 TB/s 带宽（仍高于 HBM 的 3.35 TB/s）
- 对比：HBM 延迟 ~280ns，DSM 快 **14 倍**

### 3.2 FlashFuser 编译流水线

```
┌─────────────────────────────────────────────────────┐
│               FlashFuser 编译期搜索流程               │
├─────────────────────────────────────────────────────┤
│                                                      │
│  输入: 算子链 (如 GEMM0→GatedAct→GEMM1)             │
│        Problem Size (M,N,K,L) + H100 Memory Hierarchy│
│                                                      │
│  Step 1: 统一 Loop 维度建模                          │
│    X = {x_0, x_1, ..., x_{J-1}}  // 算子链的共依赖   │
│    每个 x_j 选: Spatial(多SM并行) / Temporal(单SM串行)│
│    组合数: 最多 41 种                                 │
│                                                      │
│  Step 2: Hierarchical Tiling                        │
│    Cluster-level tile → 影响跨 Block 数据交换        │
│    Block-level tile    → 影响 reg vs SMEM 分配       │
│                                                      │
│  Step 3: Resource Mapping (贪心)                     │
│    reusable tensor 逐级放置: reg → SMEM → DSM → HBM  │
│    超出当前 level 容量 → spill 到下一级              │
│                                                      │
│  Step 4: Cost Model + Pruning                       │
│    Cost C = max(V_l / B_l)  // 最慢 level 决定延迟   │
│    V_l = data movement volume at level l             │
│    B_l = bandwidth of level l                       │
│    5 条 pruning 规则 → ~1.15×10^6 候选               │
│                                                      │
│  Step 5: Top-11 Profiling + 选择                    │
│    从 ~10^6 候选 → Top-11 (cost model) → H100 实测  │
│    → 选最优配置                                      │
│                                                      │
│  输出: Fused CUDA Kernel (CUTLASS + DSM primitives) │
└─────────────────────────────────────────────────────┘
```

### 3.3 三个 DSM 通信原语

FlashFuser 定义了三个构建在 TMA + mbarrier 上的 DSM 通信原语：

```
原语 1: dsm_all_exchange
┌─────────────────────────────────────┐
│ 用途: Cluster 内 AllReduce 或 Mul   │
│                                     │
│ 场景: GEMM0 的输出 C 被 K 维度     │
│ spatial partition → 每个 Block 只有│
│ partial C → 需要 cluster 内规约    │
│                                     │
│ 实现: TMA + mbarrier               │
│ - TMA 异步拷贝 partial tiles       │
│ - mbarrier 同步所有参与 Block      │
│ - 结果: 每个 Block 获得完整 C tile │
│                                     │
│ Standard FFN: AllReduce (加法)      │
│ Gated FFN:    Mul (SiLU gate × up) │
└─────────────────────────────────────┘

原语 2: dsm_shuffle
┌─────────────────────────────────────┐
│ 用途: Shuffle Group 内 ring        │
│       communication 交换 C tile    │
│                                     │
│ 场景: GEMM1 的 consumer 需要不同   │
│ 的 C tile slice → DSM 中交换而不是 │
│ 走 HBM                             │
│                                     │
│ 实现: 在 Shuffle Group 内旋转      │
│ C tile slices（类似环形 all-to-all）│
│                                     │
│ Group size = cls_l / cls_k          │
│ (沿 L 维度的 spatial partition     │
│  除以 K 维度的 partition)          │
└─────────────────────────────────────┘

原语 3: dsm_reduce_scatter
┌─────────────────────────────────────┐
│ 用途: 两级 hierarchical reduction  │
│                                     │
│ Phase 1: Intra-cluster reduce      │
│  多个 Shuffle Group 的 partial E   │
│  在 cluster 内归约                 │
│                                     │
│ Phase 2: Inter-cluster reduce      │
│  TMA cp.reduce.async.bulk 跨       │
│  cluster 原子归约                   │
│                                     │
│ Scatter pattern: 每个 Block 仅     │
│ 负责写回一部分 output tile          │
└─────────────────────────────────────┘
```

### 3.4 完整执行流程（以 Standard FFN G5 为例）

```
配置: M=128, N=16384, K=4096, L=4096, H100 SXM
Cluster 配置: cls_m=2, cls_n=4, cls_k=2, cls_l=4 → 64 Blocks/cluster

Timeline ─────────────────────────────────────────────────►

╔════════════════════════════════════════════════════════╗
║ PHASE 1: GEMM0 (C = A × B)                            ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║  Step 1a: Load                                       ║
║    HBM ──TMA──► SMEM: 加载 A_tile [blk_m × blk_k]   ║
║    HBM ──TMA──► SMEM: 加载 B_tile [blk_k × blk_n]   ║
║                                                        ║
║  Step 1b: Compute                                     ║
║    Tensor Core WGMMA: C_partial = A_tile × B_tile    ║
║    (FP16/BF16 → FP32 accumulate)                      ║
║    cls_k=2 → 2 个 Block 各自计算 K 维度的一半         ║
║                                                        ║
║  Step 1c: DSM AllExchange  ← 核心！                   ║
║    ┌─────┐    ┌─────┐                                 ║
║    │ C_0 │◄──►│ C_1 │  DSM AllReduce (cluster内)     ║
║    └─────┘    └─────┘                                 ║
║    结果: 两个 Block 都获得完整 C (200+KB, 存DSM)     ║
║    NOTE: C 驻留 DSM，不写 HBM！                       ║
╚════════════════════════════════════════════════════════╝
                        │
                        │ DSM 中的数据直接传给下一阶段
                        ▼
╔════════════════════════════════════════════════════════╗
║ PHASE 2: GEMM1 (E = C × D)                            ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║  Step 2a: DSM Shuffle                                 ║
║    ┌───┐    ┌───┐    ┌───┐    ┌───┐                  ║
║    │C_0│───►│C_1│───►│C_2│───►│C_3│  Ring exchange   ║
║    └───┘    └───┘    └───┘    └───┘                  ║
║    每个 Block 获得其 GEMM1 计算所需的 C_tile slice    ║
║                                                        ║
║  Step 2b: Compute                                     ║
║    HBM ──TMA──► SMEM: 加载 D_tile                    ║
║    Tensor Core WGMMA: E_partial = C_tile_slice × D   ║
╚════════════════════════════════════════════════════════╝
                        │
                        ▼
╔════════════════════════════════════════════════════════╗
║ PHASE 3: Store (Reduce Scatter)                       ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║  Step 3a: Intra-cluster Reduce                        ║
║    多个 Shuffle Group 的 partial E → cluster 内规约   ║
║                                                        ║
║  Step 3b: Inter-cluster Reduce                        ║
║    TMA cp.reduce.async.bulk → 跨 cluster 原子累加     ║
║                                                        ║
║  Step 3c: Write-back                                  ║
║    SMEM ────► HBM: 最终输出 E                        ║
║    每个 Block 仅写回自己负责的 output tile 部分        ║
╚════════════════════════════════════════════════════════╝
```

### 3.5 Gated FFN 的两种融合策略

FlashFuser 对 MoE/LLM 中常见的 Gated FFN（SiLU Gate × Up → Down projection）提供两种策略：

```
策略 A: Spatial Partitioning (cls_k=2)
┌──────────────────────────────────────────┐
│ Block Group 0        Block Group 1       │
│ ┌──────────────┐    ┌──────────────┐     │
│ │ GEMM_gate     │    │ GEMM_up       │     │
│ │ (SiLU branch) │    │ (Up branch)   │     │
│ └──────┬───────┘    └──────┬───────┘     │
│        │   DSM Mul          │             │
│        └───────┬────────────┘             │
│                ▼                          │
│        ┌──────────────┐                   │
│        │ GEMM_down    │                   │
│        └──────────────┘                   │
│ 优点: 最大化并行度                      │
│ 代价: 更多 DSM 通信                     │
└──────────────────────────────────────────┘

策略 B: Sequential within Block
┌──────────────────────────────────────────┐
│ 同一 Block 内串行:                       │
│ GEMM_gate → SiLU → DSM Mul → GEMM_down  │
│                                          │
│ 优点: 最小 DSM 通信                     │
│ 代价: 较少的并行度                      │
│                                          │
│ FlashFuser 根据 problem size 自动选择   │
└──────────────────────────────────────────┘
```

---

## 四、与其他融合方法的横向对比

### 4.1 FlashFuser vs Welder（SMEM 级 vs DSM 级融合）

| 维度 | Welder | FlashFuser |
|------|--------|------------|
| **融合介质** | SMEM (227KB/SM) | DSM (跨 SM，有效容量 >227KB) |
| **核心抽象** | Tile-Graph + Tile Propagation | Loop Schedule + DSM Primitives |
| **搜索方法** | 双层（Graph Connecting + Sub-Graph Tiling） | Cost Model pruning → Top-11 profiling |
| **融合规模** | 48-op chain（但受 SMEM 限制） | 跨 SM cluster 的大 GEMM chain |
| **中间数据** | SMEM 内驻留 | DSM 内驻留（250KB~） |
| **关键约束** | 中间 tensor 必须 ≤ SMEM 容量 | 中间 tensor 可超 SMEM 但需 ≤ DSM 有效容量 |

**互补关系**：Welder 处理小 tensor 的多算子链融合（如 NeRF MLP 全融合，DRAM traffic 节省 69%）；FlashFuser 处理大 tensor 的 GEMM 链融合（如 LLM FFN 的 4MB 中间结果，HBM access 减少 58%）。

### 4.2 FlashFuser vs Kitsune（DSM 融合 vs L2 Dataflow）

| 维度 | FlashFuser | Kitsune |
|------|-----------|---------|
| **通信介质** | DSM (SM→SM direct) | L2-resident ring buffer queue |
| **执行模型** | 单个 mega-kernel（时间复用+空间分区） | 多 CTA 空间并发（不同算子不同 CTAs） |
| **同步机制** | mbarrier (many-to-many) | Atomic queue enqueue/dequeue |
| **硬件要求** | H100 Hopper (DSM only) | 需双 arbiter grid scheduler（当前无 GPU 支持） |
| **成熟度** | 已实现在 CUDA 12.4 + CUTLASS 上 | 学术原型，需 GPU HW 修改 |

### 4.3 FlashFuser vs SN40L Streaming Dataflow（编译融合 vs 硬件原生融合）

SambaNova SN40L 走的是完全不同的路线——**硬件原生 streaming**。1040 个 PCU（可配置 SA/SIMD）+ 1040 个 PMU 组成一个空间 pipeline，算子结果在 PMU scratchpad 中流式传递，**中间结果永不物化到 off-chip**。

FlashFuser 在 GPU 上用软件+DSM 硬件模拟了类似的效果。区别在于：
- SN40L：编译器 PnR 自动映射，硬件保证 streaming
- FlashFuser：编译器搜索 + Cost Model + DSM primitives，在 SIMT GPU 上模拟

---

## 五、实验结果与关键数据

```
┌─────────────────────────────────────────────────────────┐
│              FlashFuser 性能数据（H100 SXM）              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Micro-benchmark (GEMM/Gated FFN/Conv chains):          │
│    vs PyTorch (cuBLAS):   3.1×                          │
│    vs Chimera (SOTA):     4.1×  ← 核心对比              │
│    vs TensorRT:           >2×                           │
│    HBM Access:            -58%                          │
│                                                          │
│  Ablation:                                               │
│    DA only (SMEM/L2 fusion):     1.52× vs baseline      │
│    DC+DA (DSM, random search):   2.11× vs baseline      │
│    All  (DC+DA+SE, full system): 3.29× vs baseline      │
│    → Search Engine 贡献最大增量 (2.11→3.29 = 1.56×)     │
│                                                          │
│  End-to-End (SGLang, Llama/Qwen 系列):                   │
│    平均 E2E speedup: 1.24×                               │
│    大模型 (70B/14B/32B): 1.16×-1.22×                    │
│                                                          │
│  DSM Bandwidth Profiling:                                │
│    dsm_shuffle ≈ dsm_all_exchange > dsm_reduce_scatter  │
│    Cluster=2: ~8 TB/s, Cluster=16: ~4 TB/s              │
│    DSM latency ~20ns vs HBM ~280ns (14× faster)         │
│                                                          │
│  搜索效率:                                               │
│    搜索空间: ~1.15×10^6 候选                             │
│    Cost model pruning → Top-11 → profiling              │
│    离线搜索时间: 未明确（估计分钟级）                    │
└─────────────────────────────────────────────────────────┘
```

---

## 六、算子融合的设计方法论（总结）

从 FlashFuser 和同类工作（Welder、Kitsune、SN40L、Chimera）中可以提炼出算子融合的通用设计框架：

```
┌────────────────────────────────────────────────────────┐
│              算子融合设计空间五维框架                    │
├────────────────────────────────────────────────────────┤
│                                                         │
│  1. 融合介质（数据留在哪？）                            │
│     Reg → SMEM → DSM → L2 → 全片上 SRAM                │
│     ↑ 越高越快，容量越受限                              │
│                                                         │
│  2. 调度策略（如何组织执行？）                          │
│     - 时间复用: 单个 mega-kernel 内串行各阶段           │
│     - 空间并发: 不同 SM 执行不同算子（dataflow 风格）   │
│     - 混合: FlashFuser 两者兼用                         │
│                                                         │
│  3. Tile 对齐（如何让不同算子的 tile 兼容？）           │
│     - 手工规则: TensorRT/Chimera（脆弱，不通用）        │
│     - Propagation: Welder（自动推导，通用）             │
│     - Loop Schedule: FlashFuser（枚举+Cost Model）      │
│                                                         │
│  4. 搜索方法（如何在巨大空间中找最优解？）              │
│     - 穷举 + profiling: Chimera                        │
│     - Cost Model + pruning + profiling: FlashFuser     │
│     - 双层解耦搜索: Welder                              │
│                                                         │
│  5. 硬件依赖（需要什么硬件特性？）                      │
│     - 通用: 任何有 SMEM 的 GPU 可做 L1 融合             │
│     - Hopper only: FlashFuser 的 DSM fusion             │
│     - 需改硬件: Kitsune 的双 arbiter grid scheduler     │
│     - 专用硬件: SN40L 原生 streaming                    │
└────────────────────────────────────────────────────────┘
```

**核心洞察**：算子融合本质上是一个 **memory planning 问题**——不是"能不能算得更快"，而是"能不能少搬运数据"。FlashFuser 的贡献在于发现了 DSM 这片"新大陆"：它既像 SMEM 一样快（~20ns），又跨越了单 SM 的容量限制（227KB→跨 SM 共享），使得之前因中间 tensor 太大而无法融合的 GEMM 链（LLM 推理的核心负载）首次可以在片上完成。

你选中的那条笔记 `~1.15×10^6 候选→Top-11 profiling` 本质上是 FlashFuser 的搜索剪枝策略——在面对百万级组合空间时，通过 cost model（C = max(V_l / B_l)，最慢 memory level 决定延迟）快速剪枝到 11 个候选，然后硬件实测选出最优。这个"搜索编译器"的思路（而非手工写融合规则）是整个 Welder/FlashFuser 路线的核心哲学。
