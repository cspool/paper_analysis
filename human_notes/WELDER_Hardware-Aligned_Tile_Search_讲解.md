# WELDER Hardware-Aligned Tile Search 讲解

- 导出时间：2026-06-04
- 来源：当前 Claude Code session 可用对话上下文
- 保存路径：human_notes/WELDER_Hardware-Aligned_Tile_Search_讲解.md
- 范围：仅用户输入与 Claude 最终输出

## 对话记录

### 001 User
（IDE 选中 review_draft.md 第 138 行：`| **Tile 切分与调度** | WELDER Hardware-Aligned Tile Search* | 三级硬件 penalty 驱动 tile shape 枚举：coalesced (128B transaction) + parallelism (≥128 tiles) + capacity (SMEM footprint)。MMA fragment 整除约束 (16 for FP16) | 枚举空间 = 各维度 tile size 笛卡尔积；penalty 各自 O(1) 计算；multiple candidates 可并行评估 | Q4.1, Q4.4 |`）

教我

### 002 Claude

## 一句话概括

WELDER 的 tile search 做了一件很直觉的事：**在枚举所有可能的 tile shape 时，用三个"硬件罚分"（penalty）淘汰掉硬件不友好的候选，让最终选出的 tile 天然适合 GPU 物理约束。**

---

## 1. 背景：为什么需要 Tile Search？

在 GPU kernel 开发中，选择一个 tensor 操作的 tile size/shape 是最关键的决策之一。tile 选大了 → SMEM 装不下 → 无法运行；tile 选小了 → GPU 上并行的 tile 太少 → SM 吃不饱；tile 的内维没对齐 → 非合并内存访问 → 有效带宽暴跌。

手动调 tile 极其 tedious，所以 WELDER（OSDI'22，微软研究院）提出自动搜索。但纯基于 memory traffic 的搜索会选出"数学上最优但硬件上不可行"的配置。因此 WELDER 在 traffic cost model 上叠加三个硬件 penalty，把搜索空间约束到硬件友好的子集。

---

## 2. 三级 Penalty 详解

### Penalty 1: Coalesced Memory Access（合并访存）— `128B transaction`

**硬件背景**：NVIDIA GPU 的 L1 cache line 宽度是 128 字节。一次 global memory transaction 搬运 128B（= 32 个 FP32 元素）。当一个 warp 的 32 个线程访问的地址落在同一个 128B cache line 内 → **1 次 transaction** → 100% 带宽利用率。如果分散在多个 cache line → 最多 32 次 transaction → 仅 3% 有效带宽。

```
对齐的 coalesced 访问:                   未对齐的 scatter 访问:
Warp 32 threads: [t0][t1][t2]...[t31]    Warp 32 threads: [t0] [t1]  [t2]  ...  [t31]
                    ↓ 同一 128B line                       ↓     ↓      ↓           ↓
                 1 次 transaction                    最多 32 次 transaction
                 BW 利用率: ~100%                     BW 利用率: ~3%
```

**WELDER 的做法**：检查 tile 的 leading dimension（最内维的 stride）是否整除 32（FP32）/ 128B。不整除 → 计算额外需要的 transaction 数 → 加到 penalty 上。

```
if not is_coalesced(tile, transaction_width=128B):
    extra_traffic = calculate_extra_transactions(tile)
```

**通俗理解**：好比高速公路收费——一辆大巴拉 32 人过一次收费站（coalesced）vs 32 辆摩托车每人过一次（scatter）。后者过路费（memory transaction）贵 32 倍。WELDER 的 penalty 就是把这个额外成本算进 tile 的"总价"里。

---

### Penalty 2: Parallelism Underutilization（并行度不足）— `≥128 tiles`

**硬件背景**：V100 有 80 个 SM，每个 SM 有 4 个 warp scheduler。为了"喂饱"所有 SM，并行 tile 数至少需要 ≥ 80（理想 ≥ 128，让每个 SM 有足够 warp 切换来隐藏 memory latency）。

```
V100: 80 SMs × 4 warp schedulers × 1 warp/scheduler = 320 warps 可同时活跃
      最低要求: ≥128 并行 tile（保证基本 SM 占用）
```

如果 tile 太大 → tile 总数太少 → 很多 SM 空闲 → 有效吞吐下降。

**WELDER 的做法**：

```
num_parallel_tiles = total_elements / tile_size
if num_parallel_tiles < MIN_PARALLELISM(128):
    extra_traffic *= (MIN_PARALLELISM / num_parallel_tiles)
```

例如：tile 太大导致只有 64 个并行 tile → `extra_traffic *= 128/64 = 2` → traffic 翻倍，这个 tile 在排序中排名下降。

**通俗理解**：把 128 块砖（tile）分给 80 个工人（SM），每个人至少搬一块。如果只有 64 块砖，40% 的工人闲着没事干。WELDER 的 penalty 把"工人空闲率"折成额外成本。

---

### Penalty 3: Capacity Overflow（容量溢出）— `SMEM footprint`

**硬件背景**：V100 每个 SM 有 96KB shared memory（可配置至多 164KB 的 L1/SMEM split）。tile 的中间数据（输入 tile + 权重 tile + 输出 tile + pipeline buffer）必须 fit 在 SMEM 中，否则 kernel 编译失败或 spill 到 HBM（性能灾难）。

**WELDER 的做法**：

```
footprint = MemFootprint(graph_with_tile_config)
if footprint > target_memory_capacity:
    continue  # infinite penalty → 直接淘汰此 tile shape
```

这是最硬的约束——**不满足就淘汰，不留商量的余地**。

**通俗理解**：行李箱（SMEM）容量有限。东西（tile 数据）塞不进去 → 这个 tile shape 直接出局，不参与后续比较。

---

### 额外约束：MMA Fragment 整除 — `16 for FP16`

Tensor Core 的 `mma.m16n8k16` 指令一次处理 16×8×16 的矩阵 fragment。这意味着 tile 的 M 维必须整除 16，N 维整除 8（实践中通常对齐到 16），K 维整除 16。

```
for axis marked as MMA_axis:
    enforce tile[axis] % MMA_FRAGMENT_SIZE(16, FP16) == 0
```

不整除 → Tensor Core 无法精确映射 → 要么 padding 浪费计算，要么无法使用 Tensor Core。WELDER 将这个作为硬约束（而非 penalty）。

---

## 3. 搜索算法

```
搜索空间: 各维度 tile size 的笛卡尔积
        M_dim_sizes × N_dim_sizes × K_dim_sizes × ...

搜索策略: 从 init_tile = {axis: 1} 开始
        沿每个 axis 逐步扩展 tile size
        对每个候选 tile:
            Penalty 1: coalesced?      → O(1) 检查
            Penalty 2: parallelism?    → O(1) 计算
            Penalty 3: capacity?       → O(1) MemFootprint 估算
            MMA 约束: 整除 16?          → O(1) 取模
        将合法候选按 adjusted_traffic 排序

并发性: 多候选 tile shape 可并行评估
        penalty 计算彼此无数据依赖
        每种候选独立计算 adjusted_traffic 后全局排序
```

关键设计点：
- **O(1) 计算**：每个 penalty 都是常数时间检查，不需要实际运行 kernel
- **可并行评估**：多个 tile shape 候选没有数据依赖，可以并行计算 penalty
- **排序选择**：最终选 adjusted_traffic 最小的配置（而非原始 memory traffic 最小的）

---

## 4. 在 review table 中的位置

WELDER 属于 **Tile 切分与调度** 类别。与同列的其他方法对比：

| 方法 | 核心思路 | 搜索方式 |
|------|----------|----------|
| **WELDER** | 三级硬件 penalty 驱动 tile 枚举 | 编译期 O(1) penalty 搜索 |
| ThunderKittens | 三层抽象 (Warp→Warpgroup→CTA) | 固定的 16×16 warp tile，编译期选 swizzle |
| FlashAttention | IO-aware: 最小化 HBM 访问 | 解析公式 B_c=ceil(M/(4d)) |
| CUTLASS | 分层 tile 迭代器 | 遵循 MMA 对齐 + penalty 双重约束 |
| Infera | 零调优编译 | 36 种预定义配置 + ILP/Intensity 分析 |

WELDER 的独特之处在于：**不依赖 profiling（如 TVM/AutoTVM），不依赖解析公式（如 FlashAttention），用纯静态的硬件 penalty 在编译期完成 tile 选择。** 这比 profiling-based 方法快 2-3 个数量级。

---

## 5. 一句话总结

> **WELDER 把 GPU 硬件手册里的三条铁律——128B 对齐访存、至少 128 个并行 tile、SMEM 不能超——编码为三个 O(1) 的 penalty 函数，插入 tile shape 枚举循环里，让搜索自动避开硬件反模式，选出"硬件开心"的 tile 配置。**
