# Welder Tile-Graph 算子融合

- 导出时间：2026-06-03
- 来源：当前 Claude Code session 可用对话上下文
- 保存路径：human_notes/Welder-Tile-Graph-算子融合.md
- 范围：仅用户输入与 Claude 最终输出

## 对话记录

### 001 User
（IDE 选中 L1_horizon_summary.md 中 Welder 行：`**算子融合** | Welder Tile-Graph Memory | Tile propagation 自动对齐 + traffic cost model + 双层搜索；89 种 fusion pattern 自动发现 | DRAM traffic -69% (BERT); NeRF 7-layer MLP 全融合 5× | Q1.6 |`）

教我

### 002 Claude

## 1. 背景：为什么需要算子融合？

先理解 **算子融合要解决的根本问题**：

```
传统执行方式（每个算子独立 kernel）:
  Matmul Q×K^T:  [DRAM 加载 Q,K] → TensorCore 计算 → [C 写回 DRAM]
  Softmax:       [DRAM 加载 C]   → SIMT 计算      → [P 写回 DRAM]
  Matmul P×V:    [DRAM 加载 P,V] → TensorCore 计算 → [输出写回 DRAM]
  
  问题: 中间张量 C 和 P 完整地在 DRAM 中"物化"了两轮（写→读）
  → 大量 DRAM 带宽浪费在中间结果上，而非真正的计算数据上
```

**核心矛盾**：现代 DNN（ViT、Conformer、NeRF、NAFNet 等）的内存带宽利用率高达 **96.7%**，而计算单元利用率仅 **51.6%**——这意味着现在的瓶颈已经不是"算得不够快"，而是"数据搬得不够快"。传统的计算中心优化（如减少 FLOPs）已触达收益递减点，瓶颈转移到了**内存访问**。

算子融合的核心思想是：**让中间结果不离开 on-chip memory（Shared Memory / Register），直接在片上从一个算子传递给下一个算子**，省去 DRAM 的往返开销。

---

## 2. 现有方法的局限

在 Welder 之前，算子融合主要靠三种方式：

| 方法 | 代表 | 局限 |
|------|------|------|
| **预定义规则** | TensorRT, BladeDISC | 只覆盖已知模式（如 Conv+ReLU），遇到未定义的算子组合直接降级为无融合 |
| **Register 级融合** | TVM/Ansor | 只能融合 element-wise 算子，**无法融合两个 reduction 算子**（如 Matmul+Softmax）——因为两个算子的最优 tile shape 冲突 |
| **搜索式融合** | Rammer, Nimble | 只做**水平融合**（并行算子并发执行），不做**垂直融合**（依赖算子间的 on-chip 数据复用） |

**根本难题**：不同算子的最优 tile shape 不同。比如 Matmul 喜欢 `[32×64]` 的 tile，Softmax 喜欢 `[4×128]` 的 tile——直接强行对齐会导致某一个算子性能急剧下降。这使得 "Matmul + Softmax" 这种最常见的 attention 模式一直难以在 shared memory 中融合。

---

## 3. Welder 的核心洞察：从算子级下沉到 Tile 级

Welder 的关键创新是**改变问题的抽象层级**：

```
传统视角（算子级）:
  [Matmul Op]  ──DRAM──→  [Softmax Op]
  ↑ 每个算子是原子单位，连接在 DRAM

Welder 视角（Tile 级）:
  [Matmul tile₀] ──SMEM──→ [Softmax tile₀]
  [Matmul tile₁] ──SMEM──→ [Softmax tile₁]
  [Matmul tile₂] ──SMEM──→ [Softmax tile₂]
  ↑ 每个 tile 是原子单位，连接可选 DRAM/L1(SMEM)/L0(Reg)
```

**把一个算子拆成多个 operator-tile**——每个 tile 处理一个 `[BM, BN]` 大小的 output 数据块。然后在 tile 之间建立连接（`SetConnect`），并指定连接发生在哪个 memory level：

- **L0 (Register)**：最快速，容量最小（256KB/SM）
- **L1 (Shared Memory)**：快速，容量 227KB/SM（H100）
- **L2 (Global Memory / DRAM)**：慢速，容量大

这样，算子融合问题就转变成了：**在 tile-graph 上为每条边选择最优的连接层级（L0/L1/L2），使得总 DRAM traffic 最小。**

---

## 4. 四大核心机制详解

### 4.1 Tile Propagation（Tile 形状自动对齐）

这是 Welder 最巧妙的设计。给定一个算子的张量表达式（tensor expression），**从 output tile shape 可以反向推导出需要的 input tile shape**：

```
给定 Matmul: O[i,j] = Σₖ A[i,k] × B[k,j]

如果 output tile 取 O[i₀:i₁, j₀:j₁]（大小 [BM, BN]），
则必然需要:
  A 的 tile: A[i₀:i₁, :]     （大小 [BM, BK]）
  B 的 tile: B[:, j₀:j₁]     （大小 [BK, BN]）

→ output tile shape 唯一确定了所有 input tile shape！
```

这个性质沿着整个 tile-graph 链式传播——只要确定了最终 output 的 tile shape，整个图中所有 operator tile 的 shape 都被自动推导出来，**相邻算子之间的 tile 自然对齐，无需手工规则**。

这正是 Welder 能自动发现 **89 种非常规融合模式**（包括 Matmul+Softmax 这种 reduction 算子组合）的数学基础。

### 4.2 Inter-Layer Independence + Traffic Cost Model

Welder 发现了一个关键性质：**Inter-layer independence**——每一层 memory level 的总 traffic **仅由该层的 output tile 配置决定**，与上下层的选择无关。

这意味着：
- 原本耦合的、搜索空间大小为指数级的多层联合优化问题
- 被解耦为多个独立的线性级子问题
- 每个 memory level 可以独立搜索最优 tile 配置

Traffic cost model 是解析计算的（不需要实际执行）：
```
memory_traffic = Σ (所有 input_tile 大小 + output_tile 大小) × tile_graph 数量
```

这个 cost model 使得搜索速度极快——几秒内可以评估数万个 tile 配置。

### 4.3 双层搜索（Two-Level Search）

```
外层 — Graph Connecting:
  枚举计算图中每条边连接到哪个 memory level (L0/L1/L2)
  → 这是 coarse-grained 的决策：哪些算子之间在 SMEM 融合，哪些在 DRAM 断开

内层 — Sub-Graph Tiling:
  给定外层连接决策，搜索该子图内的最优 tile 配置
  → 基于 traffic cost model + 容量约束 (MemFootprint ≤ SMEM capacity)
  → 通过 hardware-aligned penalty 过滤不良配置
    (uncoalesced memory access / inadequate parallelism / over-capacity)
```

两层搜索的结果是一组完整的执行计划：每条边在哪个 memory level 连接、每个 tile 的 shape 是什么。

### 4.4 代码生成：四指令抽象

Welder 只依赖四条硬件抽象指令，因此可以映射到任意有层次化内存的加速器：

```
LoadTiles:   从指定 memory level 加载 tile 数据
ComputeTile: 在当前 tile 上执行算子计算（TensorCore / SIMT）
StoreTiles:  将 tile 结果写回指定 memory level
Allocate:    在目标 memory level 上分配 buffer
```

---

## 5. 完整执行示例：BERT Attention Block

以 BERT 的 Attention 块（Matmul Q×K^T → Softmax → Matmul P×V）为例：

**优化前（无融合）**:
```
Kernel 1 (Matmul Q×K^T):
  DRAM→SMEM 加载 Q_tile, K_tile
  TensorCore MMA 计算 → C_tile [BM×BN]
  SMEM→DRAM 写回 C_tile  ← 第一次 DRAM 写

Kernel 2 (Softmax):
  DRAM→SMEM 加载 C_tile  ← 第一次 DRAM 读（C 被物化了）
  SIMT 计算 softmax → P_tile
  SMEM→DRAM 写回 P_tile  ← 第二次 DRAM 写

Kernel 3 (Matmul P×V):
  DRAM→SMEM 加载 P_tile, V_tile  ← 第二次 DRAM 读（P 被物化了）
  TensorCore MMA 计算 → output
  写回 DRAM

总 DRAM traffic: 840MB（中间结果 C 和 P 各完整搬运了一次来回）
```

**优化后（Welder 融合）**:
```
Welder Fused Kernel (单 kernel):
  外层循环遍历所有 tile:
    LoadTiles: DRAM→SMEM 加载 Q_tile[BM×BK], K_tile[BK×BN]
    ComputeTile (Matmul): TensorCore MMA → C_tile[BM×BN]
      ↓ C_tile 直接留在 SMEM（SetConnect at L1）← 无 DRAM 写！
    ComputeTile (Softmax): SIMT 从 SMEM 读 C_tile → P_tile[BM×BN]
      ↓ P_tile 直接留在 SMEM（SetConnect at L1）← 无 DRAM 写！
    ComputeTile (Matmul): 从 SMEM 读 P_tile, V_tile → output

总 DRAM traffic: 264MB（节省 69%）
Kernel launch 次数: 3→1
```

---

## 6. 关键实验结论

| 场景 | 融合规模 | 效果 |
|------|----------|------|
| BERT Attention | Matmul + Softmax 对 | DRAM traffic **-69%** (840→264MB) |
| NeRF 7-layer MLP | **全部 7 层融合为单个 kernel** | **5× speedup** |
| NAFNet | DWConv+PWConv+Normalization 自适应融合顺序 | 自动选择最优融合策略 |
| 通用发现 | **89 种非常规 fusion pattern** | 含 48-operator 超长融合链 |

最惊人的是，Welder 是**自动**发现这些融合模式的——编译器自己枚举 tile-graph 上的连接配置并选择最优，不需要任何手工规则。

---

## 7. 与其他融合方法的对比

| 方法 | 粒度 | 关键机制 | 限制 |
|------|------|----------|------|
| **TVM/Ansor** | Register 级 | 手工规则 + ML 搜索 | 无法融合 reduction 算子对 |
| **TensorRT** | 算子级 | 专家手工设计的融合规则 | 不支持的算子组合完全无法融合 |
| **FlashFuser** | SM Cluster 级 | Hopper DSM 跨 SM 共享 | 依赖 H100 硬件特性（DSM） |
| **SN40L Streaming** | 硬件级 | 数据流架构原生 streaming | 需要专用硬件 |
| **Welder** | **Tile 级** | Tile-graph + Propagation + Traffic Model | 通用性强，V100/A100 即可 |

Welder 的独特之处在于：**它不改变任何硬件，只改变"数据在 memory hierarchy 中的流动方式"**——这是一个纯编译器层面的优化，但收益堪比专用硬件。

---

## 总结一句话

> **Welder 将算子融合从"为每一对算子写专门的融合规则"的手工工程问题，转变为了"在 tile-graph 上搜索最优 tile 连接配置"的自动图优化问题。** 三个关键洞察支撑这一转变：(1) Tile Propagation 自动对齐——解决 tile shape 冲突；(2) Inter-Layer Independence——将指数级搜索空间降为线性；(3) Traffic Cost Model——解析计算无需实际执行。
