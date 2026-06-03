# PROBE Greedy Planning — 内存规划详解

- 导出时间：2026-06-03
- 来源：当前 Claude Code session 可用对话上下文
- 保存路径：human_notes/PROBE_Greedy_Planning_内存规划详解.md
- 范围：仅用户输入与 Claude 最终输出
- 关联论文：PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
- 关联学习输出：learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md

## 对话记录

### 001 User
（用户选中了 L1_horizon_summary.md 第 122-123 行，内容为 PROBE Greedy Planning 在 Memory Planning 分类下的表格行：T_window 约束 + water-filling 策略 + max kmax=16 iterations | 每 rank T_window 按计算-带宽比动态确定）

教我

### 002 Claude
好的，我来详细讲解你选中的这一行——**PROBE Greedy Planning**。这是 PROBE 论文中最精妙的设计之一。

---

## 先理解大背景：PROBE 要解决什么问题？

在 MoE（Mixture of Experts）推理中，不同 GPU rank 上分配的 expert 计算量是不均匀的——某些 expert 是"热门"，被更多 token 选中。这导致：

```
Rank 0: [████████████████]  ← 热门 expert，计算多
Rank 1: [████            ]  ← 冷门 expert，早早算完
Rank 2: [██████          ]
Rank 3: [████████        ]
                ↑
         所有 rank 在 All-to-All barrier 处等待 Rank 0
         → ~50% 全局算力闲置！
```

更致命的是 **"Double Penalty"（双重惩罚）**：热门 expert 的 rank 不仅计算慢（计算 skew），而且收发数据量也最大（网络 skew）——计算延迟和网络拥塞**串行叠加**，而非并行掩盖。

PROBE 的核心思路是：**不等 token 来再反应，而是预测→规划→预取**。把热门 expert 的权重提前复制到空闲 rank 上，让它们分担计算。

---

## Memory Planning 的核心：Greedy Balance Planning Solver

你选中的这一行属于 Q1.6（面向硬件的设计空间与最佳实践）的 Memory Planning 类别。PROBE 的 Greedy Planning 本质上是一个**带约束的硬件感知内存/计算均衡规划器**：

```
PROBE Greedy Planning = T_window 约束
                      + water-filling 策略
                      + max k_max=16 iterations
                      + T_window 按计算-带宽比动态确定
```

让我逐项拆解：

---

### 1. T_window 约束：隐藏窗口

**定义**：`T_window^r` 是 rank `r` 上**非通信 kernel 的执行时间窗口**——即 MoE Compute（expert GEMM）和下一层 Attention 的执行时间之和。

```
时间轴 (单个 rank):
├── Dispatch ──┼──── MoE Compute ────┼── Combine ──┼── Attention ──┤
               ├────── T_window^r ────────────────┤
               ↑                               ↑
           P2P expert 传输必须在这个窗口内完成，
           否则会暴露在关键路径上，增加延迟
```

**为什么叫"约束"？** 因为 expert 权重的 P2P 传输时间 `T_trans^r` 必须满足：

$$T_{trans}^r \leq T_{window}^r$$

如果复制太多 expert，传输时间超过窗口，就会"溢出"到关键路径上，反而增加延迟。所以 T_window 是一个**硬约束**——复制多少 expert 由此决定。

---

### 2. T_window 的动态计算：计算-带宽比

这是该行最精妙的部分——**每 rank 的 T_window 不是固定的，而是按该设备的计算-带宽比动态确定**：

$$T_{window}^r = f\left(\frac{\text{Compute}_r}{\text{Bandwidth}_r}\right)$$

**直观理解**：

| 设备类型 | 计算-带宽比 | T_window | 可复制 expert 数 |
|----------|------------|----------|------------------|
| H100 (高算力, NVLink 900GB/s) | 中等 | 中等 | 适量 |
| A100 (算力较低, NVLink 600GB/s) | 较低 | 较大 | 较多（算力缺口大，但窗口也大） |
| 低算力高带宽设备 | 低 | **大** | 多——有充足时间传输更多 expert |
| 高算力低带宽设备 | 高 | **小** | 少——算力强不需太多帮助，且带宽紧张 |

**设计哲学**：如果设备算力弱，它需要更多来自其他 rank 的 expert 复制来填充计算；同时它的计算窗口也更大（算得慢），给了更多传输时间。反之，算力强的设备自身的 expert 已经够算，不需要太多外部复制。

---

### 3. Water-Filling 策略：贪心注水均衡

Water-filling（注水算法）是信息论中的经典策略，PROBE 将其用于 expert 复制决策：

```
Algorithm: Greedy Water-Filling Balance Planning

输入: 预测的 per-rank expert token 分布 n̂[r][e]
      当前 expert placement P'
      每 rank 的 T_window^r

初始化: 所有 rank 的预估延迟 L[r] = T_comp^r + T_comm^r

for iteration = 1 to k_max (=16):           ← 最多 16 轮迭代
    # Step 1: 找瓶颈 rank
    r_bottleneck = argmax(L[r])              ← 当前最慢的 rank

    # Step 2: 找该 rank 上最热的 expert
    e_hot = argmax(n̂[r_bottleneck][e])       ← token 最多的 expert

    # Step 3: 找最低负载的目标 rank（注水！）
    r_target = argmin(L[r])                  ← 当前最快的 rank
    约束: T_trans(r_bottleneck→r_target, e_hot) ≤ T_window^r_target

    # Step 4: 注水——复制 expert 并更新负载
    if 满足约束:
        复制 e_hot 从 r_bottleneck 到 r_target
        更新 n̂[r_target][e_hot] += 转移的 token 量
        更新 L[r_bottleneck] -= 减轻的负载
        更新 L[r_target] += 新增的负载
    else:
        标记 e_hot 为"本轮不可转移"，尝试次热 expert

    if 无法找到可转移的 expert:
        break  ← 收敛

输出: Δ_r^{in}（每个 rank 需接收的 expert 集合）
      Δ_r^{out}（每个 rank 需驱逐的 expert 集合）
      A（更新后的 token-to-expert 分配表）
```

**"注水"的含义**：像往多个容器中注水一样，从"水位"最高的 rank（瓶颈）往"水位"最低的 rank 转移负载，直到所有 rank 的水位趋近均衡。

**可视化**：

```
迭代前:                         迭代后 (经 water-filling):
Rank 0: ████████████ (瓶颈)     Rank 0: ████████
Rank 1: ████                    Rank 1: ████████  ← 接收了 Rank 0 的热门 expert
Rank 2: ██████                  Rank 2: ████████
Rank 3: ████                    Rank 3: ██████

IR (Imbalance Ratio): 3.0      IR: ~1.09
```

---

### 4. k_max = 16：为什么最多 16 轮？

**硬上限的来源**：

1. **单 SM 执行的实时性约束**：Planning Solver 运行在**单个 SM** 上，与 MoE Compute 的 Grouped GEMM **并发执行**（利用 GPU 剩余 SM 资源）。如果迭代太多，solver 自身延迟会超过 MoE Compute 窗口，暴露在关键路径上。

2. **收敛性经验值**：实验表明 16 轮内 water-filling 基本收敛——IR 从 >2.0 降至 ~1.09。更多迭代的边际收益极小。

3. **这个约束也反映了 PROBE 的设计哲学**：**"够好就行"而非"最优"**——用贪心近似换取零关键路径开销，比追求全局最优但增加延迟更明智。

```
Plan Solver 的时间约束:
├──── MoE Compute ────────────────────┤
├── Plan Solver (≤16 iter, 1 SM) ────┤  ← 必须在这个窗口内完成
```

---

### 5. 放在 Memory Planning 分类下的原因

你可能会疑惑：这看起来更像是"负载均衡"或"调度"，为什么 Horizon Summary 把它归入 **Memory Planning**？

从 Q1.6 的视角看，PROBE 的 Greedy Planning 涉及**显存资源的规划和分配**：

- **Replicated Expert Buffer**：每个 rank 使用 NVSHMEM symmetric memory 维护一个双缓冲 expert 副本区（最多 6 slots），需要在有限显存中规划哪些 expert 权重驻留
- **T_window 本质是显存带宽约束**：`T_window` 由计算-带宽比决定，限制了 P2P 传输的数据量 = 限制了可复制的 expert 权重总字节数
- **与 Welder Tile-Graph、MPK Paged SMEM 并列**：三者都是解决"在有限显存/带宽下如何最优放置数据"的问题——Welder 是 tile 级 memory traffic 优化，MPK 是 SMEM 页式管理，PROBE 是 expert 权重的跨 rank 显存规划

---

### 总结：一张图理解全貌

```
┌──────────────────────────────────────────────────────────────────┐
│                  PROBE Greedy Balance Planning                     │
│                                                                   │
│  输入                    约束                   输出              │
│  ┌──────────┐    ┌──────────────────┐    ┌──────────────┐        │
│  │ 预测 n̂   │    │  T_window^r =    │    │ Δ_r^{in}     │        │
│  │ placement│───▶│  f(Compute/      │───▶│ Δ_r^{out}    │        │
│  │   P'     │    │    Bandwidth)    │    │ token assign │        │
│  └──────────┘    └──────────────────┘    └──────────────┘        │
│                        │                                          │
│                   Water-Filling                                   │
│                   ≤ 16 iterations                                 │
│                   single SM CUDA kernel                           │
│                                                                   │
│  关键性质:                                                        │
│  • 贪心非最优 → 换取零关键路径开销                                │
│  • 硬件感知 → 不同设备自适应 window 大小                          │
│  • 与 MoE Compute 完全重叠 → 对用户透明                           │
│                                                                   │
│  效果: IR 2.13→1.09, Max/Avg 计算延迟比 2.27→1.18                │
└──────────────────────────────────────────────────────────────────┘
```

这个设计的优雅之处在于：**它把"均衡"这个全局优化问题分解为每个 MoE 层的局部贪心决策，用预测打破了"必须先路由再均衡"的串行依赖，并且把规划开销完全隐藏在计算窗口内——用户感知到的只有更低的延迟和更高的吞吐。**

---

### 003 User
gpu rank?

### 004 Claude
在分布式计算/MoE 推理的语境下，**rank** 是一个核心概念：

---

## Rank = 分布式进程的唯一标识符

**定义**：Rank 是分布式计算中每个进程（process）的唯一编号。在 GPU 场景下，**1 rank = 1 个 GPU 上运行的 1 个进程**。Rank 编号从 0 到 N-1（N=总 GPU 数）。

```
EP Group (Expert Parallelism, ep=8):

Rank 0    Rank 1    Rank 2    Rank 3    Rank 4    Rank 5    Rank 6    Rank 7
┌────┐   ┌────┐   ┌────┐   ┌────┐   ┌────┐   ┌────┐   ┌────┐   ┌────┐
│GPU0│   │GPU1│   │GPU2│   │GPU3│   │GPU4│   │GPU5│   │GPU6│   │GPU7│
│    │   │    │   │    │   │    │   │    │   │    │   │    │   │    │
│E0  │   │E1  │   │E2  │   │E3  │   │E4  │   │E5  │   │E6  │   │E7  │
│E8  │   │E9  │   │E10 │   │E11 │   │E12 │   │E13 │   │E14 │   │E15 │
│... │   │... │   │... │   │... │   │... │   │... │   │... │   │... │
└────┘   └────┘   └────┘   └────┘   └────┘   └────┘   └────┘   └────┘
  ↑ 每个 rank 持有部分 expert 的权重
```

---

## 为什么叫 "Rank"？

这个词源自 **MPI（Message Passing Interface）** 标准。MPI 用 `MPI_Comm_rank()` 获取当前进程在通信组中的排名/编号。深度学习分布式框架（PyTorch DDP/FSDP、DeepSpeed、NCCL）沿用了这个术语。

```python
# PyTorch 分布式中的典型用法
import torch.distributed as dist
rank = dist.get_rank()       # 当前进程的 rank 编号
world_size = dist.get_world_size()  # 总进程数 = 总 GPU 数
```

---

## 在 PROBE/MoE 场景下的具体含义

在 MoE 的 Expert Parallelism (EP) 中：

| 概念 | 含义 | PROBE 中的例子 |
|------|------|---------------|
| **Rank** | 一个 GPU 上的一个进程 | 8×H800 → 8 个 rank (rank 0~7) |
| **Per-rank expert placement** | 每个 rank 持有哪些 expert 的权重 | Rank 0 持有 E0, E8, E16... |
| **T_window^r** | 第 r 号 rank 的计算窗口时间 | Rank 3 因算力弱 → T_window 大 |
| **Bottleneck rank** | IR 最大、拖慢全局的 rank | "Rank 0 是 bottleneck，IR=2.6" |
| **IR (Imbalance Ratio)** | max(L_r) / mean(L_r)，跨 rank 的负载不均衡度 | 各 rank 负载差异越大，IR 越高 |

```
PROBE 的 per-rank 视角:

         Rank 0           Rank 1           Rank 2           Rank 3
        (热门)           (冷门)           (中等)           (冷门)
     ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
     │ T_comp=8ms│    │ T_comp=3ms│    │ T_comp=5ms│    │ T_comp=2ms│
     │ T_comm=2ms│    │ T_comm=1ms│    │ T_comm=1ms│    │ T_comm=1ms│
     │ L=10ms ←──瓶颈             │    │          │    │          │
     └──────────┘    └──────────┘    └──────────┘    └──────────┘
          ↑
     Water-filling: 把 Rank 0 的热门 expert 复制到 Rank 1, Rank 3
     → 所有 rank 的 L ≈ 5ms, IR 从 5.0 降至 ~1.1
```

---

## 简单记忆

> **Rank = GPU 进程编号 = 分布式世界中"这台机器叫什么名字"**

在单机 8 卡场景下就是 0~7；在多机场景下，比如 4 台机器各 8 卡，rank 范围是 0~31。PROBE 的 Greedy Planning Solver 就是**在每个 rank 上独立运行**（单 SM），用 All-Gather 同步 per-rank 预测结果，然后各自决定本 rank 需要接收/驱逐哪些 expert。
