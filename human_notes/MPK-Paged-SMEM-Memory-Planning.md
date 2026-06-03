# MPK Paged SMEM — Memory Planning

- 导出时间：2026-06-03 19:30 CST
- 来源：当前 Claude Code session 可用对话上下文
- 保存路径：human_notes/MPK-Paged-SMEM-Memory-Planning.md
- 范围：仅用户输入与 Claude 最终输出
- 论文背景：MoEDiT 多模态 Video — L1_horizon_summary Q1.6

## 对话记录

### 001 User
（IDE 选中 L1_horizon_summary.md 中的一行：**Memory Planning** | MPK Paged SMEM | 32KB fixed pages + interval graph coloring 复用 + cross-task pipelining 预取 | H100: 7 pages/SM, A100: 5 pages/SM | Q1.6）

教我

### 002 Claude
# 教你：Memory Planning — MPK Paged SMEM

你选中的这行来自 MoEDiT 多模态 Video 的 L1 Horizon Summary，属于 **Q1.6（面向硬件的设计空间与最佳实践）** 中的 Memory Planning 条目。我来逐层拆解。

---

## 一、先理解背景：为什么需要 Memory Planning？

在 GPU 上执行深度学习模型时，传统的做法是 **"一个算子 = 一个 CUDA kernel"**。比如 Q 投影是一个 kernel，K 投影是另一个 kernel，Attention 又是一个 kernel……每个 kernel 独立 launch、独立使用 shared memory（SMEM）、用完释放。

**问题在哪？**

1. **Kernel launch overhead**：每次 launch ~100µs，一个 Transformer 层可能有几十个 kernel，累积延迟可观
2. **SMEM 利用率低**：每个 kernel 用完 SMEM 就释放了，下一个 kernel 重新分配——中间数据无法跨 kernel 复用
3. **无法做跨 kernel 的流水线**：kernel 之间有硬件 barrier，前一个 kernel 全部完成才能开始下一个

**MPK (Mirage Persistent Kernel)** 的思路是：把所有算子塞进**一个 mega-kernel**，在 kernel 内部自己管理 SMEM 的分配/释放/复用/预取——这就是 **Memory Planning** 的用武之地。

---

## 二、MPK Paged SMEM 是什么？

**核心概念**：把每个 SM 的 shared memory（H100 上 227KB/SM）分成**固定大小的 page（32KB）**，task 按需 acquire/release pages，就像操作系统的虚拟内存管理一样。

```
┌────────────────── SM Shared Memory (227KB on H100) ──────────────────┐
│  Page 0 (32KB)  │  Page 1 (32KB)  │  Page 2 (32KB)  │ ... │ Page 6  │
│  [Task A data]  │  [Task B data]  │  [空闲]          │     │[预取中]  │
└──────────────────────────────────────────────────────────────────────┘
```

**为什么选 32KB？**
- H100 SMEM = 227KB → 227/32 ≈ 7 pages/SM
- A100 SMEM = 164KB → 164/32 ≈ 5 pages/SM  
- 32KB 是 TMA（Tensor Memory Accelerator，Hopper 的异步 DMA 引擎）的 natural transfer granularity，对齐硬件最优

---

## 三、Interval Graph Coloring 复用

这是编译期决定**哪些 tensor 可以共享同一个 page** 的方法。

### 3.1 问题

Mega-kernel 内部有几十个 task（Q_proj, K_proj, V_proj, Attention, O_proj, gate_proj, up_proj, down_proj...），每个 task 需要临时占用 SMEM pages 存放输入 tile。如果每个 task 独占 pages，227KB 根本不够用。

### 3.2 核心洞察：tensor 有"生命周期"

```
时间轴 →
Task A: [======A 存活======]
Task B:         [=====B 存活=====]
Task C:                           [=====C 存活=====]
```

A 和 B 的生命周期有重叠 → **不能**共享 page（冲突）。
A 和 C 的生命周期不重叠 → **可以**共享同一个 page！

### 3.3 区间图着色算法

```
算法: Interval Graph Coloring for SMEM Page Allocation

输入: tGraph 中所有 task 的 SMEM 需求 (每个 task 需要的 page 数)
      和 task 间的依赖/调度顺序 (确定每个 tensor 的 live range)

Step 1: 构造区间图 (Interval Graph)
  - 节点 = 每个 SMEM tensor
  - 边 = 两个 tensor 的生命周期有重叠 → 冲突，不能同 page

Step 2: Greedy Coloring (区间图的多项式时间解)
  - 按 live range 开始时间排序
  - 为每个 tensor 分配最小可用颜色 (即最小 page ID)
  - 区间图的色数 = 最大团大小 = 最大同时活跃 tensor 数
  
Step 3: 输出
  - page_assignment[tensor_i] = page_id
  - page_count = 最大着色数 ≤ 7 (H100) / 5 (A100)
```

**为什么区间图着色特别好？**
- 一般图的着色是 NP-hard，但**区间图**的着色可以在 O(n log n) 时间内最优解
- 因为区间图是 chordal graph 的特例，贪心算法按左端点排序即可得到最优着色
- 这意味着编译器可以在**编译期**高效地确定最优 page 分配方案

### 3.4 具体例子

```
Transformer Layer 的 task 调度顺序 (简化):

Time ──────────────────────────────────────────────────►

Q_proj:  [需要 2 pages: Q_weight_tile, Q_output_tile]
K_proj:     [需要 2 pages: K_weight_tile, K_output_tile]  
V_proj:        [需要 2 pages: V_weight_tile, V_output_tile]
Attention:        [需要 3 pages: Q_tile, K_tile, V_tile]
O_proj:                         [需要 2 pages: O_weight, O_output]
Gate_proj:                         [需要 2 pages: ...]
Up_proj:                             [需要 2 pages: ...]
Down_proj:                                     [需要 2 pages: ...]

着色结果 (page 复用):
  Page 0: Q_weight_tile → Attn_Q_tile → O_weight_tile → Gate_weight → Down_weight
  Page 1: Q_output_tile → Attn_K_tile → O_output_tile → Up_weight   → (free)
  Page 2: K_weight_tile → Attn_V_tile → (free)         → Up_output  → ...
  ...
```

**关键效果**：在没有 paging 的情况下，所有 task 的 SMEM 需求之和可能 > 500KB，远超 227KB 限制。但通过 interval graph coloring，7 个 32KB pages 即可满足整个 mega-kernel 的执行。

---

## 四、Cross-Task Pipelining 预取

这是**运行时**利用空闲 pages 提前加载下一 task 数据的机制。

### 4.1 核心思想

```
传统执行 (无流水线):
  Task A:  [TMA加载数据到SMEM] → [计算] → [释放SMEM]
  Task B:                                [TMA加载] → [计算]
  ↑ 加载和计算串行，有 bubble

MPK Cross-Task Pipelining:
  Task A:  [TMA加载] → [计算~~~~~~~~~~~~~~~~~~]
                          ↑ 计算期间 TMA 空闲！
  Task B:               [TMA预取到空闲page] → [计算]
  ↑ Task A 还在计算时，Task B 的 TMA 预取已开始
```

### 4.2 伪代码

```cuda
// Cross-task software pipelining in MPK worker SM

void worker_execute(task* current, task* next) {
    // Phase 1: 当前 task 的 pre-load（如果还没做）
    if (!current->preloaded) {
        acquire_smem_pages(current->needed_pages);
        TMA_async_copy(current->global_src, current->smem_dst);
        wait_TMA_complete();
    }
    
    // Phase 2: 执行当前 task 的计算
    execute_kernel_body(current);  // Tensor Core MMA / CUDA Core
    
    // ★ 关键：在计算期间，同时预取下一个 task 的数据 ★
    // TMA 是异步的，可以和计算完全重叠
    if (has_free_pages() >= next->needed_pages) {
        acquire_smem_pages(next->needed_pages);  // 使用空闲 pages
        TMA_async_copy(next->global_src, next->smem_dst);
        next->preloaded = true;
    }
    
    // Phase 3: 完成
    notify_event(current->output_event);
    release_smem_pages(current->pages);  // 释放当前 task 的 pages
}
```

### 4.3 条件与约束

Cross-task pipelining 的触发需要两个条件同时满足：
1. **当前 task 已发出所有 data-transfer 指令**（TMA 命令已提交）
2. **有足够的空闲 SMEM pages**（通过 interval graph coloring 保证）

实验效果（B200 上 Qwen3-8B 的 final linear layer）：**1.2–1.3× 加速**。

---

## 五、H100 vs A100 的 pages/SM 差异

| | A100 | H100 | B200 |
|---|---|---|---|
| **SMEM/SM** | 164KB (可配 228KB) | 227KB | 227KB |
| **32KB pages** | 5 (164/32) | 7 (227/32) | 7 |
| **TMA 支持** | ❌ 无 | ✅ 有 | ✅ 有 |
| **DSM 支持** | ❌ 无 | ✅ 有 | ✅ 有 |

- **A100 只有 5 pages/SM**：因为 SMEM 更小（164KB vs 227KB），且没有 TMA 硬件支持异步 copy，无法实现 cross-task pipelining 的完整收益
- **H100 有 7 pages/SM**：更大的 SMEM + TMA 异步引擎 → 更多 pages = 更灵活的 page 复用 = 更深的流水线深度
- **B200 也是 7 pages/SM**：SMEM 大小同 H100，但计算能力更强，cross-task pipelining 的加速比更显著（1.2-1.3×）

---

## 六、总结：Memory Planning 的完整图景

```
编译期 (Compiler)                          运行时 (Runtime)
┌──────────────────────┐          ┌──────────────────────────┐
│ tGraph 构造           │          │ SM 分区                   │
│  - 算子→task 分解     │          │  - 128 Workers + 4 Sched  │
│  - task 依赖分析      │          │                           │
│                      │          │ Event-Driven 调度          │
│ Interval Graph       │          │  - scheduler 分发 task     │
│ Coloring             │          │  - worker 执行 + notify    │
│  - 确定 SMEM page    │ ──────► │                           │
│    分配方案           │          │ Cross-Task Pipelining     │
│  - 保证 ≤7 pages     │          │  - 当前计算 + 下一预取     │
│                      │          │  - TMA 异步加载隐藏延迟    │
│ Hybrid JIT/AOT       │          │                           │
│  - JIT: attention    │          │ Paged SMEM 管理            │
│  - AOT: MatMul      │          │  - acquire/release pages   │
│                      │          │  - 编译期方案指导运行时     │
└──────────────────────┘          └──────────────────────────┘
```

**一句话概括**：MPK Paged SMEM 将 GPU shared memory 虚拟化为 32KB 固定 pages，编译期通过**区间图着色**决定 page 复用方案（保证不超过硬件 page 上限），运行时利用**cross-task pipelining** 在空闲 pages 上预取下一 task 数据——从而在单个 mega-kernel 内实现接近硬件理论下限（~10ms/token）的端到端延迟（实际 12.5ms/token）。

### 003 User
为什么"无法做跨 kernel 的流水线：kernel 之间有硬件 barrier，前一个 kernel 全部完成才能开始下一个"?

### 004 Claude
# 为什么传统 kernel-per-operator 无法做跨 kernel 流水线？

你说的"barrier"不是一个显式的 `__syncthreads()` 指令，而是 **GPU 硬件调度模型 + 内存模型** 共同制造的三个刚性边界。

---

## 边界一：SM（Streaming Multiprocessor）不能同时跑多个 kernel 的 thread block

GPU 的硬件调度器（NVIDIA 叫 **GigaThread Engine**）的核心规则：

```
一个 SM 上，同一时刻只能执行来自同一个 kernel 的 thread block。
```

这意味着：

```
        SM 0         SM 1         SM 2         SM 3
Kernel A: [Block 0]  [Block 1]  [Block 2]  [Block 3]     ← 全部 SM 跑 A
         ↓  A 的 Block 0-3 全部结束后 ↓
Kernel B:            [Block 0]  [Block 1]  [Block 2]     ← 才能开始 B
```

**Kernel B 的 thread block 必须等 Kernel A 的所有 thread block 从 SM 上退出后才能被调度。** 这不是 CUDA API 层面的限制，而是 GPU 硬件 thread block 调度器（compute engine 内部的 warp scheduler）的设计决定的——它不维护"跨 kernel 上下文"，每个 kernel 对它来说是一个全新独立的任务。

**有例外吗？** 有，但非常受限：

- **CUDA Streams**：不同 stream 的 kernel 可以**空间上**并发——比如 Kernel A 占 64 个 SM，Kernel B 占剩下 68 个 SM。但这不是流水线（pipelining），因为 B 的输入不是 A 的输出。而且 decode 阶段 batch=1 时只有一个 thread block，根本没法拆分到不同 SM 上。
- **MPS (Multi-Process Service)**：类似，多个进程的 kernel 可以在不同 SM 上并发，但同 SM 上仍不能混合不同 kernel 的 block。
- **CUDA Concurrent Kernels**：从 Kepler 开始支持，但同样是**空间分区**，不是**时间流水线**。

**关键限制**：在 decode batch=1 的场景（LLM 推理的最常见场景），每个 kernel 可能只有 1-4 个 thread block。此时 CUDA Streams 基本无用——没有足够 block 去做空间分区。

---

## 边界二：Shared Memory 的生命周期止于 kernel 结束

```
┌────────── Kernel A ──────────┐  ┌────────── Kernel B ──────────┐
│ SMEM 分配 (kernel launch 时)  │  │ SMEM 分配 (kernel launch 时)  │
│                              │  │                              │
│ Q_proj 写中间结果到 SMEM      │  │                              │
│                              │  │                              │
│ SMEM 释放 (kernel 结束时)     │  │ ⚠️ A 的中间结果已经没了！       │
│                              │  │ B 必须从 HBM 重新读取          │
└──────────────────────────────┘  └──────────────────────────────┘
         ▲                                  ▲
         │─────── 中间数据必须途经 HBM ──────│
```

CUDA 的 shared memory 是 **per-kernel-block 的私有存储**。kernel 结束时，SMEM 被硬件自动回收，下一个 kernel 拿到的是清零/未初始化的 SMEM。

这意味着：
- **Kernel A 的输出不能直接留在 SMEM 里给 Kernel B 用**
- 数据流必须是：A 写 SMEM → A 写 HBM →（A 结束，SMEM 销毁）→ B 从 HBM 读到 SMEM → B 计算

这额外引入了 **两次 HBM 往返**：一次写（A 的输出存回 HBM）、一次读（B 从 HBM 加载输入）。在 HBM 带宽 ~3.35 TB/s 的情况下，这对 memory-bound 的 decode 阶段是致命开销。

---

## 边界三：CPU 在中间"插了一脚"

传统 kernel-per-operator 的执行流程：

```
CPU:  launch(Kernel_A) ──────── 等待完毕 ──────── launch(Kernel_B)
       │                         │                    │
       └─ ~100µs dispatch ───────┘                    └─ ~100µs dispatch
                                │
GPU:  [Kernel A 执行] ── idle ── [Kernel B 执行]
```

每次 kernel launch 需要 CPU 参与：
1. CPU 检查 kernel A 是否完成（cudaStreamSynchronize 或隐式同步）
2. CPU 做 shape inference / memory allocation
3. CPU 设置 kernel B 的 launch parameters
4. CPU 通过 CUDA driver 向 GPU 提交 kernel B

这个 CPU-GPU 往返延迟约 **100µs/kernel**。一个 Transformer decode step 有 ~20-30 个 kernel，累积 ~2-3ms 纯 dispatch overhead——在 12.5ms 的 decode 延迟中占了 ~20%。

**更关键的**：这个 CPU-GPU 同步点等价于一个**隐式的全局 barrier**——CPU 必须看到 kernel A 全部完成才能发起 kernel B。

---

## 对比：MPK 如何打破这三个边界

```
传统 Kernel-per-Operator:
┌────────────┐   ┌────────────┐   ┌────────────┐
│  Kernel A  │   │  Kernel B  │   │  Kernel C  │
│  (Q_proj)  │──▶│  (K_proj)  │──▶│ (Attn)     │
│  SMEM 私有  │   │  SMEM 私有  │   │  SMEM 私有  │
│  经 HBM 传  │   │  经 HBM 传  │   │            │
└────────────┘   └────────────┘   └────────────┘
    边界1: SM 独占         边界2: SMEM 不共享       边界3: CPU 同步

MPK Mega-Kernel:
┌─────────────────────────────────────────────────────────┐
│  单个 Mega-Kernel (永驻 GPU，持久运行)                     │
│                                                         │
│  SMEM Pages:  [Page0] [Page1] [Page2] [Page3] ...       │
│                ↑───────↓───────↑───────↓                 │
│              Q_proj → K_proj → Attention → O_proj → ... │
│                │       │         │                       │
│                └─── 区间图着色复用 pages ───┘              │
│                                                         │
│  Cross-task Pipelining:                                 │
│    Task A compute ═══╦═══                                │
│    Task B preload    ╚═══ TMA 异步预取到空闲 page         │
│                                                         │
│  软件 Scheduler (在 Scheduler SM 上):                    │
│    纯 GPU 端 event-driven，零 CPU 参与                   │
└─────────────────────────────────────────────────────────┘
     ✅ 同 kernel → 同 SMEM 空间 → pages 直接传递
     ✅ 编译期 page 分配 → 区间图着色 → 跨 task 复用
     ✅ 纯 GPU 端调度 → 无 CPU-GPU 往返
```

| 维度 | Kernel-per-Operator | MPK Mega-Kernel |
|------|---------------------|-----------------|
| **SMEM 数据传递** | 必须经 HBM（2 次往返） | task 间直接 page 传递 |
| **调度延迟** | ~100µs/kernel × 30 kernels | 纯 GPU 端，~0 |
| **跨算子流水线** | ❌ 不可能 | ✅ cross-task pipelining |
| **SMEM 复用** | ❌ 每次重新分配 | ✅ 区间图着色复用 pages |
| **与理论下限的距离** | ~50%（大量 HBM 往返） | ~80%（12.5ms vs ~10ms 下限） |

---

## 总结

你问的"为什么 kernel 之间不能做流水线"，本质上不是因为没有一条 CUDA API 叫"barrier"，而是因为：

1. **硬件调度器**把每个 kernel 当独立世界——同 SM 不同时容纳两个 kernel 的 block
2. **Shared memory** 的生命周期与 kernel 绑定——kernel 结束 = SMEM 销毁，中间数据被迫绕道 HBM
3. **CPU 必须参与** kernel 间切换——引入 ~100µs/kernel 的 dispatch 延迟和隐式全局同步点

MPK 的做法是把所有这些算子**物理上合并为一个 kernel**，然后用软件层面的 `event-driven scheduler + paged SMEM + interval graph coloring` 在 kernel 内部重建流水线——本质上是把"kernel 间的流水线"变成了"kernel 内的 task 级流水线"，从而绕过了硬件的三个刚性边界。
