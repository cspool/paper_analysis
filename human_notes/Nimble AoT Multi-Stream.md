# Nimble AoT Multi-Stream

- 导出时间：2026-06-03 17:00 CST
- 来源：当前 Codex session 可用对话上下文
- 保存路径：human_notes/当前session完整对话记录.md
- 完整性说明：完整导出当前可用上下文；早期大型工具输出在当前上下文中已经以截断/摘要形式呈现，因此按当前可见内容记录。

## 完整对话记录

### 001 User

```text
# Context from my IDE setup:

## Open tabs:
- L1_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L1_horizon_summary.md

## My request for Codex:
教我:'多算子并发与调度	Nimble AoT Multi-Stream	AoT CUDA Graph capture + MEG + Ford-Fulkerson 最大匹配 → 多 stream 并发	vs PyTorch up to 22.34×; max concurrency=15'
```

### 002 Codex

最终回答：

```md
可以，把这行当成一句“论文方法卡片”来读：

> **Nimble AoT Multi-Stream**：先离线/AoT 把 PyTorch 模型的 GPU 执行录成 CUDA Graph，再用 MEG + Ford-Fulkerson 最大匹配找出 DAG 中可并发的算子，把它们分到多个 CUDA stream 上并发执行。结果是在特定模型上相对 PyTorch 最高 22.34×，NASNet-A 的最大逻辑并发度为 15。

**一句话理解**
Nimble 解决的是：PyTorch eager mode 每个算子都要 CPU 调度一次，而且默认很多算子在一个 stream 里串行跑。Nimble 把“每次运行都调度”改成“提前调度一次，之后直接重放”，并把模型 DAG 里真正独立的分支放到多个 stream 并行跑。

**核心链路**
```text
PyTorch model
  -> TorchScript trace 得到算子 DAG
  -> MEG 去掉冗余传递依赖
  -> 构建二分图
  -> Ford-Fulkerson 最大匹配
  -> stream assignment
  -> CUDA Graph capture
  -> runtime cudaGraphLaunch replay
```

**为什么 AoT CUDA Graph 有用**
PyTorch eager 每个 operator 都会经历：

```text
dispatch -> shape inference -> kernel selection -> args preparation -> kernel launch
```

如果模型有很多小 kernel，比如 NASNet-A mobile，一个 kernel 可能 GPU 只跑十几微秒，但 CPU 调度开销可能更大，GPU 会频繁等 CPU。Nimble 在首次准备阶段把这些都做完，并录成 CUDA Graph；之后每次输入只需要一次 `cudaGraphLaunch`，GPU 按预录制的图自己执行。

可以粗略记成：

$$
T_{\text{PyTorch}} \approx \sum_i (T_{\text{cpu\_sched},i} + T_{\text{gpu},i})
$$

$$
T_{\text{Nimble}} \approx T_{\text{graph\_launch}} + T_{\text{multi-stream critical path}}
$$

Annotations：第一个式子里 CPU 每算子调度会重复发生；第二个式子里调度开销被搬到 AoT preparation，运行时主要剩 graph launch 和 GPU 关键路径。

**MEG + 最大匹配在干什么**
原始 DAG 里可能有冗余边，例如：

```text
A -> B -> C
A ------> C
```

`A -> C` 其实是传递依赖，因为 A 已经通过 B 影响 C。MEG 会删掉这种冗余边，只保留真正必要的直接依赖。这样后续算法不会被假依赖误导。

然后 Nimble 把 MEG 转成二分图，用 Ford-Fulkerson 求最大匹配。直觉上，最大匹配帮助它找到“哪些边/算子链可以安全地并行组织”，最终生成 stream 分配，并在真正跨 stream 依赖处插入 CUDA event。

**多 stream 并发长这样**
```text
stream 0: branch_A_op1 -> branch_A_op2 ----\
stream 1: branch_B_op1 -> branch_B_op2 -----+-> concat -> next ops
stream 2: branch_C_op1 -> branch_C_op2 ----/
                         event sync here
```

Annotations：同一个 stream 内仍然 FIFO 串行；不同 stream 的 kernel 可以并发，但前提是 GPU 资源允许同时驻留。`concat` 这种汇聚点必须等所有分支完成。

**怎么读性能数字**
`vs PyTorch up to 22.34×`：这是最高值，不是所有模型都这样。它通常出现在小 kernel 很多、CPU 调度开销占比很高、DAG 又有分支并发机会的模型上。

`max concurrency=15`：表示 NASNet-A 中最多有 15 个逻辑上可并发的算子，不等于任意时刻一定物理并发 15 个 kernel。真实并发还受 SM、寄存器、shared memory、block 数量等资源限制。

**适用场景**
适合：静态模型、shape 固定、分支多、小 kernel 多、batch size 小的推理或训练。

不适合：动态 control flow、shape 经常变化、主要由大 GEMM/大 Conv 组成的模型。大 kernel 本身已经能吃满 GPU，Nimble 的调度收益就会小很多。

本地依据主要来自：`Ahead-of-Time GPU Scheduling`、`CUDA Graph`、`Stream Assignment Algorithm`、`Nimble` 实验笔记。
```
