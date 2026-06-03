# Ascend 910B NPU 硬件体系结构详解

- 导出时间：2026-06-03 (Asia/Shanghai)
- 来源：当前 Claude Code session 可用对话上下文（MoEDiT 多模态 Video 学习实验 L1 Horizon Summary 中的 Q1.3 硬件体系结构）
- 保存路径：`human_notes/Ascend_910B_NPU_硬件体系结构详解.md`
- 范围：仅用户输入与 Claude 最终输出

## 对话记录

### 001 User
（从 `L1_horizon_summary.md` 选中第 133 行）

```
**硬件体系结构** | NPU SA (Ascend 910B) | Da Vinci Core × N; Cube Unit (矩阵乘) + Vector Unit (激活) + Scalar Unit (控制流) + MTE (通信卸载); 64GB HBM ~1.2 TB/s | 需软件显式管理 L1 Buffer; MTE CoC 通信-计算重叠 | Q1.3 |
```

教我

### 002 Claude

（以下是 Claude 对 Ascend 910B NPU 硬件体系结构的完整讲解）

---

## 1. 什么是 Ascend 910B？

Ascend 910B 是华为设计的 **NPU (Neural Processing Unit)**，属于 **SA (Systolic Array，脉动阵列)** 架构的 AI 加速器。它和 GPU 一样做 AI 推理/训练，但硬件设计哲学完全不同。

对比 NVIDIA H100：

| 维度 | H100 (GPU SIMT) | Ascend 910B (NPU SA) |
|------|-----------------|---------------------|
| 计算范式 | Warp-SIMT + Tensor Core | 32×32 Systolic Array |
| 显存 | 80GB HBM3, 3.35 TB/s | 64GB HBM, ~1.2 TB/s |
| FP16 算力 | 989 TFLOPS | ~256 TFLOPS |
| INT8 算力 | 1979 TOPS (FP8) | ~200 TOPS (INT8) |
| 制程 | TSMC 4nm | 7nm |
| L1 管理 | 硬件自动调度 | **软件显式管理** |

---

## 2. Da Vinci Core 内部四大单元

每个 Da Vinci Core 包含四个功能单元，分工明确：

```
┌─────────────────────────────────────────────────┐
│              Da Vinci Core × N                    │
│                                                   │
│  ┌──────────────┐  ┌──────────────┐              │
│  │  Cube Unit   │  │ Vector Unit  │              │
│  │  (矩阵乘法)   │  │  (激活函数)   │              │
│  │  32×32 SA    │  │  逐元素操作   │              │
│  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                       │
│  ┌──────┴─────────────────┴───────┐              │
│  │        Unified Buffer          │              │
│  │    (Vector/Scalar 共享)         │              │
│  └────────────────────────────────┘              │
│         │                                         │
│  ┌──────┴───────┐  ┌──────────────┐              │
│  │ Scalar Unit  │  │     MTE      │              │
│  │  (控制流)     │  │ (通信卸载)    │              │
│  │  分支/循环    │  │  异步DMA搬运  │              │
│  └──────────────┘  └──────────────┘              │
│                                                   │
│  L1 Buffer (Cube Unit 专用)                       │
└─────────────────────────────────────────────────┘
```

### 2.1 Cube Unit — 矩阵乘法引擎

- **本质**：32×32 的脉动阵列 (Systolic Array)，类似 GPU 的 Tensor Core
- **做什么**：执行 `D = A × B + C` 这类矩阵乘累加操作
- **为什么用脉动阵列**：数据在 PE 之间"脉动式"流动，每个 cycle 相邻 PE 传递数据，减少寄存器读写功耗。相比 GPU Tensor Core 的 Warp-SIMT 方式，SA 在规整的大矩阵乘上能效更高
- **对应 MoE/DiT 的哪些操作**：Expert FFN 中的 `gate_proj`, `up_proj`, `down_proj` 三个 GEMM；Attention 中的 `Q/K/V/O` 投影

### 2.2 Vector Unit — 逐元素操作

- **做什么**：激活函数 (SiLU, GELU, ReLU)、LayerNorm/RMSNorm、Softmax、逐元素乘加
- **为什么不能全用 Cube Unit**：Cube Unit 只能做矩阵乘。激活函数是逐元素的，不需要矩阵乘法器，用 Vector Unit 更高效
- **一个典型的执行交错**（MoE Expert FFN）：
  ```
  Cube:   gate = tokens @ W_gate   ← 矩阵乘
  Vector: gate = SiLU(gate)         ← 逐元素激活
  Cube:   up   = tokens @ W_up     ← 矩阵乘
  Vector: hidden = gate * up        ← 逐元素乘
  Cube:   down = hidden @ W_down   ← 矩阵乘
  ```
  Cube → Vector → Cube 交替执行，两个单元可流水线化

### 2.3 Scalar Unit — 控制流

- **做什么**：分支判断、循环控制、地址计算、标量运算
- **为什么需要**：MoE 的 Router（Top-K 选择）涉及 `argsort`、条件分支，这些不适合 Cube/Vector 做。Scalar Unit 处理控制流逻辑，调度 Cube/Vector 的执行
- **对标 GPU**：类似 GPU 的 Warp Scheduler + dispatch unit，但 GPU 是硬件自动调度，NPU 需要软件（编译器）配合

### 2.4 MTE (Memory Transfer Engine) — 通信卸载引擎

这是 Ascend 910B 最关键的差异化设计：

- **本质**：一个独立的 DMA 引擎，专门负责**数据搬运**
- **做什么**：
  - HBM ↔ L1 Buffer 的数据传输
  - NPU 间 All-to-All 通信（MoE 的 token dispatch/combine）
  - 远程内存访问（类似 GPU 的 NVSHMEM RDMA）
- **核心优势 — Communication over Computation (CoC)**：
  ```
  时间轴 ──────────────────────────────────────►
  Cube Unit:  [ Batch N MatMul .................]
  MTE:        ......[ Batch N+1 All-to-All 通信 ]
                       ↑
                   MTE 与 Cube 并行执行！
                   当前 batch 计算不等待通信完成
  ```
  - GPU 上 All-to-All 通常要占用 SM 资源或用 Copy Engine，和计算**争抢**内存带宽
  - Ascend 上 MTE 是**独立硬件单元**，有自己的 DMA 通道，通信和计算真正物理并行
  - MoE 训练在 32N/64N/256N Ascend 集群上效率提升 **5.4-46.6%**

---

## 3. 存储层次与"软件显式管理 L1 Buffer"

```
HBM (64GB, ~1.2 TB/s)          ← 片外大容量存储
        │
        ▼ MTE 搬运
L1 Buffer (Cube 专用)           ← 矩阵乘输入/输出缓存
Unified Buffer (Vector/Scalar)  ← 向量/标量操作共享缓存
```

### 为什么 GPU 不需要"显式管理"而 NPU 需要？

**GPU (H100)**：
- 每个 SM 有 227KB Shared Memory (L1/SMEM)
- **硬件 Warp Scheduler** 自动调度 thread block 到 SM，L1 由硬件自动管理
- 程序员只需要在 kernel 里 `__shared__ float tile[128][128]` 声明 SMEM，硬件处理替换/淘汰

**Ascend 910B NPU**：
- L1 Buffer 是 Cube Unit 的**专用**缓存，不像 GPU L1 是通用缓存
- **没有硬件自动替换策略**——编译器/CANN 框架必须在编译时规划好：
  - 哪些 tile 放在 L1 Buffer？（分块策略）
  - L1 Buffer 如何分区给不同算子？（多算子并发时的空间分配）
  - 何时从 HBM 预取下一批数据到 L1？（双缓冲/Double Buffering）
- 这就是 **MikPoly** 编译器做的事——离线生成最优 micro-kernel tile 配置 + 运行时 <1ms 聚合选择

**类比**：GPU 的 L1 像"自动挡汽车"（硬件自动换挡），Ascend NPU 的 L1 像"手动挡赛车"（软件精确控制每一档位）——后者理论上可以压榨出更高效率，但编程复杂度更高。

---

## 4. 多算子并发在 Ascend 910B 上的实现

Ascend NPU 的多算子并发通过**多级 Buffer 分区**实现：

```
┌────────────────────────────────────────────────┐
│ L1 Buffer 分区                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ 算子 A   │ │ 算子 B   │ │ 算子 C   │       │
│  │ GEMM     │ │ SiLU     │ │ 通信预取  │       │
│  │(Cube)    │ │(Vector)  │ │(MTE)     │       │
│  └──────────┘ └──────────┘ └──────────┘       │
│       ↑            ↑            ↑              │
│       └────────────┼────────────┘              │
│                    │                            │
│   编译器（CANN/MikPoly）静态分配每个 buffer 分区 │
└────────────────────────────────────────────────┘
```

与 GPU 的对比：

| 并发机制 | H100 GPU | Ascend 910B NPU |
|----------|----------|-----------------|
| 计算单元并发 | CUDA Streams + MPS + MIG | L1/UB 多分区 + Cube/Vector 交替 |
| 通信-计算重叠 | Multi-stream + NVSHMEM + TB specialization | **MTE 专用硬件** (CoC) |
| 算子融合 | 编译器 fusion (SMEM 227KB 限制) | CANN graph fusion + TBE 自定义 |
| 调度方式 | 硬件 Warp Scheduler 动态 | 编译器静态分配 + 硬件执行 |

---

## 5. 一句话总结

**Ascend 910B NPU = 脉动阵列做矩阵乘 (Cube) + 向量单元做逐元素操作 (Vector) + 标量单元做控制流 (Scalar) + 独立 DMA 引擎做通信卸载 (MTE)**。核心设计哲学是用**专用硬件单元 + 软件精确管理**替代 GPU 的**通用硬件 + 自动调度**——在特定负载（如 MoE 大矩阵乘）上能效更优，但编程和生态成熟度仍有差距。
