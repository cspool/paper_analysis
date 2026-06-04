# ACS 乱序 Kernel 调度 对话记录

- 导出时间：2026-06-04
- 来源：当前 Claude Code session 可用对话上下文
- 保存路径：human_notes/ACS_乱序Kernel调度_对话记录.md
- 范围：仅用户输入与 Claude 最终输出

## 对话记录

### 001 User
[IDE selection: draft/review_draft.md line 55 — ACS Out-of-Order Kernel Dispatch 表格行]
教我

### 002 Claude
# ACS 乱序 Kernel 调度 — 详解

## 一句话核心思想

**让 GPU 像 CPU 一样"乱序执行"**——在一个固定大小的调度窗口内，对已发射但尚未执行的 kernel 进行运行时依赖检测，一旦某个 kernel 的所有上游依赖都已完成，就立即发射它，而不必等待排在它前面的所有 kernel 执行完毕。

---

## 1. 背景：为什么需要这个？

### 问题场景：Input-Dependent Irregular DAG

某些应用（如强化学习仿真 Brax、动态神经网络 InstaNAS-A）的 kernel 计算图在**编译时无法确定**——每次输入不同，kernel 之间的依赖关系就不同。这类图有两个致命特征：

1. **Input-dependent**：计算图随输入变化，必须每次运行时重新确定依赖
2. **Irregular**：依赖关系不规则，难以简单切分成几个独立的 CUDA stream

### 传统方案为什么不行？

| 方案 | 为什么失败 |
|------|-----------|
| **单 Stream 串行** | 小 kernel 多→GPU 利用率仅 ~34%（RTX 3060, 28 SM） |
| **多 Stream + Event 同步** | 依赖太复杂，需要大量 `cudaStreamWaitEvent`，CPU-GPU 同步开销 5-20μs/kernel |
| **CUDA Graph** | 每次输入都要重建 DAG→构建时间占执行时间 47%（Brax 实测，见下图） |
| **Persistent Thread** | 只适用于同构 kernel |
| **Dynamic Parallelism** | 只支持父子依赖，不支持多对多依赖 |

### GPU Kernel Launch 的开销构成

```
每次 CUDA kernel launch ≈ 5-10μs (PCIe 往返 + Command Processor 处理)
但很多小 kernel（如 elementwise add/activation）本身执行仅 2μs
→ scheduling overhead >> 执行时间 → GPU 大量空闲
```

---

## 2. ACS 核心机制

### 三步走：

```
应用发射 kernel（标注 read/write segments）
        ↓
  ① 输入 FIFO 队列（排队等待进入调度窗口）
        ↓
  ② 固定大小 Scheduling Window (N=32~64 slots)
     - 新 kernel 进入窗口时做依赖检测
     - 标记 upstream 依赖列表
     - upstream 为空 → READY
        ↓
  ③ 调度器发射 READY kernel 到独立 CUDA stream 并发执行
```

### 依赖检测算法

核心是**地址范围重叠检测**，复杂度 O(segments²)：

```python
# 伪代码
def check_dependency(new_kernel, window_kernels):
    upstream = []
    for k in window_kernels:
        # 新 kernel 的 write segment 与窗口中 kernel 的 read/write segment 有重叠？
        for wseg in new_kernel.write_segments:
            for rwseg in (k.read_segments + k.write_segments):
                if overlap(wseg, rwseg):  # 地址范围相交
                    upstream.append(k.id)
    return upstream  # 这些 kernel 必须先完成
```

关键前提：程序员通过 `ACS_wrapper` 为每个 kernel 标注 `__read_segments__` 和 `__write_segments__`（虚拟地址起始 + 大小列表），在 kernel launch 前由 `get_addresses()` 解析。

---

## 3. 两种实现：ACS-SW vs ACS-HW

```
                    ACS-SW                      ACS-HW
调度窗口位置        CPU 内存（软件数据结构）      GPU Command Processor 内 SRAM
窗口大小            可配置                        N=32 → ~1KB SRAM
dispatch overhead   ~5-10μs (含 StreamSync)      ~50-100ns (64 cycles, 纯硬件)
加速比              1.87×                        2.19×
硬件改动            无                           GPU CP 内加调度窗口 + Upstream Load Module
同步方式            cudaStreamWaitEvent          硬件自动更新 upstream list
```

### ACS-SW 架构

```
┌─────────────────────────────────────────────┐
│                    CPU                       │
│  ┌──────────────┐    ┌───────────────────┐  │
│  │ Window Module │◄──►│ Scheduling Window │  │
│  │  (独立线程)    │    │   (共享内存)      │  │
│  └──────┬───────┘    └───────┬───────────┘  │
│         │                    │               │
│  ┌──────┴──────────────────────┐            │
│  │  Scheduler Module           │            │
│  │  多线程，每线程绑定1个       │            │
│  │  CUDA Stream                │            │
│  │  - poll READY kernel        │            │
│  │  - cudaLaunchKernel         │            │
│  │  - cudaStreamSynchronize    │            │
│  └─────────────────────────────┘            │
└─────────────────────────────────────────────┘
         │  │  │  (每个线程一个 stream)
    ┌────┘  │  └────┐
    ▼       ▼       ▼
  ┌─────────────────────┐
  │       GPU           │
  │  Stream0 Stream1 .. │
  └─────────────────────┘
```

瓶颈：StreamSync 仍需 CPU↔GPU 同步，每个 kernel 完成后 CPU 线程才被唤醒。

### ACS-HW 架构

```
┌────────── CPU ──────────┐      ┌────── GPU ──────────────────┐
│  Input FIFO             │      │                              │
│  (CUDA Stream)          │─────►│  Command Processor            │
│                          │      │  ┌────────────────────────┐ │
│  Scheduled List (可能    │      │  │ Scheduling Window SRAM  │ │
│  stale，容错)            │      │  │ N=32 slots, ~1KB        │ │
│                          │      │  │ ┌────┬────┬────┬────┐  │ │
│  预计算 upstream list    │      │  │ │KID│U1..│UN-1│STAT│  │ │
│  (可能与实际有偏差)       │      │  │ └────┴────┴────┴────┘  │ │
│                          │      │  │        ...×32          │ │
│                          │      │  └──────────┬─────────────┘ │
│                          │      │             │               │
│                          │      │  ┌──────────▼─────────────┐ │
│                          │      │  │ Upstream Load Module   │ │
│                          │      │  │ 修正CPU端stale list:   │ │
│                          │      │  │ (1)去除已完成kernel    │ │
│                          │      │  │ (2)保证不遗漏执行中    │ │
│                          │      │  └──────────┬─────────────┘ │
│                          │      │             │               │
│                          │      │             ▼               │
│                          │      │    Kernel Dispatch Unit     │
│                          │      │    → 发射 READY kernel      │
│                          │      └─────────────────────────────┘
└──────────────────────────┘      └──────────────────────────────┘
```

**硬件调度窗口**的每个 slot 结构：
- **8-bit kernel ID**：标识 kernel
- **(N-1) × 8-bit upstream kernel ID**：全关联存储，记录该 kernel 依赖的所有上游 kernel
- **2-bit 状态**：READY / PENDING / EXECUTING

**Upstream Load Module** 解决 CPU 端 `scheduled_list` 可能 stale 的问题：
1. 移除已完成的 kernel（CPU 不知道 GPU 内已有 kernel 完成）
2. 通过 oldest-scheduled-kernel 追踪，保证不会遗漏仍在执行的 kernel

**硬件开销**：N=32 仅需 ~1KB SRAM（整颗 GPU 级别），可以忽略。

---

## 4. 关键设计决策

### RW-Segments 标注

每个 kernel 必须声明它读/写哪些 GPU 内存区域。对于常规 kernel（矩阵乘、卷积、elementwise），数据存储在连续区域，标注很简单：

```
MatMul: __read_segments__  = {input1[m×n], input2[n×k]}
        __write_segments__ = {output[m×k]}
```

对于**间接内存访问**（indirect access）无法确定范围时，保守地标记为"可能访问整个 GPU 内存"→该 kernel 与窗口中所有 kernel 都有依赖→退化为串行（保证正确性优先）。

### CPU 依赖检查延迟不是瓶颈

```
依赖检查延迟: 410ns ~ 1640ns  (纯计算，数据在 L1 cache 中)
vs
Kernel launch 延迟: 5-20μs

瓶颈在 launch/sync，不在依赖分析！
所以 CPU 端做依赖检查完全可行
```

### Stale Scheduled List 的容错设计

ACS-HW 中 CPU 维护的 `scheduled_list` 允许与 GPU 实际状态不同步——Upstream Load Module 在硬件侧修正。这避免了频繁的 CPU↔GPU 同步。

---

## 5. 适用场景

| 适合 | 不适合 |
|------|--------|
| Input-dependent irregular DAG（动态图） | 静态 DAG（CUDA Graph 更好） |
| 大量小 kernel（执行时间 < launch 时间） | 少量大 kernel（已充分利用 SM） |
| 单请求内大量细粒度算子（如 RL 仿真每步） | 批量推理（已按 batch 聚合） |
| 无法提前知道依赖关系的场景 | kernel 间无并行度 |

**实测数据**：Deep RL Brax Ant 环境，RTX 3060 (28 SM)，GPU occupancy 从 ~34%→近满载，加速比 ACS-HW 2.19× / ACS-SW 1.87×。

---

## 6. 与其他调度方法的系统对比

```
                      Applicability  Sync+Launch  Preparation
                      (不规则动态图)  Overhead     Overhead
Multi-Stream          ✓              ✗            ✓
CUDA Graph / DAG      ✓              ✓            ✗ (构建DAG 47%总时间)
Persistent Thread     ✗ (只同构)      ✓            ✓
Dynamic Parallelism   ✗ (只父子依赖)  ✗            ✓
─────────────────────────────────────────────────────────
ACS-SW                ✓              ✗            ✓
ACS-HW                ✓              ✓            ✓ (≈最优)
```

**ACS-HW 是唯一同时满足三个维度的方案**：适用于不规则动态图 + 消除同步/发射开销 + 准备开销低。

---

## 7. 局限与开放问题

1. **RW-segments 标注依赖程序员/编译器**：间接内存访问场景精度有限
2. **窗口大小限制并行度**：N=32~64，超出的 kernel 只能在 FIFO 中等待
3. **ACS-HW 需要硬件改动**：目前没有商业 GPU 实现，仅论文方案
4. **论文代码未公开**：论文声称开源 ACS-SW，但当前无公开仓库
5. **不适合大 kernel**：如果每个 kernel 已充分利用 SM（occupancy 高），乱序调度无额外收益

---

## 一句话总结

> ACS 把 CPU 乱序执行（Tomasulo 算法的思想）搬到了 GPU kernel 调度层——用固定大小的调度窗口 + 运行时内存地址重叠检测，替代了 DAG 全局静态构建，使得 input-dependent irregular 计算图也能高效并发执行。ACS-HW 进一步把调度窗口硬化到 GPU Command Processor 内，将 dispatch overhead 从微秒级降到纳秒级。

### 003 User
/swanlab

### 004 Claude
# SwanLab — 开源实验追踪工具

SwanLab 是一个开源的 ML 实验追踪平台，支持本地部署、自托管和云端使用。核心能力：

## 快速上手

```python
import swanlab

run = swanlab.init(
    project="my-project",
    experiment_name="baseline",
    config={"learning_rate": 1e-3, "epochs": 10, "batch_size": 32},
)

for epoch in range(run.config.epochs):
    swanlab.log({"train/loss": train_loss, "val/loss": val_loss})
run.finish()
```

## 主要功能

| 功能 | 说明 |
|------|------|
| **指标追踪** | `swanlab.log()` 记录 scalar、自动分 namespace（`train/loss`） |
| **配置管理** | `config` 参数跟踪超参、模型名、seed 等 |
| **媒体日志** | Image、Audio、Text、GIF/Video、PointCloud、Molecule |
| **本地模式** | `mode="local"` 离线记录，`swanlab watch -l` 本地查看 |
| **框架集成** | Transformers、PyTorch Lightning、Fastai 原生 callback |

## 安装

```bash
pip install "swanlab>=0.7.11" "pillow>=9.0.0" "soundfile>=0.12.0"
pip install "swanlab[dashboard]>=0.7.11"  # 本地 dashboard
```
