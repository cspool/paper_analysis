## Accelerating Distributed MoE Training and Inference with Lina

- baseline方法是什么？
  **Baseline**: DeepSpeed MoE 的混合并行（Data Parallelism + Expert Parallelism），使用独立的 CUDA streams 分别处理 all-to-all（expert-parallel 通信）和 allreduce（data-parallel 通信），不做跨 stream 协调。Inference 使用 uniform expert-device allocation（每 device 1 个 expert）。

  **Baseline 缺陷**:
  1. **Training 缺陷**: backward pass 中 all-to-all (Stream b) 与 allreduce (Stream c) 并发时公平共享 InfiniBand 带宽，all-to-all 是阻塞式操作（无法与计算并行），被延长 median 1.83x（worst 4.14x）。且 PyTorch DDP gradient bucketing 导致 allreduce 实际 bucket size 变化剧烈，无法预估精确 arrival/running time 做静态调度。
  2. **Inference 缺陷**: 真实推理请求下 expert popularity 高度倾斜（最 popular expert 收到 4.02x~5.56x tokens），uniform allocation 导致 popular expert device 过载，unpopular expert device 空闲（最大 idle time 29.4%）；且 all-to-all 的各 link 使用不均衡，带宽未充分利用。

  **Baseline 全栈执行例子（以 16-expert Transformer-XL Training 一个 MoE layer backward pass 为例）**:
  - **算法层**: top-2 gating, 16 FFN experts → Gate 输出 (token→expert) 映射
  - **系统框架层**: DeepSpeed MoE → 16 GPU (1 expert/GPU) → Data Parallelism (gradient allreduce) + Expert Parallelism (token all-to-all dispatch/combine)
  - **编译框架层**: 论文未明确说明（PyTorch eager execution + NCCL 通信原语）
  - **Kernel/运行时调度层**: Stream a (FFN backward kernel) → 完成后梯度分入 Stream b (all-to-all, 完整大 tensor 一次发射) 和 Stream c (PyTorch DDP gradient bucketing → allreduce, 完整大 tensor 一次发射)，两 stream 并发，NCCL 底层 fair-share 带宽
  - **硬件架构层**: A100 SMs 计算 FFN backward → gradient 经 PCIe/NVSwitch 进入 IB HCA → InfiniBand 传输；GPU SM efficiency 在 all-to-all 期间仅 3.7%（大量空闲等待）

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **论文方法**: Lina = **Training 端**: micro-op priority scheduler (tensor partitioning + priority queue + pipelining) + expert packing; **Inference 端**: expert popularity estimation (token-level expert selection pattern profiling) + two-phase dynamic resource scheduling.

  **Defect→Design 映射**:

  | Baseline 缺陷 | Lina 设计选择 | 解决机制 |
  |---|---|---|
  | all-to-all 与 allreduce 无协调争抢带宽 | Tensor Partitioning (30MB micro-ops) + Priority Queue | all-to-all micro-op 始终优先，allreduce micro-op 仅 idle 时发射 |
  | DDP gradient bucketing → allreduce size 不可预测 | 每个 gradient 独立 partition，不跨 gradient 混合 chunk | 所有 micro-op 大小均匀，调度器可精确控制 |
  | 大 tensor 一次发射阻塞时间长 | Micro-op pipelining: all-to-all 也分区 → 每 micro-op 完成后即启动对应 token 的 FFN | 消除 bubble: FFN time 被 all-to-all 覆盖 |
  | FFN micro-op << all-to-all micro-op 导致 pipeline bubble | Expert Packing: 2^n 递增每 device expert 数 → FFN total time 对齐 all-to-all | Pipeline efficiency: 33%→86% (Transformer-XL) |
  | Inference 中 expert popularity 倾斜且无法提前获知 | Token-level expert selection pattern profiling + sample path estimation | 在 gate 执行前估算 expert popularity 做预调度 |
  | 事后调度（gate 后）阻塞过长 | Two-phase 调度: phase 1 (预调度, 与计算重叠) + phase 2 (少量微调, ~23% cases) | 调度 overhead 从每层 blocking 降为大部分被重叠 |
  | Uniform all-to-all 各 link 负载不均 | Unequal split all-to-all (按实际 token 量 split) | 匹配 popular expert link 高带宽需求 |

  **Lina 全栈执行例子（同 16-expert Transformer-XL Training backward pass 对比 baseline）**:
  - **算法层**: 同 baseline (top-2 gating, 不变模型精度)
  - **系统框架层**: DeepSpeed MoE + Lina Communication Scheduler → 修改 PyTorch DDP bucketing → gradient 不 fuse 而是独立 partition 为 30MB micro-ops
  - **编译框架层**: 论文未明确说明
  - **Kernel/运行时调度层**: 
    1. FFN backward 完成后 gradient tensor 入 priority queue
    2. Scheduler: `chunk(grad, 30MB)` → 5 micro-ops
    3. 若队列有 all-to-all micro-op → launch NCCL all-to-all → 等待完成
    4. 若队列无 all-to-all → launch allreduce micro-op（但 combine computation 阶段停止发射）
    5. All-to-all micro-op 1 完成 → 对应 tokens 进入 FFN → 覆盖计算延迟
    6. Expert Packing: 2 experts/device → FFN total time 增长至接近 all-to-all micro-op → pipeline efficiency 86%
  - **硬件架构层**: A100 SMs 在 all-to-all 期间不再全 idle（pipelining 使 FFN 计算重叠）→ GPU utilization +17.6%；all-to-all 获满 100Gbps IB 带宽 → all-to-all time speedup 2.21x

  **Lina Inference 全栈对比**:
  - Baseline: Gate → [Uniform All-to-All Dispatch] → Expert Compute (popular expert 过载) → [All-to-All Combine] → 尾部延迟拉长
  - Lina: Profile path patterns → Phase 1: 估算 popularity (layers 1-3 warm-up) → Phase 1 piggback on all-to-all → Scheduler compute mapping (popular expert → multi-device replica) → Expert swap → [Unequal All-to-All] → Balanced Expert Compute → Phase 2: check accuracy (~23% need re-schedule) → [Unequal All-to-All Combine] → median inference 1.45x faster, tail 95%ile 1.63x faster
