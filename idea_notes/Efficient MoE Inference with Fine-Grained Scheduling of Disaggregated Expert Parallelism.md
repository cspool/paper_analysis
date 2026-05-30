## Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism

- baseline方法是什么？
  Baseline 是 MegaScale-Infer [36] 中的 Ping-Pong Pipeline Parallelism (PPPipe) 算法，该算法是 DEP 架构下最先进的任务调度方法。

  **Baseline 全栈执行例子（以 DeepSeek-V2、ag 个 AG GPU + eg 个 EG GPU 为例）**：
  - **算法层**：MoE 层包含 Multi-Head Latent Attention (MLA) + Shared Expert + Routed Experts。每个 token 经 router 选择 top-k experts。Shared Expert 对所有 token 计算。
  - **系统框架层**：DEP 将 GPUs 分为 Attention Group (AG) 和 Expert Group (EG)。AG 存储所有 Attention 层参数和 Shared Expert 参数（全复制到 ag 个 GPU），EG 存储 E 个 sparse experts（分布在 eg 个 GPU，每个 GPU 持有 E/eg 个专家）。PPPipe 将 mini-batch 切分为 r1 个 micro-batch，使 AG 和 EG 可并行执行。A2E 和 E2A 通信通过 NCCL 实现，无 group 内通信（AG 内参数全复制、EG 内 token 按 expert 路由不跨设备）。
  - **编译框架层**：论文未明确说明（标准 PyTorch + CUDA + NCCL 编译路径）
  - **Kernel 调度层**：PPPipe 以 micro-batch 为粒度进行流水线调度。在 AG 端，Attention + Shared Expert 在每个 micro-batch 内顺序执行，完成后启动 A2E 通信。在 EG 端，每个 expert 对其分配的 token 执行 GEMM（Gate/Up/Down 三层投影），每个 micro-batch 的 expert 计算和 A2E/E2A 通信串行执行（仅 micro-batch 间有重叠）。A2E 与 Shared Expert 不可并行（PPPipe 将 Shared Expert 视为 Attention 的一部分）。
  - **硬件架构层**：四种 GPU 平台（8×A6000 48GB Ampere NVLink、8×A10 24GB Ampere PCIe only、8×H20 96GB Hopper NVLink、32×H20 四节点 NVLink）。计算在 Tensor Cores，通信通过 NVLink/PCIe。

  **Baseline 核心痛点**：
  1. Shared Expert 计算调度不当：PPPipe 假设无 Shared Expert，将 Shared Expert 视为 Attention 的一部分串行执行，但实际 Shared Expert 与 A2E 通信和 routed expert 计算之间无数据依赖，可以并行，造成 GPU 空闲。
  2. Micro-batch 级别流水线不足以完全隐藏通信：PPPipe 仅做 coarse-grained micro-batch 重叠，未能进一步将 A2E/E2A 通信与 expert 计算重叠。粗粒度下，一个 micro-batch 内的通信仍占用较长时间，导致另一端 GPU 等待。
  3. 解空间巨大难以找到最优调度：引入 shared expert 支持和细粒度流水线后，r1/ma/r2/me 的组合搜索空间爆炸，Brute-force 枚举不可行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  FinDEP 提出三方面创新：

  **1) 细粒度任务切分与流水线（解决痛点 1 和 2）**：
  将 AG 端任务沿 batch 维度切分为 r1 个 pipeline 段（每 GPU 处理 ma 个样本），将 EG 端任务沿 token 维度进一步切分为 r2 个 fine-grained 段（每个 expert 处理 me 个 token）。通过 r1 和 r2 两级流水线实现：AG 内 Attention 与 Shared Expert 交替执行（ASAS 策略），A2E 通信与 Shared Expert 计算并行，A2E/E2A 通信与 EG 内的 expert 计算重叠。

  **2) 形式化优化问题（解决痛点 3）**：
  建立线性性能模型（α-β 模型）预测 GEMM/Attention/A2E/E2A 的执行时间，将 DEP 推理时间形式化为包含 r1、ma、r2、me 和任务顺序的优化问题（目标函数 Eq.13），证明目标函数关于 ma 单调递增（Theorem 1/2）、关于 r1 单调非递减（Theorem 3）、关于 1/r2 是凸函数（Theorem 4）。

  **3) 高效近似最优求解算法（解决痛点 3）**：
  Algorithm 1 基于单调性和凸性约束搜索空间：(a) 利用 ma 和 r1 的单调性，只在 Pareto 前沿上搜索 (ma, r1) 组合；(b) 对每个组合，固定 r1/ma 后对 1/r2 做凸优化快速收敛；(c) 同时评估 ASAS 和 AASS 两种执行顺序。算法复杂度 O(C·√M)，实际求解 < 1s，使在线自适应成为可能。

  **论文方法全栈执行例子（以 DeepSeek-V2 为例，与 Baseline 对比）**：
  - **算法层**：同 baseline，MoE gating/routing 不变
  - **系统框架层**：DEP 架构不变，但在 AG 端增加两种执行顺序选择：(a) AASS（先全部 Attention、再全部 Shared Expert）——使 A2E 最早启动，利于 EG 尽早计算；(b) ASAS（Attention 与 Shared Expert 交替执行）——提高 AG 内 GPU 利用率。通过 Algorithm 1 在线选择最优顺序。A2E/E2A 通信从每个 micro-batch 一次变为每个 fine-grained 段一次（r2 倍频率但每次数据量减少），与 expert 计算重叠。
  - **编译框架层**：论文未明确说明（标准 PyTorch + CUDA 路径）
  - **Kernel 调度层**：
    AG 端：ma 个 sample 的 Attention → ma 个 sample 的 Shared Expert → me 个 token 的 A2E 通信，pipeline 重复 r1 次。ASAS 策略下 Shared Expert 与下一 micro-batch 的 Attention（或无数据依赖的 A2E）交替执行。
    EG 端：r2 段 fine-grained pipeline，每段处理 me 个 token。A2E 收到 me 个 token 后立即启动 expert 的 GEMM 计算，同时下一段 me 个 token 的 A2E 通信可并行进行。关键调度参数：X(ma)=ta+ts, Y(me)=max(te, ta2e), F(ma,me)=max(X, r2·Y) 控制各段之间的时序约束。
  - **硬件架构层**：同 baseline 的 4 种 GPU 平台。效果：在 8×A6000 DeepSeek-V2 S=4096 下，非重叠通信时间从 PPPipe 的 528.94ms 降至 309.81ms（1.7× 减少）。在 32×H20 Qwen3-MoE S=4096 下吞吐提升 1.24×。
