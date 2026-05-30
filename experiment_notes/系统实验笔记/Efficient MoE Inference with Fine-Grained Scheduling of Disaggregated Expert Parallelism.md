## Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 FinDEP —— 一个针对 Disaggregated Expert Parallelism (DEP) 的细粒度任务调度框架，用于优化 MoE 模型推理吞吐。包含三个关键创新：(1) 将 AG 和 EG 中的计算和通信任务沿 batch 维度和 token 维度分别切分为更小的子任务（AG 端沿 batch 维度切分为 r1 个 micro-batch pipeline，EG 端沿 token 维度切分为 r2 个 fine-grained pipeline），以实现细粒度任务流水线和最大程度的重叠；(2) 建立包含计算和通信开销的端到端性能模型，形式化一个优化问题来表征 DEP 推理时间，涵盖任务顺序、tensor 切分粒度 r1/r2 和 micro-batch size ma/me；(3) 开发一个多项式时间复杂度的算法（Algorithm 1）在巨大解空间中搜索近似最优调度配置。实验比较 FinDEP 与 PPPipe (MegaScale-Infer 中的 Ping-Pong Pipeline)，评估在四种 GPU 平台（8×A6000、8×A10、8×H20、32×H20）上使用 DeepSeek-V2（有 shared experts）和 Qwen3-MoE（无 shared experts）两种 backbone 的推理吞吐（tokens/s），以及非重叠通信时间的减少效果。同时评估在线场景下 FinDEP 快速 solver（<1s）的自适应能力。

- 硬件平台是什么，配置是什么。
  四个硬件 Testbed：
  - Testbed A: 单节点 8×NVIDIA RTX A6000 (48GB, Ampere, Boost 1.46GHz, NVLink Yes, PCIe 4.0×16)
  - Testbed B: 单节点 8×NVIDIA A10 (24GB, Ampere, Boost 1.41GHz, NVLink No, PCIe 4.0×16)
  - Testbed C: 单节点 8×NVIDIA H20 (96GB, Hopper, Boost 1.98GHz, NVLink Yes, PCIe 4.0×16)
  - Testbed D: 四节点 32×NVIDIA H20 (每节点 8×H20, 96GB, Hopper, Boost 1.98GHz, NVLink Yes, PCIe 4.0×16)
  软件环境：Ubuntu 22.04, Python 3.10, CUDA 11.3, PyTorch 2.4, NCCL 2.27.5, FlashInfer 0.3.0

- 开源Serving框架是什么。修改了什么。
  论文基于 MegaScale-Infer [36] 中的 PPPipe 算法进行复现和对比，但并未修改特定开源 Serving 框架来部署 FinDEP。论文在白盒实现中直接实现了 DEP 的基础设施：(1) 实现了 AG/EG 分组，AG 负责 Attention 层和 Shared Expert（如有）计算，EG 负责所有 sparse experts 的计算；(2) 使用 NCCL 实现 A2E 和 E2A 通信原语；(3) 使用 FlashInfer 0.3.0 实现 Attention 计算；(4) 实现了 PPPipe 的 micro-batch 流水线调度和 FinDEP 的细粒度任务调度（ASAS 和 AASS 两种执行顺序）；(5) 实现了 offline 性能模型参数采集（α_gm, β_gm, α_attn, β_attn, α_a2e, β_a2e）和 online 快速 solver（Algorithm 1）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未明确提供独立的开源代码仓库链接。FinDEP 的 DEP 推理全流程如下：

  ```
  === Offline Phase (one-time) ===
  1. 选定 serving model (DeepSeek-V2 或 Qwen3-MoE)
  2. 确定 AG 和 EG 大小 (ag, eg)，满足 ag+eg=P
  3. 运行 micro-benchmark 采集性能模型系数：
     - GEMM: t_gm(x) = α_gm + β_gm * x，测试 MLA 中所有矩阵配置
     - Attention: t_attn(y) = α_attn + β_attn * y
     - A2E/E2A Communication: t_c(z) = α_c + β_c * z，测试不同 (ag, eg) 组合
     全过程 < 2 分钟

  === Online Phase (per-request adaptive) ===
  4. 接收用户请求，获取 sequence length S 和 batch size B
  5. 执行 Algorithm 1 快速求解:
     for ma = M downto 1:                           // 按内存上限递减
         r1 = getMaxR1(ag, eg, ma, ...)             // 内存约束下的最大 r1
         if r1 == 0 or r1 == previous: continue     // 跳过非Pareto最优
         for order in {ASAS, AASS}:
             r2* = solve convex min(1/r2) Eq.17     // 凸优化求最优 r2
             me = ma * ag * top_k * S / (r2* * E)   // 反推 me
             if tps > best_tps: update best_config
     返回 best_config = (ma, r1, me, r2, order)
     耗时 < 1 秒

  === Per-Layer DEP Execution (with FinDEP schedule) ===
  For each MoE layer t=1..T:
    AG (per GPU):
      For i = 0..r1-1:
        τ_a^(t,i): Attention 计算 ma 个样本             // t_a(ma) 时间
        τ_s^(t,i): Shared Expert 计算 (ASAS顺序下与下一 Attention 交替)
        τ_a2e^(t,i,j): j=0..r2-1, A2E 通信发送 me 个 token
    EG (per GPU, E/eg experts per device):
      For i = 0..r1-1, j = 0..r2-1:
        τ_e^(t,i,j): Expert FFN 计算 me 个 token         // t_e(me) 时间
        τ_e2a^(t,i,j): E2A 通信返回 expert 输出
  ```

  FinDEP 的核心效果体现在：(1) r1 micro-batch pipeline 使 AG 和 EG 可并行执行，A2E 与 Shared Expert 可并行；(2) r2 fine-grained pipeline 使 A2E/E2A 通信与 EG 计算进一步重叠；(3) ASAS/AASS 两种执行顺序选择使系统能根据 shared expert 开销自适应选择最优策略。在 8×A6000 DeepSeek-V2 S=4096 条件下，非重叠通信时间从 Naive-DEP 的 905.49ms → PPPipe 的 528.94ms → FinDEP 的 309.81ms。
