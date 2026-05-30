## HarMoEny: Efficient Multi-GPU Inference of MoE Models

- baseline方法是什么？
  - **现有 MoE 多 GPU 推理系统**：使用 expert parallelism (EP) 将 expert 分布到多 GPU，通过 all-to-all 通信完成 token dispatch/combine。负载均衡策略分为两类：
    1. **Static placement（DeepSpeed, FastMoE, FasterMoE）**：使用 round-robin 将 expert 分配到 GPU，不含 token 级负载均衡。FasterMoE 增加 dynamic shadowing（将热门 expert 参数广播到所有 worker），但 shadowing 受 GPU memory 限制（大 expert 模型如 Qwen 33MB/expert 时效果受限）。
    2. **Profiling-based placement（ExFlow）**：离线 profiling 后使用 integer programming 计算最优 expert placement，利用 inter-layer expert affinity 减少 all-to-all 通信。但 profiling 开销极大（Switch128 需 8.5 分钟，Qwen 需 45 分钟），无法适应 batch 间动态变化的 expert 流行度。
  - Baseline 痛点：
    1. **动态 Expert 流行度偏斜导致 GPU 严重欠利用**（核心痛点）：实际 workload 中 expert 流行度随输入 domain 变化（如 medical vs programming prompts），且 batch 间剧烈波动。图 1 显示 Qwen model 的层 0 仅 3/128 expert 接收平均 19% token，最后层 3 expert 接收 60%。这导致 GPU idle time 高达 82-86%。
    2. **Static/profiling 方案无法适应动态偏斜**：静态方案（round-robin placement）完全无法处理偏斜；profiling 方案（ExFlow integer programming）在 batch 间偏斜波动时来不及重新计算（profiling 时间 >> batch 处理时间）。
  - 全栈执行例子（Baseline DeepSpeed/FastMoE on 8×V100, Switch128, 90% skew workload）：
    - **模型推理算法层**：Switch128 MoE, 128 experts, top-1 routing。Token → self-attention（各 GPU 复制执行）→ Router（各 GPU 复制执行）→ all-to-all dispatch token 到 expert 所在 GPU → 各 GPU 本地 expert FFN (GeMM) → all-to-all combine 返回输出。
    - **系统框架层**：DeepSpeed Tutel/FastMoE，EP=8。Expert 按 round-robin 分配：GPU0 持有 expert 0,8,16,..., GPU1 持有 expert 1,9,17,...。All-to-all 通信引入两个同步 barrier（dispatch + combine）。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL 后端。
    - **kernel 调度层**：NCCL all-to-all collective kernel + cuBLAS GEMM。GPU0（持有热门 expert 0-9 中的 2 个）计算时间远长于其他 GPU → GPU1-7 在 all-to-all barrier 处等待 GPU0。图 5(a) 显示 GPU1-7 idle >82% 时间，mean batch latency 289ms。
    - **硬件架构层**：8× V100 (32GB), NVLink, 500GB system memory。PCIe 带宽用于可能的 expert 交换。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HarMoEny 方法**：通过两个互补的轻量级技术在不做 profiling 的前提下实现 near-perfect 负载均衡：
    1. **Dynamic Token Rebalancing（Algorithm 2）**（解决痛点 1）：每个 batch 中 GPU 先交换轻量 metadata（约 4kB），构建全局 token-to-expert-to-GPU 分布。贪心算法迭代识别最过载 GPU g_max → 贡献最大 token 的源 GPU g_from → 发送最多 token 的 expert e_max → 将 e_max 的 token 从 g_max 重路由到最欠载 GPU g_min。重复直至所有 GPU 负载平衡或无可转移 token。token threshold q 控制最小 transfer 粒度（由硬件规格决定的静态参数：q > φ·d_type/(2β)）。
    2. **Asynchronous Expert Prefetching（Section 4.3）**（使 rebalancing 可行）：rebalancing 可能将 token 分配到未持有对应 expert 的 GPU。HarMoEny 通过独立 CUDA stream 从 system memory 异步 prefetch 所需 expert 权重，直接覆写已完成 expert 的内存（无需写回 system memory），overwrite 加速 5.5×。prefetch 与当前 expert 计算重叠，隐藏传输延迟。
  - 全栈执行例子（HarMoEny on 8×V100, Switch128, 90% skew，对比 baseline）：
    - **模型推理算法层**：与 baseline 相同的 Switch128 MoE 模型。差异在于 MoE forward 流程被重写为 Algorithm 1 的 6 steps——在 all-to-all dispatch 之前插入了 Step 2 (metadata exchange) + Step 3 (token rebalancing)。
    - **系统框架层**：HarMoEny 用 1115 行 PyTorch 代码实现自定义 MoE Layer（nn.Module），替换标准 MoE 层。Step 2 metadata exchange 在所有 GPU 间广播 token-expert assignment（4kB），Step 3 各 GPU 独立并行运行相同 rebalancing algorithm（因 metadata 一致，结果 deterministic）。Step 4 all-to-all 基于 rebalanced schedule S（而非原始 round-robin assignment）。Step 5 中 async expert loading 在独立 CUDA stream 执行，overlap 当前 expert computation。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL + CUDA streams。
    - **kernel 调度层**：与 baseline 差异：(a) all-to-all 通信的 token 分布从 skewed（GPU0 接收 9× tokens）变为 balanced（各 GPU 处理 ≈t_avg tokens）；(b) 异步 CUDA stream 执行 system memory → GPU memory 的 expert weight copy（expert 18MB → ~2ms transfer on V100, vs sync 11ms），与 compute stream 的 expert FFN GeMM 重叠。图 11 显示 rebalancing 后 GPU idle 从 82.6% → 2.6%，async prefetch 进一步降低 latency 8.6%。
    - **硬件架构层**：与 baseline 相同（8×V100 DGX1）。关键差异：GPU idle time 从 82.6% → 2.6%（rebalancing alone）→ 进一步减少（async prefetch）。Scheduler 开销：30.8% (Switch128) / 20.3% (Qwen) of mean layer latency，但 total latency 仍显著降低（Switch128: 289ms → 136.6ms, -52.7%；Qwen: -63.7%）。
  - **关键性能对比**：
    - 90% skew, Switch128: throughput 186 tok/s vs ExFlow 106 tok/s (+75%), TTFT 5.36ms vs 9.38ms (-43%)
    - 50% skew, Switch128: throughput 201 tok/s vs FasterMoE 155 tok/s (+30%)
    - Qwen (larger experts 33MB): throughput 36 tok/s, consistently 15-28% faster than FastMoE/FasterMoE across real-world datasets
    - Real-world datasets: HarMoEny maintains steady 201 tok/s (Switch128) and 36 tok/s (Qwen), while baselines fluctuate with expert popularity changes
    - Fluctuating skew (0-95% per batch): HarMoEny throughput variance 152 tok²/s² vs FasterMoE 447 tok²/s² (+2.9× more stable)
    - GPU idle reduction: 84.7% vs baseline policies
  - **核心设计洞察**：HarMoEny 的本质是将 MoE 推理的负载均衡问题从一个"offline placement 问题"重新定义为"online scheduling 问题"。其核心创新在于利用了 MoE 推理的一个被忽视的性质——all-to-all 同步 barrier 之前存在天然的决策窗口。通过在这个窗口中插入轻量 metadata exchange（4kB），所有 GPU 获得全局视图，可以 deterministic 地计算相同的 rebalanced schedule，无需额外同步。这使得负载均衡的开销从 profiling 的分钟级降到 metadata exchange 的微秒级，从而能够适配 batch-by-batch 的 expert 流行度波动。Async expert prefetching 是一个精巧的补充设计——它将"rebalancing 需要 expert 移动"这个看似限制转化为优势：overwrite-based loading 比传统的 write-back-then-load 快 5.5×，因为省去了 system memory 回写步骤。论文通过 Equation (4) 将 token threshold q 形式化为仅依赖硬件规格的静态参数，使系统设计者无需 per-model/per-workload 调参。最终效果是近乎完美的 GPU 负载均衡（图 2 ECDF），将 GPU idle time 降至几乎为零。
