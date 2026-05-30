## Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling

- baseline方法是什么？
  Baseline 方法：SGLang（vLLM）的 Expert Parallelism (EP) 推理方案。在 MoE 推理中，expert 按轮询或默认策略均匀分布到各 GPU 上，attention 层使用 DP 或 TP。每个 token 的 gate function 在运行时选择 top-k experts，通过 all-to-all 集合通信将 token dispatch 到远端 expert 所在 GPU 执行 FFN 计算，再 combine 回来。由于 expert 放置不考虑 token-expert affinity，导致大量 cross-device token 路由（~75% tokens 需远端通信），all-to-all 通信成为主要延迟瓶颈（占 MoE 层 59.2% forward latency，见图 1）。

  全栈执行例子（Baseline: SGLang Attention-DP + EP8, DeepSeek-V2-Lite, 请求 A/B/C/D 到达）：
  - **算法Pipeline层**：MoE Gate: Softmax(W_g · h) → Top-K → expert indices（如 token t1→E3,E15; t2→E7,E22; t3→E1,E8...）。Attention: Q/K/V projection → FlashAttention → output。每个 token 独立选择 expert，无先验知识引导。
  - **系统框架层**：SGLang continuous batching → 请求 A-D 被轮询分配到 4 个 DP rank → 各 rank 独立执行 attention → MoE layer: gate → all-to-all dispatch（每 rank 将 token 发送到其选中 expert 所在 rank）→ expert FFN → all-to-all combine（将输出发回原 rank）。expert 默认以轮询方式放置，与 token 语义无关。EP 通信由 NCCL all-to-all 实现，all-to-all 通信量 = αkBS/G。
  - **编译框架层**：SGLang 使用 Triton fused MoE kernel，将 gate + expert computation 融合以减少 kernel launch 开销。但 all-to-all 通信与计算串行执行，无法通过编译优化消除。
  - **Kernel调度层**：NCCL/HCCL all-to-all collective → GPU SM 执行 expert FFN（fused MoE kernel）→ NCCL all-to-all。通信与计算无重叠（EP 的 all-to-all 必须在 gate 之后、expert computation 之后分别执行，形成两道通信 barrier）。
  - **硬件架构层**：8-GPU server（96GB HBM/GPU，>400GB/s 互联），all-to-all 数据经 NVLink/NVSwitch 全交换。

  核心缺陷：(1) Expert placement 与 token 语义无关——轮询放置导致任何 token-expert 对的本地激活概率仅 ~1/E≈12.5%（EP8），~75% tokens 需要跨设备通信；(2) 请求调度不考虑 expert affinity——DP 场景请求被轮询分配，不同请求的 token 被随机散到各 device，加剧 all-to-all 通信；(3) TP 场景 token 在 reduce-scatter 后随机分布，到 MoE 层再通过 all-to-all 重新路由，重复通信。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 **Sem-MoE (Semantic Parallelism)** 通过 model-data collaborative scheduling 从根本上减少 EP 通信量：
  
  **(1) Offline Model Scheduling → 解决 expert placement 与 token 语义无关**
  通过对 ShareGPT/lmsys-chat-1m/MMLU 数据集的 offline profiling，发现 token-expert activation 存在强 context-independent correlation（median cumulative hotness of top-k experts: 0.833-0.976，Table 1）。基于此构建 token-to-expert confidence table C_p，将 expert 放置建模为 ILP co-clustering 问题——最小化 objective = θ·load_imbalance + (1-θ)·remote_activation，通过交替优化求解（Algorithm 1）。结果：语义相近的 expert 被 clustered 到同一 device，使得该 device 上的 token 的 top experts 大概率在本 device 上。

  **(2) Online Inter-Request Data Scheduling (Attention-DP) → 解决请求调度不考虑 expert affinity**
  将请求级调度纳入 affinity 框架：对每个请求 r，聚合其所有 token 的 device assignment → S_r = argmax Σ_i R_ij。配合 workload-aware round-robin（每 E 个请求一轮，每 rank 各一个），语义相似的请求被分配到同一 DP rank，大幅减少 all-to-all 通信。
  
  **(3) Online Intra-Request Data Scheduling (Attention-TP) → 解决 token 重复通信**
  在 post-attention reduce-scatter 中嵌入 speculative token shuffling（SRS kernel）：根据 token-to-device table T 和 inter-layer 2-gram confidence table A，预测每个 token 在下一 MoE 层的 target device → argsort 重排 token → reduce-scatter 分发。MoE 计算后通过 shuffled-allgather (SAG) 恢复顺序。将原本独立的 all-to-all dispatch 通信融入已有的 reduce-scatter 操作。

  **(4) Inter-Layer Expert-Expert Affinity Modeling → 增强 TP 场景预测精度**
  利用 2-gram Markov chain 建模跨层 device transition：Pr(D_k^(L)|D^(L-1), D^(L-2))。当 token-level confidence 低时（OOV tokens），切换到 inter-layer prediction。两表竞争选置信度高者，提升预测鲁棒性。

  **(5) 零额外修改 MoE 架构**
  Sem-MoE 不修改 MoE 模型架构（无 pre-gate module，无 router 修改），仅通过重排 gate matrix column + expert device placement + 通信原语修改实现，与现有 MoE 模型完全兼容。

  全栈执行例子（Sem-MoE Attention-DP, DeepSeek-V2-Lite, 4 DP ranks, 请求 A-D 到达）：
  - **算法Pipeline层**：同 baseline——Gate: Softmax(W_g · h) → Top-K experts。算法本身不变，但 gate matrix W_g 的 column 被 Sem-MoE 重排以匹配新的 expert placement（透明 shuffle）。
  - **系统框架层**：Sem-MoE on SGLang：
    Offline: profile → C_p → co-clustering solver → E (expert labels) + T (token labels) + A (inter-layer table)
    Online inter-request: 请求 A-D 到达 → Aggregator 查 T 表统计各 token 的 device assignment → S_r = argmax → 请求 A（代码类语义）→ DP_0（host experts for code tokens），请求 B（数学类语义）→ DP_1 → ...。Round-robin 保证每轮每 rank 各一个请求，防止解码阶段负载倾斜。
  - **编译框架层**：论文未明确说明，SGLang 原有 Triton fused MoE kernel 保持不变。
  - **Kernel调度层**：
    Attention (DP, 各 rank 独立) → MoE layer:
    1. Gate: G = Softmax(W_g · X)  （Gate column 已 shuffle，transparent to user）
    2. All-to-All Dispatch: 仅 cross-device token 参与通信（LAR 从 ~25% 提升至 ~62%）
    3. Expert FFN: 大部分 token 的 expert 在本地（LAR↑），远程通信量大减
    4. All-to-All Combine: 同样缩减
    
    (Attention-TP 场景):
    Post-attention: Shuffled-Reduce-Scatter (SRS):
      - 查 T 和 A 表 → 选置信度高者 → argsort → token shuffle → reduce-scatter
      - Token 被预送到 expert 所在 device，省去后续 all-to-all dispatch
    MoE: gate + local expert FFN (高 LAR)
    Post-MoE: Shuffled-AllGather (SAG): allgather + 反向 argsort 恢复 token 顺序
    
    调度表内存：~11.72 MB for DeepSeek-V2（int16），完全驻留 GPU memory。
  
  - **硬件架构层**：同 baseline——8-GPU server（96GB HBM/GPU，>400GB/s 互联），但 all-to-all 通信量减少，LAR 从 25% 升至 62%（DeepSeek）/68%（Qwen3），expert layer 延迟降低 41.8%/46.6%。

  关键结果：Attention-DP: Throughput ↑ 2.78× (E2E SLO vs MoETuner)、↑ 31% (TTFT SLO vs SGLang)；Attention-TP: TTFT ↓ 24.9% (Qwen3, input=512)。Cross-dataset 零样本迁移：ShareGPT 训练 → lmsys-chat-1m LAR 从 25% 提升至 41.25%（1.65× baseline），接近 in-domain 最优（47.19%）。
