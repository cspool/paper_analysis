## Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：本论文提出一套 Serving 系统方法论，通过 ArI（算术强度）roofline 分析指导 MLA + MoE 模型的多加速器部署决策，而非实现新的调度器。核心分析包括：(1) **Batch Size 约束推导**：推导出 memory capacity 约束的 $B_{\rm cap}$ 公式（Eq. 11）和 SLO 约束的 $B_{\rm SLO}$ 上界（Eq. 12），分析 MLA（减小 $M_{\rm KV}$ 和 $M_{\rm attn}$）和 MoE（增大 $M_{\rm MoE}$）对可行 batch size 的互补效应；(2) **ArI 驱动的 batch size 目标**：推导 $B_{\rm RP} = \max(B_{\rm attn}, B_{\rm MoE}) = \max(RP_{\rm acc} \cdot deg_{\rm DP}, RP_{\rm acc} \cdot n_e/n_k)$，指导系统配置以满足计算利用率最大化；(3) **通信成本模型**：建立 all-to-all MoE 通信延迟模型 $Comm_{MoE}(B) = 2 \cdot \max_a(\Gamma_{imb}^{acc}(a)) \cdot M_{token} \cdot n_k \cdot B / (BW_{Int} \cdot n_{acc}) + \alpha$；(4) **分解式 serving 架构**：假设 prefill 和 decode 分离在不同机器上执行（disaggregated system），专注于 decode 阶段优化；(5) **部署粒度决策**：对比 256 GPU monolithic vs 32 GPU×8 多实例部署在不同 sequence length 和 skewness 下的吞吐量和负载均衡。
  - 实验比较：(a) **End-to-end throughput-latency tradeoff (Figure 9)**：GPT-3 vs Llama4-Maverick vs DeepSeek-R1 在 32 B200 GPU 下的 decode 吞吐量和 TPOT，分析各模型 $B_{\rm cap}$（DeepSeek-R1: 7360, GPT-3: 124, Llama4: 3328 at L=8192）；(b) **Interconnect 带宽影响 (Figure 10)**：NVLink (1.8 TB/s) vs InfiniBand XDR (100 GB/s) 下 DeepSeek-R1 的系统吞吐量和各阶段执行时间占比（FC/MoE/Attn/Comm），不同 L 和 B 下 all-to-all 通信延迟（151.8 µs vs 17.65 µs at B=128）；(c) **部署粒度对比 (Figure 11)**：256 GPU vs 32 GPU×8，不同 L (2048, 16384) 和不同互联带宽 (900/300/100 GB/s) 下的吞吐量；(d) **Expert 分布偏斜影响 (Figure 12/13)**：Zipfian skewness s 从 0.0 到 0.8，分析系统吞吐量、TPOT、load imbalance ratio ($\Gamma_{imb}^{acc}$) 的变化，以及 256 GPU vs 32 GPU×8 在不同 skewness 下的吞吐量对比；(e) **$B_{\rm RP}$ 与 $B_{\rm cap}$ 分析**：验证 MLA+MoE 模型是否能通过 batch size 使 FC 层达到 ridge point；(f) **FP8 精度对 $B_{\rm cap}$ 的影响**：低精度权重使 $B_{\rm cap}$ 增加，可以匹配 $B_{\rm RP}$ 实现最大吞吐量。

- 硬件平台是什么，配置是什么。
  - 主要评估平台：32 B200 GPU 系统，NVLink 5th Gen 全互联（1.8 TB/s 双向），遵循 NVL72 拓扑。
  - 部分配置使用 InfiniBand XDR（100 GB/s）连接 GPU 组间。
  - 真实硬件验证：DGX H100。
  - $deg_{\rm TP}=8$（GPT-3, Llama4），$deg_{\rm TP}=1$（DeepSeek-R1，因 reordered MLA 中 TP 无益）。
  - $deg_{\rm DP}=4$（GPT-3, Llama4），$deg_{\rm DP}=32$（DeepSeek-R1）。
  - $deg_{\rm EP}=32$（Llama4, DeepSeek-R1），GPT-3 无 EP。

- 开源Serving框架是什么。修改了什么。
  - 论文未使用或修改开源 Serving 框架（如 vLLM、SGLang 等）。实验基于自研 in-house simulator（基于 LLMSimulator https://github.com/scale-snu/LLMSimulator 构建）。模拟器中建模了现代 kernel 级和系统级优化（FlashAttention、FlashMLA、fused kernels、optimized communication），并在 DGX H100 上验证了单节点计算特性，使用 DeepEP 验证了多节点通信时间。
  - 并行策略：Attention block 使用 DP（数据并行），MoE block 使用 EP（专家并行）+ DP。DeepSeek-R1 的 attention block 不使用 TP，因 reordered MLA 中所有 head 共享 $\mathbf{C}_{\rm KV}$ 导致 TP 无延迟收益。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文的分析基于公开可复现的方法论，核心工具链包括：(1) LLMSimulator — 开源 LLM 推理模拟器；(2) FlashMLA — DeepSeek 开源的 MLA decode kernel；(3) DeepEP — DeepSeek 开源的 expert-parallel 通信库。
  - Serving 系统执行全过程（以 DeepSeek-R1 on 32 B200 GPU, decode 阶段为例）：
    ```
    # === 系统配置 ===
    - 32 GPUs, deg_DP=32, deg_TP=1, deg_EP=32
    - Attention Block: 每 GPU 独立处理 B/32 个请求（DP）
    - MoE Block: 256 experts 分布在 32 GPUs（每 GPU 8 experts）（EP）
    - 互联: NVLink 5th Gen, 1.8 TB/s 双向带宽

    # === 单个 Decode Step 的全栈执行流程 ===

    ## Phase 1: Attention Block (DP)
    For each GPU g in [0..31]:
      # 输入: B/32 个请求的 hidden state H (B/32, d_emb=7168)
      # 1. Q 压缩: C_Q = H @ W_CQ[g]  → (B/32, 1536)
      # 2. KV 压缩: C_KV_new = H @ W_CKV[g] → (B/32, 512)
      # 3. 更新 KV cache: C_KV[g] ← append(C_KV_new)
      #    每 token KV$ 仅 68.6KB (576×61×2B)
      # 4. Q 解压缩 (reordered): Q_i @ W_DK_i^T → (B/32, 512)
      # 5. Score (reordered): S_noPE = QW_i @ C_KV^T → (B/32, L)
      #    内存访问: 读 C_KV (d_KVco=512) 而非 K (d_dec=16384)
      #    ArI ≈ 100-200 Op/B, 接近 B200 的 RP_acc=281.25
      # 6. Context (reordered): O_i = softmax(S_i)@C_KV @ W_DV_i
      # 7. Output: U = concat(O) @ W_attn_out → (B/32, 7168)

    ## Phase 2: MoE Block (EP, all-to-all communication)
      # 8. Gating (本地计算):
      #    gate_score = U @ W_route → (B/32, 256)
      #    top_k_experts = topk(gate_score, k=8)  # 选 8/256 experts
      # 9. Token Dispatch (all-to-all通信):
      #    将 B/32 个 token 按 routing 结果发送到对应 expert 所在的 GPU
      #    Comm_dispatch = max_a(Γ_imb^acc(a)) × M_token × n_k × B / (BW_Int × 32)
      #    NVLink: ~17.65 µs at B=128; InfiniBand: ~151.8 µs at B=128
      # 10. Expert Computation (每 GPU):
      #     处理分配给该 GPU 上 8 个 experts 的 token
      #     每 expert: expert_out += gate(W_up × token) × W_down
      #     共享 expert: shared_out = shared_expert(U)  # 每 token 都执行
      #     平均每 expert 处理 B × n_k/n_e = B/32 个 token (均匀分布时)
      #     实际含偏斜: Γ_imb × B/32
      #     MoE FC 层 ArI 随 B 和 n_k/n_e 变化
      # 11. Token Combine (all-to-all通信):
      #     将 expert 输出从各 GPU 传回原 GPU
      #     Comm_combine ≈ Comm_dispatch

    ## Phase 3: 下一个 Decoder Block
      # 12. 重复 Phase 1-2 共 61 次（61 decoder blocks）
    ```
    - 系统瓶颈分析结果：当 B 足够大时，attention block 的延迟占比从 59%（K decompress）+ 40%（core-attention）降至 negligible，MoE 通信时间（dispatch/combine）和执行时间成为主要瓶颈。互联带宽和 expert 负载偏斜是决定端到端性能的主导因素。
