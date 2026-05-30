## Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Sem-MoE，基于 SGLang 构建的 semantic-aware model-data collaborative scheduling 系统（~5000 行 Python + 自定义 Triton kernel）。核心包含三部分：(1) **Offline Model Scheduling**——基于离线 profiling 构建 token-to-expert confidence table，将 ILP-based co-clustering 问题通过交替优化算法求解，生成 expert-to-device 调度表 E 和 token-to-device 调度表 T，在部署前调整每层 expert 的 device placement，同时重排 gate matrix 列实现透明 expert shuffle；(2) **Online Inter-Request Data Scheduling（Attention-DP）**——扩展 SGLang request scheduler，根据 token-expert affinity 将语义相似的请求动态路由到同一 DP rank，最大化 request-expert-group affinity。采用 workload-aware round-robin：每轮每个 device 各分配一个请求，一轮完成后 reset device mask，防止解码阶段负载倾斜；(3) **Online Intra-Request Data Scheduling（Attention-TP）**——在 attention 后的 reduce-scatter 中融合 speculative token shuffling（shuffled-reduce-scatter, SRS），将 token 提前 shuffle 到预测的 expert 所在 device，MoE 计算后再用 shuffled-allgather (SAG) 收集并恢复 token 顺序。同时利用 inter-layer expert-expert affinity（2-gram 马尔可夫链）增强预测精度。
  - 实验比较：(a) Attention-DP 场景——Sem-MoE vs SGLang（含 DeepEP）vs MoETuner，在三种数据集（MMLU/lmsys-chat-1m/ShareGPT）下绘制 latency-throughput 曲线，评估 TTFT SLO 和 E2E Latency SLO 下的最高吞吐（req/s）；(b) Attention-TP 场景——Sem-MoE vs SGLang vs MoETuner，请求率 1 req/s 不变，变化输入长度（256/512/1024），评估 TTFT 和 median E2E 延迟；(c) EP 通信缩减细节——单 MoE layer 的 LAR（Local Activation Rate）vs SGLang vanilla placement，LAR 从 25% 提升至 62%（DeepSeek）和 68%（Qwen3）；(d) 算法评估——Sem-MoE co-clustering vs SGLang vanilla vs MoETuner 的 LAR 和 load imbalance rate；(e) Cross-dataset 零样本迁移——预测器在 ShareGPT/lmsys-chat-1m/MMLU 间跨域评估 LAR；(f) Moonlight-16B 额外验证。

- 硬件平台是什么，配置是什么。
  - GPU：8-GPU server，每 GPU 96GB HBM，fast homogeneous interconnect（>400GB/s 专用互联带宽）
  - CPU：2× 44-core Intel CPU，2TB DDR5 memory
  - 软件栈：SGLang + PyTorch + Triton + DeepEP + NCCL/HCCL

- 开源Serving框架是什么。修改了什么。
  - 开源框架：SGLang（https://github.com/sgl-project/sglang）
  - 修改内容：(1) **Request Scheduler 扩展**——新增 token-expert affinity 感知的调度逻辑，在 Attention-DP 场景将语义相似的请求 batch 到同一 DP rank；(2) **Expert Placement 重配置**——利用离线求解的 expert placement table E 重排各层 expert 的设备分布，同时 shuffle gate matrix 列实现透明重分布；(3) **Shuffled-Reduce-Scatter (SRS) 和 Shuffled-AllGather (SAG) 融合通信原语**——在 Attention-TP 的 post-attention reduce-scatter 中嵌入 speculative token shuffling，用优化 argsort kernel（比 PyTorch 原生快 25%）计算 shuffle indices；(4) **DeepEP 集成**——使用 DeepEP 作为 EP 通信后端（normal mode），提升 all-to-all 效率；(5) **共享内存优化**——将 token-to-device table 等调度表常驻 GPU memory（<12 MB for DeepSeek-V2）。论文未提供独立开源仓库链接，框架修改集成在 SGLang 之上。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文基于 SGLang（开源），但 Sem-MoE 的自研修改代码未在论文中提供独立 GitHub 链接。可基于论文算法描述复现。
  - 框架输入→硬件执行全过程（Attention-DP 场景，DeepSeek-V2-Lite, 8 GPUs, EP8 + DP8）：
    ```
    输入：多 client 请求到达 SGLang HTTP server
    
    阶段 0 — 离线准备（一次性）：
    1. 用 20% ShareGPT/lmsys-chat-1m/MMLU 数据 profile DeepSeek-V2-Lite 各 MoE layer
       → 记录每 token 的 expert activation 频率
    2. 构建 token-to-expert confidence table C_p ∈ R^{t×N}（t=词汇量, N=experts/layer）
    3. 运行交替优化 co-clustering 算法（Algorithm 1）：
       - alternating between expert_place(Cp, req) and request_schedule(Cp, ep)
       - 输出：E（expert labels）、T（token labels）、Tp（confidence table）
    4. 按 E 重排各层 expert 的 device placement，shuffle gate matrix column

    阶段 1 — 在线请求调度（Attention-DP）：
    5. 多个 requests 到达 continuous batching scheduler
    6. Sem-MoE inter-request scheduler:
       - 对每个 request r，提取其 token IDs
       - 查表 T: 聚合各 token 的 device assignment
       - S_r = argmax_{j∈[E]} Σ_{token_i∈r} R_{ij}（最匹配 device）
       - dev_mask round-robin: 每 E 个请求一轮，每轮各 device 分配一个请求
    7. 请求被分配到各自最优 DP rank

    阶段 2 — MoE Layer 前向（每层）：
    8. Attention 计算在各 DP rank 独立完成（数据并行）
    9. Gate function: Softmax(W_g · x) → TopK experts per token
    10. Token→Expert Dispatch（all-to-all）：
        - 由于 expert placement 和请求调度已对齐 token↔expert affinity
        - 大部分 token 的 target expert 在本地 device
        - 仅少量 cross-device token 参与 all-to-all
    11. Expert FFN 计算（各 device 本地 + 来自远程的 token）
    12. Expert→Token Combine（all-to-all 反向）
    13. 输出 hidden states 进入下一 layer

    阶段 3 — Attention-TP 场景（Token-Level 调度）：
    14. Attention TP 计算后，不执行标准 allreduce
    15. 代之以 Shuffled-Reduce-Scatter (SRS):
        - 查表 T 和 inter-layer table A（2-gram Markov model）
        - 选置信度更高的表预测 token 应 shuffle 到的 device
        - argsort 产生 shuffle_indices
        - 按 shuffle_indices 重排 token → reduce-scatter（各 device 获得其管辖的 token 分片）
    16. MoE 层计算（gate + expert FFN）
    17. Shuffled-AllGather (SAG): 收集 token 分片 → 按 reverse_shuffle_indices 恢复顺序

    性能结果：
    - Attention-DP: Throughput ↑ 2.78×（DeepSeek-V2-Lite E2E SLO vs MoETuner）
    - Attention-TP: TTFT ↓ 24.9%（Qwen3-30B-A3B, input=512）
    - LAR 提升: 从 ~25% 到 ~62%（DeepSeek）/ ~68%（Qwen3），对应 41.8%/46.6% expert layer latency reduction
    ```
