## Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现是系统表征研究（无新增 Serving 代码）：以 vLLM v1 推理引擎（PagedAttention，block size B=16，默认 FCFS 调度）为被测 Serving 系统，通过调节 max_num_batched_tokens 与 max_num_seqs 两个调度参数，系统刻画 reasoning（长 CoT）负载下 Serving 调度的瓶颈与并行策略选择。属于"Serving 调度"最接近层次：论文不改代码、而是量化 vLLM 调度器在 KV 容量压力下的行为（并发-容量权衡、preemption、chunked prefill、admission control），并给出 KV-aware 调度与 DP/TP/PP 并行策略选择指南。论文明确将 kernel 级优化抽象掉，只分析系统级容量与调度动力学。
  - 实验比较什么：①并发-容量权衡：DeepSeek-8B 在单 H200 上把 max_num_seqs 从 1K 扫到 10K（10K 输入序列 batch），对比 TTFT/TPOT/E2E/Waiting-Running 时长与吞吐、HBM 带宽利用、KV 占用时间线，发现 E2E 凸曲线与 ≈2K 并发最优甜点（Observation 1/2）；②DP 扩展：固定 8×H200、DP=8，batch size 从 500 扫到 5000，对比聚合吞吐、HBM 带宽、KV 利用率与 E2E（61s→165s 亚线性增长），证明 DP 无法池化内存（Observation 3）；③DP 从 1→8 GPU 扩展的"stranded capacity"与带宽 sawtooth（40%–85% 振荡）现象（Observation 4）；④DP vs TP vs PP vs 混合并行：8B/14B/32B（batch 2K）下 DP 4.9× vs TP 6.15×（32B），14B 最优 DP=8（332s）、32B 最优 DP=4+TP=2（484s）混合策略（Observation 5）；⑤frontier 模型：Llama-405B 密集模型 TP=8（986s）优于 PP=8（7537s，7.6× 慢），DeepSeek-R1-671B 稀疏 MoE 模型 PP=4+TP=2（1663s）优于 TP=8（2047s）（Observation 6）；⑥8B/70B/671B 参数扩展：吞吐亚线性下降（9× 参数→5–6× 吞吐下降），HBM 利用率 8B≈85% vs 671B≈50–60%（带宽-计算反转）（Observation 7）；⑦prefill vs decode 资源发散：prefill compute-bound（SM 占用高、HBM 带宽 ≈30%），decode bandwidth-bound（HBM 带宽 ≈85%）；⑧KV 缩放与"reasoning cliff"：Llama-405B 在 batch 4K/5K 时 KV 在 prefill 阶段即耗尽（Observation 8/9）。
- 硬件平台是什么，配置是什么。
  - 单节点 8× NVIDIA H200 Tensor Core GPU（SXM5）：每卡 141 GB HBM3e、峰值带宽 4.8 TB/s、FP16/BF16 峰值 1979 TFLOPS；第四代 NVLink + NVSwitch 互连，每 GPU 双向 900 GB/s（TP 的 all-reduce 用）；主机双 Intel Xeon Platinum 8558P + 2 TB DDR5 系统内存。所有实验在单个 NVLink 8-GPU 节点内进行（论文视为现代推理的基本扩展单元，TP 限制在 NVLink 域内，更大部署用 DP 复制节点行为）。
- 开源Serving框架是什么。修改了什么。
  - 开源 Serving 框架：vLLM v1（https://github.com/vllm-project/vllm），启用 PagedAttention（KV 按块管理消除内部碎片），block size B=16；调度策略默认 FCFS，但调节 max_num_batched_tokens 与 max_num_seqs 以刻画并发上限。论文未修改 vLLM 源码，只做参数扫描与遥测；复用 LMCache/Mooncake 等 prefix cache 讨论（非实验主体）。论文自身未开源（arXiv:2605.19775，ISCA'26 Industry Track，无代码链接）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文无 GitHub/复现工件（arXiv 页无代码链接，无法确认）；vLLM 开源；数据集 Meta Natural Reasoning（arXiv:2502.13124，NaturalReasoning，1.15M 多跳推理样本，profiling 抽样 100k：77% prompt 为 50–150 token，45% 响应 >5000 token，43.04% 含 >5000 reasoning token）；模型开源：DeepSeek-R1-Distill-Llama/Qwen 变体（Llama-8B、Qwen-14B、Qwen-32B、Llama-70B，GQA，KV 262 KB/token@32B、328 KB/token@70B）、Llama-3.1-405B（密集 GQA，≈1.05 MB/token）、DeepSeek-R1-671B（MoE 激活 ≈37B，MLA 低秩 latent 压缩 KV）。
  - 使用例子（一个含 100 token prompt、将生成 10k reasoning token 的请求，进入 vLLM 跑 DeepSeek-8B 的 Serving 框架输入到硬件执行全过程）：
    ```
    # 输入：HTTP 推理请求（prompt ≈100 token，来自 Natural Reasoning 数据集的推理题）+ 引擎配置（max_num_seqs=2K、max_num_batched_tokens 受限）
    # 1) 请求准入：vLLM 调度器（FCFS）把请求放入 Running 队列并为其分配 PagedAttention KV 块（block size=16，HBM 上按块表管理）
    # 2) Prefill（compute-bound，决定 TTFT）：对 prompt 全部 token 并行执行矩阵乘（GEMM），H200 tensor core 高占用、HBM 带宽仅 ≈30%；
    #    batch 内先到的其他请求也加入；KV 块按 token 写入（KV 占用上升）
    # 3) Decode（bandwidth-bound，决定 TPOT）：逐 token 自回归，每步从 HBM 读全部权重 + 活跃 KV cache（PagedAttention 按块取）；
    #    算术强度塌缩、HBM 带宽饱和 ≈85%；2K 并发下 TPOT≈0.08–0.48s 区间
    # 4) KV 容量压力：10k reasoning token 输出使每请求 KV 累积，聚合 KV 占用逼近 100% → 触发 vLLM 调度器 preemption
    #    （请求降级到 Waiting 队列 / swap 到 CPU）；恢复时 prefix cache 命中失败 → 全量 prefill 重算惩罚（观测到的 E2E 尾部延迟尖峰）
    # 5) 并行扩展路径：单卡瓶颈 → DP=8 复制模型分请求（每卡独立 HBM，625 请求/卡，仍各自撞容量墙）；
    #    32B+ 转 TP（权重分片 64GB→8GB/卡，释放 133GB/卡给 KV）；671B MoE 用 PP=4+TP=2 混合
    # 6) 遥测输出：nvidia-smi 采 HBM 带宽利用、vLLM 指标采 TTFT/TPOT/吞吐/KV 占用/请求状态机（Waiting/Running）→ 判定 Capacity Trap
    ```
    作用：以 vLLM 作为被测系统，量化 reasoning 负载下"并发提高占用率 vs KV 耗尽引发抢占"的根本矛盾，论证 KV-aware 并发上限、TP 容量释放、混合并行与 prefill/decode 解耦等 Serving 层设计原则（论文核心贡献为决策框架而非代码）。
