## Towards MoE Deployment Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：论文在开源 fairseq [23] MoE Transformer 推理框架上实现了三项 Serving 调度优化：(1) **Dynamic Gating（动态门控）**：将原有的静态容量（static capacity）all-to-all token 分发机制改为动态可变长度的 argsort + 两阶段 all-to-all（先交换各专家的 token 数量，再传输实际 token），消除了 dispatch mask 的大矩阵乘法和空 token placeholder 传输；(2) **Expert Buffering（专家缓冲）**：在 GPU 内存中维护一个可配置大小的 expert cache，仅缓存活跃（hot）专家参数，其余专家参数存放于 CPU 内存，按需通过 Memcopy 加载，结合 LIFO 淘汰策略利用时序局部性；(3) **Load Balancing（负载均衡）**：基于历史 expert 激活数据的 greedy 多路分区算法（NP-hard 近似），将高负载和低负载专家混合分配到各 GPU，以及针对 MT Decoder 的 Anti-correlation Balancing 变体。
  - 实验比较：(a) Dynamic gating vs Static gating (baseline [2]) vs Tutel gating [16] 在不同 batch size 下的吞吐量和内存消耗（Figures 9, 10）；(b) single-node vs multi-node (2/4 nodes) 部署下动态门控的吞吐提升（LM: 6.21× single-node, MT Encoder: 10.98× multi-node, MT Decoder: 5.71× multi-node）；(c) Expert Buffering 在不同 GPU cache size（每 GPU 1-16 experts）下的 cache miss rate vs Belady's MIN 理论最优（Figure 12）；(d) Expert Buffering 的延迟-内存帕累托前沿（Figure 13）；(e) Load Balancing 的负载均衡效果（Max Load 和 Avg Max Load，Figure 14）；(f) Dynamic Gating + Expert Buffering + Load Balancing 三者组合的端到端吞吐对比（Figure 9）。

- 硬件平台是什么，配置是什么。
  - CPU: 2×Intel Xeon E5-2698 v4 @ 2.2GHz，700GB 内存
  - CPU-GPU 互联: PCIe 3.0，带宽 16GB/s
  - GPU: 8×NVIDIA Tesla V100，每卡 5120 CUDA cores，32GB HBM2 @ 900GB/s，NVLink 互联 @ 300GB/s
  - 单节点 8 GPU 到多节点（2/4 节点，即 16/32 GPU）的扩展实验

- 开源Serving框架是什么。修改了什么。
  - 开源框架: fairseq [23]（https://github.com/facebookresearch/fairseq），Meta 开源的可扩展序列建模工具包，内置 MoE Transformer 实现
  - 修改内容: (1) **Gating 模块**：将 fairseq 中基于 static capacity + dispatch mask（E, S, S×C 大小的稀疏矩阵乘）的 token 分发改为 argsort 排序 + 两阶段 all-to-all（先 size all-to-all，再 data all-to-all）+ index-based 重排，取消了 token dropping；(2) **Expert 内存管理**：新增 Expert Buffering 缓存管理器，包含 GPU expert cache（可配置大小）、CPU 内存专家参数存放、LIFO 淘汰策略、Memcopy 与 all-to-all 通信的 overlap 机制；(3) **Expert 放置策略**：新增基于历史激活数据的 greedy load balancing 算法，在模型加载时重排 expert-to-GPU 的分配映射。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未明确提供独立开源仓库。修改基于 fairseq（开源 MoE Transformer 工具包），本论文工作实现了系统原型。fairseq 本身在 GitHub 开源。论文明确声明"implement it on an open-source, state-of-the-art MoE-based Transformer [23]"，但未提供论文修改的具体 fork/branch 链接。建议在 fairseq 仓库中搜索动态 gating 相关的 commit 或分支。
  - 框架输入到硬件执行全过程（以 Language Modeling，Dynamic Gating + Expert Buffering 为例）：
    1. **输入**：batch_size=8 的 token 序列（PILE 数据集），经过 tokenizer 嵌入后形成大小为 (B, S, TD) 的张量
    2. **Dense Transformer Layer**：输入经过 MHA（Multi-Head Attention）+ residual 计算，输出 (B, S, TD) 张量
    3. **MoE Layer - Gating**：Gating 函数（轻量级线性层）计算每个 token 到每个专家的 affinity score，top-2 gating 为每个 token 选择 2 个专家，输出 assignments (B, S, 2)
    4. **Dynamic Token Dispatch**：
       - argsort assignments 得到最优 token 排列索引（O(S log S)）
       - bin-count 统计每个 GPU 上各专家的 token 数量
       - 第一阶段 size all-to-all：各 GPU 交换 token 分配量信息（极小消息）
       - 第二阶段 data all-to-all：按 idx 重排 tokens 后分片发送
    5. **Expert Buffering Check**：
       - 检查 kernel 分配到的 experts 是否在 GPU expert cache 中
       - Cache hit → 直接使用 GPU 缓存中的 expert FFN 参数
       - Cache miss → 从 CPU 内存 Memcopy 加载 expert 参数到 GPU（与 all-to-all 通信并行 overlap）；若 cache 满则按"非当前 batch 活跃 + LIFO"策略淘汰
    6. **Expert FFN 执行**：各 GPU 上按 expert id 顺序串行执行 expert FFN（两个线性层 + 激活函数），输入 (tokens_per_expert, TD) → FFN1 → GELU → FFN2 → (tokens_per_expert, TD)
    7. **Token Collection**：处理完成后，通过另一个 all-to-all 将 tokens 发回原始 GPU，用 index-based gather 恢复原始顺序
    8. **后续层处理**：经残差连接合并，继续经过后续 dense/MoE 层（LM: 共 24 层，MoE 层频率 MF=2，即每 2 层中 1 层为 MoE）
    9. **输出**：LM head 产生 vocab 概率分布，输出下一个 token 预测
