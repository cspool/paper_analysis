## Context Parallelism (CP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Context Parallelism (CP，上下文并行) 是一种分布式并行策略，将长序列沿sequence/context维度切分到多个GPU上并行处理。与Data Parallelism（沿batch维度）和Tensor Parallelism（沿hidden/head维度）不同，CP切分的是输入序列而非模型权重或batch。CP的核心优势是**不受global batch size限制**的scalability——当batch size因收敛性要求受限时，DP无法扩展，但CP可随context length线性扩展。数学基础：attention公式 $O_{h,c_q} = \sum_{c_{kv}, \oplus} Q_{h,c_q} \otimes KV_{h,c_{kv}}$ 中的 $\oplus$（online softmax reduction）满足结合律和交换律，允许在$c_{kv}$维度任意split和reduce。Attention可沿三个维度并行：h（Head Parallelism，受head数约束）、$c_q$（需通信KV）、$c_{kv}$（需通信Q和O）。UltraAttn沿两个维度同时切分（2D context-tiling），从stripe-like的O(N) projection降至curled-up的$O(\sqrt{N})$。

从系统架构角度拆解，ring attention-based CP（CP=4）的执行流程：
1. 序列S个token划分为4 chunk，每GPU持有S/4个Q/K/V
2. 每GPU计算本地Q×本地KV的attention
3. GPU间peer-to-peer轮转KV：GPU i → GPU (i+1)%4
4. 重复CP-1=3次，使每GPU的Q attend到完整KV
UltraAttn改进：2D context-tiling → ILP分配block到GPU → 每GPU仅接收需要的Q/KV（非全量轮转）→ ILP runtime调度最大化computation-communication overlap

术语一般如何实现？如何使用？实现：Ring Attention (Li/Liu et al.)、Striped Attention (Brandon et al.)、ZigZag Ring Attention (Fang & Zhao)、DeepSpeed Ulysses (head维度)、UltraAttn (2D tiling)。框架：Megatron-LM sequence parallelism、PyTorch FSDP2混合CP+DP、ring-flash-attention开源库。CP通常与其他并行策略（DP/TP/PP）组合为4D parallelism。通信量在固定context length下随CP增加几乎不变（ring-based），是long-sequence scaling的瓶颈——UltraAttn的目标。

涉及论文标题：
- UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training

DSV在稀疏场景下对CP的贡献：(1) 形式化分析HCP和SCP在稀疏attention下的trade-off。HCP——通过All-to-All重分布head子集，通信量comm_i^hcp = 4·H_i^r·S/N·(N-1)/N·D，近乎恒定但需解决head-wise sparsity heterogeneity导致的load imbalance（用Longest Processing Time算法优化head分配）；SCP——每GPU持有部分序列+全部heads，仅传输critical KV（通信量∝(1-α)·σ，α为sparsity ratio，σ为remote critical KV比例），天然负载均衡但通信量动态变化。(2) 提出Hybrid Sparsity-Aware CP，将CP group划分为HCP和SCP子组，形式化为min-max优化问题（最小化max_i(T_i^comm + T_i^comp)，约束mem_i ≤ M），用Gurobi周期性求解。(3) 部署策略：HCP组优先节点内（利用NVLink高带宽All-to-All），SCP组用于跨节点（仅传输critical KV，通信量小）。
