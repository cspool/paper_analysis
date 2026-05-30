## InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种免训练的模块化层次化 token 剪枝算法（Modular Hierarchical Token Pruning），通过多阶段剪枝动态消除不相关上下文 token，结合动态 RoPE 调整实现超长上下文外推（OOL generalization）。算法核心：(1) 将输入序列划分为固定大小的 chunk（l_c，默认 Stage 1: 256, Stage 2: 32, Stage 3: 8），在每个 chunk 内通过层次化 top-1 选择（SelectRep，O(log₂ l_c) 时间）选出代表性 token；(2) 利用代表性 token 估计每个 chunk 的注意力分数（跨 head max-pooling），保留 top-K 个 chunk（K = k/l_c），其余丢弃；(3) 通过堆叠 3 个剪枝 stage（N=3），逐步将候选 key 从全量缩减到 2K-4K tokens（3K/5K window preset），最终输出 block sparse attention mask；(4) 动态 RoPE 调整：前 3 层使用 Chunk-indexed RoPE（每 chunk 一个 position ID），后续层使用 Relative-style RoPE（层次化选择中左右分支获得不同偏移），Block Sparse Attention 阶段使用 StreamingLLM-style RoPE；(5) 稀疏注意力 mask 缓存：利用 mask 的时序局部性，以 refresh interval（默认 16/8/4 step）周期性更新各 stage mask，大幅降低 decoding 开销。实验比较 LongBench（平均 32K tokens）、∞Bench（>100K tokens）、RULER 上的 NLU 性能，在 Llama 3 8B、Mistral 0.2 7B、Gemma2 9B、EXAONE 3/3.5 7.8B 上与 FA2（truncated）、Dynamic-NTK、SelfExtend、LM-Infinite、StreamingLLM、H2O、InfLLM、HiP Attention 对比。

- 硬件平台是什么，配置是什么。
  评测使用单卡：(1) NVIDIA RTX 4090 24GB（PCIe 4.0 x8），搭配 AMD Ryzen 7950X 16 核 CPU、128GB DDR5 5600MHz RAM、Ubuntu 22.04.4 LTS、GPU Driver 535.171.04；(2) NVIDIA L40S 48GB（AWS g6e.48xlarge 节点）。长上下文吞吐量测试因显存限制采用估计值（SRT 基线在 1M 上下文需约 64GB KV cache，3M 需约 192GB KV cache，超出单卡容量）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama 3 8B Instruct、Llama 3.1 8B（AWQ 量化 + FP8 KV cache）、Mistral 0.2 7B Instruct、Gemma2 9B、EXAONE 3 7.8B、EXAONE 3.5 7.8B、DeepSeek R1 Distilled Qwen2 14B（Passkey 测试）。Benchmark：(1) LongBench（含 NQA、Qasper、MFQA、HQA、2WMQ、MSQ、GR、QMS、MN、TREC、TQA、SAMS、PC、PR、RBP、LCC 共 16 子集，平均长度 ~32K）；(2) ∞Bench（含 RPK、RN、RKV、MF、MC、QA、SUM 子集，平均长度 >100K，额外含 En.MC/En.QA 用于 OOL 评测）；(3) RULER（含 NIAH 1-3 SK/MK/MV/MQ、VR、CWE、FWE、QA1/2）；(4) Passkey Retrieval（评估 KV cache offloading 延迟）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：(1) hip-attention 核心库：https://github.com/DeepAuto-AI/hip-attention/；(2) SGLang 集成：https://github.com/DeepAuto-AI/sglang/。算法 pipeline 如下：

  **阶段 1 — Multi-stage Context Pruning（生成稀疏 attention mask）**：
  
  给定 query Q ∈ R^(H×T_q×d)，key K ∈ R^(H×T_kv×d)，设 n_sink=256, n_stream=1024, N=3 stages：
  
  1. 初始化 I_m^(0) = [n_sink, ..., b_q^(1)*m - n_stream]（排除 sink/streaming 保持因果性）
  
  2. 对每个 pruning stage i=1..N，参数 S^(i) = (b_q^(i), l_c^(i), k^(i))：
     - 将 query 分为 b_q^(i) 大小的 block：q_{h,m}^(i) = Q_{h, m*b_q : (m+1)*b_q-1}
     - 将上阶段 key indices I_m^(i-1) 分为 l_c^(i) 大小的 chunk：C_{m,j}
     - 对每个 chunk j，每个 head h，执行 SelectRep(q_{h,m}, C_j)：
       * 层次化二分搜索，log₂(l_c) 次迭代
       * 每次迭代：取左右两个分支的第一个 token，ApplyRopeK后用与 q 的点积计算分支分数
       * 选择高分分支继续，最终收敛到单个代表 token 索引 r_{h,m,j}
     - 估计 chunk 分数：s_{m,j} = max_{h=1..H, t=1..b_q} [(q̃_{h,m})_t^T · k̂_{h, r_{h,m,j}}]
     - 保留 top K = k^(i)/l_c^(i) 个 chunk：I_m'^(i) = ∪_{j∈T_m} C_{m,j}（T_m = argtop_K(s_{m,j})）
  
  3. 最终输出稀疏 key indices I_m^(N)，用于 Block Sparse Attention

  **阶段 2 — Block Sparse Attention（基于 mask 的稀疏注意力计算）**：
  - 使用 Triton kernel，Combine PagedAttention + FlashAttention (prefill) / FlashDecoding (decoding)
  - 仅对 I_m^(N) 中的选中 key token 计算完整注意力
  - 注意力 mask 缓存：每 n_refresh^(i) 步更新一次第 i stage 的 mask（默认 16/8/4）
  
  **阶段 3 — Dynamic RoPE for OOL Generalization**：
  - 前 3 层 (l ≤ 3)：ApplyRopeQ_l(q) = ApplyRope(q, p[min(i_orig, l_c + n_stream)])（Chunk-indexed RoPE）；ApplyRopeK_{l,j}(k) = ApplyRope(k, p[c_orig])（chunk 索引作为 position）
  - 第 4 层及以上 (l > 3)：ApplyRopeQ_l(q) = ApplyRope(q, p[n_stream+1])（Relative-style RoPE）；ApplyRopeK_{l,j}(k) = ApplyRope(k, p[j-1])（分支索引作为 position，j ∈ {1,2}）
  - BSA 阶段：选中 key（含 sink+streaming）按原始顺序排列，尾部 token 获得与当前 query 相同的 position ID（StreamingLLM-style）

  **复杂度分析**：
  - 初始 pruning stage: O(T_q * T_kv)（分 chunk+SelectRep 每个 chunk 仅 O(log l_c) 次点积）
  - 后续 pruning stages: O(T_q)（候选 key 数已缩减至常数 k^(i)）
  - BSA: O(T_q * k^(N))，其中 k^(3) = 2K-4K，远小于 T_kv

  **关键超参数**（默认 3K preset）：
  - n_sink=256, n_stream=1024, N=3
  - Stage 1: b_q=64, l_c=256, k=32K
  - Stage 2: b_q=64, l_c=32, k=8K
  - Stage 3: b_q=64, l_c=8, k=2048 (4096 for l≤3)
  - refresh interval: (16, 8, 4)
