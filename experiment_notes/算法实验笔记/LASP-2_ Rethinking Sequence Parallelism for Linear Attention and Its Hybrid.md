## LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid

- 属于算法pipeline的实现是什么？实验比较什么？
  LASP-2 是一种针对线性注意力（Linear Attention）的序列并行（Sequence Parallelism, SP）算法。核心实现：将长序列切分为 T 个 chunk 分布到 W 个设备上，各设备并行计算 Q_t, K_t, V_t 及 local memory state M_t = K_t^T V_t（形状 d×d，与序列长度无关），然后通过**单次 AllGather 集合通信**将所有设备的 M_t 汇聚到所有设备，各设备本地累加得到全局 M_{1:T} = Sum([M_t]_1^T)，最后本地计算 O_t = Q_t M_{1:T}。对有 causal mask 的自回归任务，LASP-2 采用计算分解（computation decomposition）：intra-chunk 部分保持 quadratic 左乘计算 O_{t,intra} = [(Q_t K_t^T) ⊙ Ψ] V_t，inter-chunk 部分用线性右乘 O_{t,inter} = Q_t M_{1:t-1}，且 AllGather 通信可与 intra-chunk 计算 overlap。LASP-2H 将同样的 AllGather 通信范式扩展到标准 softmax attention 的 Context Parallelism（AllGather K_t, V_t 后本地计算 attention）。实验比较 LASP-2 vs Megatron-SP、Ring Attention、LASP-1 在 throughput (tokens/s)、scalability（序列长度 2K-2048K，GPU 数 16-128）、convergence performance（多种 linear attention 变体）等维度。

- 硬件平台是什么，配置是什么。
  最多 16 台 DGX-A100 服务器，每台 8 张 A100 GPU（共 128 卡），NVSwitch 互联提供 600 GB/s GPU 间带宽。PyTorch 2.3.1 + CUDA 12.1 + cuDNN 8.9.2 + NCCL 2.20.5。基于 Megatron-Core 0.9.0 开发，Triton 2.3.1 加速 GPU 上的线性注意力计算，FlashAttention-2 作为标准 attention 实现。

- 模型是什么。数据集和bench分别是什么。
  模型：Linear-Llama3-1B（16 层，将 Llama3 的标准 softmax attention 替换为多种线性注意力模块：Basic Linear Attention、Lightning Attention、Retention、GLA、Based、Rebased），hidden dim d=2048，16 heads。Hybrid 模型：每第 4 层保留标准 softmax attention（1/4 hybrid）。额外评估 RoBERTa + Basic Linear Attention 在双向语言建模任务上。
  数据集：SlimPajama（627B tokens 全量），实验使用训练集第一个 chunk 的 50B tokens 子集。Llama3 tokenizer。GPT-style 自回归语言建模（带 causal mask）。评估指标为 training loss 和 validation loss。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/OpenSparseLLMs/Linear-MoE（LASP-2 作为 Linear-MoE 系统的 SP 子模块发布）

  **算法 Pipeline（LASP-2 without Masking，基于 Algorithm 1）**：

  **输入**：序列 X ∈ R^{N×d}，分布式 world size W，SP size T=W，将 X 切分为 T 个 chunk [X_t]_1^T。

  **Step 1 - 并行计算 local Q/K/V/M**（每个设备并行）：
  ```
  Q_t = X_t W_Q    # shape: [C, d]
  K_t = X_t W_K    # shape: [C, d]
  V_t = X_t W_V    # shape: [C, d]
  M_t = K_t^T V_t  # shape: [d, d]  ← 与 chunk 长度 C 无关！
  ```

  **Step 2 - AllGather 通信**：
  ```
  [M_1, M_2, ..., M_T] = AllGather([M_1, M_2, ..., M_T])
  # 每个设备获得全部 T 个 memory states，通信量 = T × d × d × BH
  ```

  **Step 3 - 本地累加**（所有设备并行，递归执行）：
  ```
  M_{1:T} = M_1 + M_2 + ... + M_T   # 使用递归：M_{1:t} = M_{1:t-1} + M_t
  # 缓存 M_{1:T} 到 HBM 用于 backward pass
  ```

  **Step 4 - 本地输出计算**：
  ```
  O_t = Q_t M_{1:T}   # shape: [C, d]
  ```

  **有 Mask 版本（LASP-2 with Masking, Algorithm 2）**：
  - **Intra-chunk**：O_{t,intra} = [(Q_t K_t^T) ⊙ Ψ] V_t（quadratic 左乘，但仅限 chunk 内，可并行）
  - **Inter-chunk**：O_{t,inter} = Q_t M_{1:t-1}（线性右乘，其中 M_{1:t-1} = PrefixSum([M_1, ..., M_{t-1}])）
  - **关键优化**：AllGather（line 7）与 intra-chunk 计算（line 8）可在不同 CUDA stream 上 overlap
  - **最终输出**：O_t = O_{t,intra} + O_{t,inter}

  **Backward Pass（Algorithm 3, 4）**：
  - 计算 dM_t = Q_t^T dO_t
  - AllGather([dM_t]_1^T) 汇聚梯度
  - 无 mask：dM_{1:T} = Sum([dM_{t+1}]_T)，推导 dQ_t, dK_t, dV_t
  - 有 mask：intra/inter 分别计算后合并 dQ_t = dQ_{t,intra} + dQ_{t,inter}（类似 forward）

  **与 LASP-1 的关键差异**：
  - LASP-1：ring-style P2P 逐设备顺序收发 M_t，共 2(W-1) 个通信步骤，每次传 BHd^2 数据
  - LASP-2：单次 AllGather，仅 2 个通信步骤（forward + backward），通信量同为 BHd^2 但并行度大幅提升

  **LASP-2H 混合模型扩展（Algorithm 7）**：
  - Linear Attention 层：同上，AllGather M_t（d×d 大小）
  - Standard Attention 层：AllGather K_t, V_t（C×d 大小），本地计算 Softmax(Q_t K^T / √d) V

- 属于算法pipeline的实现是什么？实验比较什么？
  KVzip 是一种 query-agnostic（查询无关）的 KV cache 淘汰算法。核心实现是基于上下文重建（context reconstruction）的 KV pair 重要性评分：将 "Repeat the previous context:" prompt + 原始 context chunk 拼接后通过 LLM forward pass，利用 teacher-forced decoding 模拟上下文重建过程，对每个 KV pair 取其在重建过程中收到的最大 cross-attention score 作为重要性分数 S ∈ R^{L×H×n_c}，随后按 non-uniform head-budget allocation 保留 top r% 高分的 KV pairs，淘汰低分 pairs。支持 context-dependent eviction（per-context 压缩，更高压缩比）和 context-independent eviction（预计算 head-level score，部署时零开销）两种模式。实验比较 KVzip 与 query-aware 方法（H2O、SnapKV、PyramidKV）以及 head-level 淘汰方法（DuoAttention）在 KV cache budget ratio 0.1-1.0 下的多查询/单查询场景性能，涵盖 12 个 benchmark 数据集。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA A100 80GB GPU，Bfloat16 精度。FlashAttention-2 加速注意力计算。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA3.1-8B（GQA group=4）、Qwen2.5-7B-1M（GQA group=7）、Qwen2.5-14B-1M、Gemma3-12B（hybrid attention: global + sliding window 1:5）、LLaMA3.1-3B、LLaMA3-8B-W8A8KV4（QServe W8A8KV4 量化）。
  数据集/Benchmark：SQuAD、GSM8K（数学推理）、Needle-in-a-Haystack / NIAH（检索）、SCBench 9 个任务（En.QA、En.MultiChoice、Retr.KV、Retr.Prefix-Suffix、Math.Find、En.Summary 等，含 retrieval-intensive、contextual understanding、high context redundancy 三类）、RULER benchmark、SCBench multi-task datasets（Mix.Sum+NIAH、Mix.RepoQA+KV）。上下文长度 100 到 170K tokens（Qwen2.5 tokenizer），评估主要在多查询 query-agnostic 框架下（Figure 1c）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/snu-mllab/KVzip
  
  **算法 Pipeline（基于论文 Algorithm 1）**：
  
  **Step 1 - Prefill**：将输入 context c（n_c tokens）通过 f_LM 前向传播，生成完整 KV cache KV_c，共 L×H×n_c 个 KV pairs（L 层，H 个 KV head，使用 GQA）。
  
  **Step 2 - Chunking**：将 c 划分为 T = ⌈n_c/m⌉ 个 chunk，每 chunk 固定大小 m=2K（与上下文长度、模型、任务无关，Section C.1 验证影响 <2%）。
  
  **Step 3 - 逐 Chunk 重要性评分**（for t = 1,...,T）：
  - 构造 input：
    - t=1: `"Repeat the previous context:" + c_1`
    - t≥2: `"Repeat the previous context starting with <c_{t-1} last 8 tokens>:" + c_t`
  - 将 input（长度 n_in = n_prompt + m）通过 f_LM 前向，使用 KV_c 作为 KV cache。
  - 对每层 l、每 KV head h：
    - 获取 Query: Q_{l,h} ∈ R^{G×n_in×d}（G 为 grouped-query size）
    - Subsample Key: K̄_{l,h} ∈ R^{(m+n_in)×d}（从 KV_c 中取出当前 chunk 对应部分 + input 自身的 keys）
    - 计算注意力: A_{l,h} = Softmax(Q_{l,h} K̄_{l,h}^T) ∈ R^{G×n_in×(m+n_in)}
    - 切片 KV_c part: Ā_{l,h} = A_{l,h}[:,:,:m] ∈ R^{G×n_in×m}
    - 沿 query 维度取 max: S_{l,h,t} = max_{g=1..G, i=1..n_in} Ā_{l,h}[g,:,i] ∈ R^{H×m}
  
  **Step 4 - 聚合得分**：将所有 chunk 得分拼接为完整得分 S ∈ R^{L×H×n_c}。
  
  **Step 5 - 淘汰（Non-uniform head-budget）**：保留所有 KV pairs 中 top r% 最高 S 值的 pairs。system prompt 的 KV pairs 始终保留。r=1.0 为全量 cache。
  
  **Step 6 - Decoding**：使用压缩后的 KV_{c,evicted} 进行 FlashAttention 解码，享受降低后的内存占用和注意力延迟。
  
  **复杂度**：O(m·n_c) 线性于上下文长度（vs 标准 prefill 的 O(n_c²/2)），压缩开销约 2× prefill（Figure 8b）。峰值内存 O(m²) 恒定。chunked scoring 的 FlashAttention 总 FLOPs 为 O(n_c² + n_c·m/2)。
  
  **上下文无关变体（Context-independent eviction）**：对每 head 取 S_head[l,h] = max_i S[l,h,i]，使用单个 88K-token 英文书样本预计算 head-level 重要性分数。部署后无需任何压缩开销，直接应用 DuoAttention 的 head-level KV eviction 策略。性能略低于 context-dependent mode 但显著优于 DuoAttention 的原版 head-score 优化（后者需数小时 8-GPU 优化，KVzip 仅需数次 forward pass 一分钟内完成）。
