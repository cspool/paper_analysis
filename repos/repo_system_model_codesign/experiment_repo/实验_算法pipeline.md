# 实验_算法pipeline

## Using Span Queries to Optimize Cache and Attention Locality

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **Span Query 范式**——将 LLM inference 的输入从线性 token 序列（Chat Completion API）泛化为带 commutativity constraints 的 expression tree（Span Query API）。核心算法包括：(1) **Commutativity Hypothesis**：若两个 span 不相互依赖（order 无关），它们的 KV cache pages 可任意排列和复用——"AB=BA → Commutativity"；(2) **三层优化栈**：high-level tree rewriting optimizer（固定点重写）、query tokenization（将 expression tree 编码为带特殊 token 的线性序列）、low-level optimizer（block alignment + trailing partial block cropping）；(3) **CIDRA 算法**（Concurrent In-place Duplicating ReROPE Algorithm）——构建 block repositioning 依赖图，SCC 分析 + GPU bin packing + duplicate-on-demand（出度 >1 的 block 被复制），最大吞吐 500 tokens/ms；(4) **Dual-output paradox resolution**——通过"plus distribution"重写规则将 commutative join 分布到 token generation，使模型服务向 client 的输出与向 KV cache 的写入结构一致。
  实验比较：(1) Span query stack cache hit vs cache miss 的 TTFT（RAG microbenchmark，1-32 个各 2857 token 的 document，每配置 10 次测量）；(2) Span query optimized vs unoptimized 的 needle-in-haystack accuracy（2B vs 8B granite3.3 模型，1000 runs/query variant）；(3) Bulk execution TTFT speedup（2Wiki 和 NaturalQuestions 数据集，bulk size 1-1024-whole corpus）；(4) Nested generation TTFT reduction（Judge-Generator pattern，fan-out 1-24，100-500 executions/configuration）。

- 硬件平台是什么，配置是什么。
  论文未明确说明 GPU 型号。实验使用 granite3.3 模型（2B 和 8B 参数）。RAG microbenchmark 使用 2857 token documents。Bulk execution 在 2Wiki（Ho et al., 2020，multi-hop QA）和 NaturalQuestions（Kwiatkowski et al., 2019）上评测，fragment 约 100 tokens。

- 模型是什么。数据集和bench分别是什么。
  模型：granite3.3（2B 和 8B 参数）。
  数据集/Benchmark：(1) RAG microbenchmark——1-32 个各 2857 token 的 documents；(2) 2Wiki——multi-hop QA dataset (Ho et al., 2020)；(3) NaturalQuestions——(Kwiatkowski et al., 2019)；(4) Needle-in-haystack——inner generate 产生随机 names（needles）穿插随机内容（hay），judge 提取 names，1000 runs/query variant；(5) MSMARCO 和 HotpotQA——accuracy evaluation with span table implementation（slide 中提及，细节未展开）。
  评估指标：TTFT（Time-to-First-Token）、KV cache hit rate、repositioning throughput（tokens/ms）、needle-in-haystack accuracy（correct extraction fraction）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/IBM/spnl（SPNL library），Rust crate + Python package + CLI + Docker。arXiv: 2511.02749（CC BY 4.0）。
  
  算法 pipeline（Span Query 从 expression tree 到 token sequence 的全过程）：
  
  **Phase 1: High-level Optimization（固定点树重写）**
  ```
  输入: Span Query expression tree T（含 commutativity annotations）
  
  # 四条重写规则（固定点迭代直到收敛）：
  while changed:
      # Rule 1: Desugar Chat
      # C(prompt, response) → G(prompt) + ⨝(response)
      # Chat 展开为 generator + join
      
      # Rule 2: Desugar RAG  
      # R(query, docs) → ++(fold(query, docs))
      # RAG 展开为 concat + fold
      
      # Rule 3: Simplification
      # 合并连续的 commutativity hints 链
      # ++(++(a,b), c) → ++(a,b,c)
      
      # Rule 4: Plus Distribution（解决 dual-output paradox）
      # G(p) + ⨝(r1,r2) → ⨝(G(p)+r1, G(p)+r2)
      # 将 commutative join 分布到 generator 两输出路径
  ```
  
  **Phase 2: Query Tokenization（expression tree → token sequence）**
  ```
  输入: 优化后的 expression tree T
  特殊 token: □(pad), ((subtree start), )(sibling boundary), )n(subtree end+pos ptr)
  
  def tokenize(node):
      if node is Leaf(token_seq):
          return [□]*pad_to_block(token_seq) + token_seq
      elif node is CommutativeJoin(children):
          # Span: contiguous sequence of pages → any order OK
          result = [<bos>]
          for i, child in enumerate(children):
              if i > 0: result.append(<sibling_sep>)   # )(
              result.extend(tokenize(child))
          result.append(<subtree_end>)                  # )n
          # )n 中的 n 指向该序列化子树的起始位置
          # vLLM 可直接从 token 序列定位而无需解析栈
          return result
      elif node is Sequential(children):
          result = []
          for child in children:
              result.extend(tokenize(child))
          return result
  ```
  
  **Phase 3: Low-level Optimization（block 级别）**
  ```
  # Block Alignment: 特殊 token 对齐到 block 边界
  # 仅每 block 首 token 需扫描（hash chaining 决策）
  for special_token in [<bos>, <subtree_start>, <sibling_sep>, <subtree_end>]:
      pad_to_block_boundary(special_token)
  
  # Trailing Partial Block Crop:
  # vLLM 不缓存 partial blocks → crop inner generate 尾部
  for inner_gen_output in expression_tree:
      if len(inner_gen_output) % block_size != 0:
          crop_trailing = len(inner_gen_output) % block_size
          inner_gen_output = inner_gen_output[:-crop_trailing]
  ```
  
  **Phase 4: CIDRA（Concurrent In-place Duplicating ReROPE Algorithm）**
  ```
  输入: Block repositioning requests R = {(block_id, new_pos, request_id)}
  输出: Repositioned KV cache blocks
  
  # Step 1: Build dependency graph
  G = DirectedGraph()
  for each request r in R:
      # 若 r 要将 block b 从 pos p_old 移到 p_new
      # 而另一请求 r' 需要 b 在 p_old（原位置）
      G.add_edge(r, r')  # r 依赖 r'（r' 必须先读 b 然后 r 才能移动）
  
  # Step 2: SCC analysis
  sccs = tarjan(G)  # 识别循环依赖
  for scc in sccs:
      if len(scc) > 1:  # 循环依赖
          if len(scc) <= threshold:  # 通常 size 2
              # CPU 回退（处理循环）
              cpu_reposition(scc)
          else:
              # 大循环罕见，duplicate 打破循环
              duplicate_and_resolve(scc)
  
  # Step 3: Duplicate blocks with out-degree > 1
  for block b in G:
      if G.out_degree(b) > 1:
          # 多个并发请求将同一 block repo 到不同位置
          duplicate_block(b, count=G.out_degree(b))
  
  # Step 4: GPU bin packing + parallel execution
  independent_subgraphs = topological_sort(sccs)
  for subgraph in independent_subgraphs:
      gpu_bins = bin_pack(subgraph.blocks, gpu_memory)
      parallel_launch(gpu_bins)  # 并行执行 repositioning
  
  # 小 batch: concatenate layers
  if batch_size < threshold:
      concatenate_layers_for_single_kernel()
  
  # Max throughput: 500 tokens/ms
  ```
  
  **关键设计洞察**：
  1. **Commutativity 的形式化**：Span query 引入 `++`（commutative join）操作符——`++(A, B)` 表示 A 和 B 可任意顺序排列。Chat input 不 commute（"你好"→"我很好" vs "我很好"→"你好" 语义不同），RAG documents 之间 commute（document order 不影响 relevance）。
  2. **RoPE on read 的动机**：若 RoPE 在 KV cache 写入时施加（vLLM 默认），同一 page 被不同上下文的请求复用时 position encoding 错误。RoPE on read 使 position encoding 在 attention 计算时动态注入，page 本身 position-free。
  3. **Hash chaining selective disable**：Prefix caching 依赖 block hash（hash of token content + prior block hash）识别相同 prefix。Span 内 pages 顺序无关 → hash 不能链式依赖前序 block → 以 `(` 开头的 block 暂停 hash accumulation，以 `)n` 开头的 block 恢复。这使 prefix caching 在 span 粒度而非 token 粒度工作。
  4. **Dual-output paradox**：模型 server 向 client emit token stream（一种顺序）和向 KV cache 写入（另一种顺序）——当 query 含 commutative join 时，两种顺序可能不同。Plus distribution 重写确保两者结构一致。
  5. **Lost-in-the-middle 的解决**：Span query 将 attention 限制在 document 内（sparser attention），避免长上下文中 attention mass 分散到无关区域。注意优化后的 2B span query 在 needle-in-haystack 上 accuracy 超过 stock 8B 模型。

## SageBwd

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 SageBwd，一种可训练的 INT8 低比特注意力机制。SageBwd 将 7 个注意力矩阵乘法中的 6 个量化为 INT8：(1) 前向 pass——QKᵀ 使用 per-block INT8 量化（配合 K-smoothing 消除 key 通道异常值），PV 使用混合 per-token/per-block INT8 量化；(2) 反向 pass——dP = dOVᵀ 保持 FP16 精度不量化，其余 4 个 MatMul（dV = PᵀdO、dS 计算、dQ = dSK、dK = dSᵀQ）全部使用 per-block INT8 量化。保留 dP 为 FP16 是核心设计选择，因为 dP 的量化误差会通过 dS = P∘(dP−δ) 路径严重放大。
  实验比较：SageBwd vs Full-Precision Attention（FPA = FlashAttention2），在 325M Llama 模型预训练中，对比 loss 曲线（78B tokens）、中间张量 cosine similarity/relative ℓ² error（δ, P, dP, dS, O, dQ, dK, dV）、Kernel 吞吐量（TOPS/s）以及不同 QK-norm、K-smoothing、Q-smoothing 设置下的消融实验。

- 硬件平台是什么，配置是什么。
  NVIDIA B200 或 RTX 4090 单 GPU 节点。所有实验使用 BF16 混合精度训练。Kernel 吞吐量基准在 RTX 4090 上测试，head dim D=64 和 D=128。

- 模型是什么。数据集和bench分别是什么。
  模型：325M Llama（hidden dim=3072, context length=4096, GPT2 tokenizer, norm epsilon=1e-6）。
  数据集：OpenWebText，训练 78B tokens。
  Benchmark 指标：预训练 loss（cross-entropy）、中间张量 cosine similarity 和 relative ℓ² error（δ, P, dP, dS, O, dQ, dK, dV vs FPA）、Kernel 吞吐量（TOPS/s）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源仓库：https://github.com/thu-ml/SageAttention（预计 2025 年 7 月 15 日左右开源 SageAttention3/SageBwd 代码）。实现基于 OpenAI Triton。
  
  算法 pipeline（SageBwd 前向+反向，单头 attention 为例）：
  
  **前向（Algorithm 1）**：
  输入 Q,K,V ∈ R^{N×d}，block size B_q, B_kv。
  ```
  1. K_sm = K - mean_row(K)                // K-smoothing
  2. 分块：Q → {Q_i} (T_m=N/B_q blocks), K_sm → {K_j}, V → {V_j} (T_n=N/B_kv blocks)
  3. Per-block INT8 量化: {s_Q, Q̂_i}=ψ(Q_i), {s_K, K̂_j}=ψ(K_jᵀ), {s_V, V̂_j}=ψ(V_j)
  4. for i=1..T_m:                          // FlashAttention tiling
       for j=1..T_n:
         S_ij = Q̂_i × K̂_j × s_Q × s_K       // INT8 MatMul for QKᵀ
         m_ij = max(m_i,j-1, rowmax(S_ij))  // online softmax
         P̃_ij = exp(S_ij - m_ij)
         s_P = exp(rowmax(S_ij)-m_ij)/127   // per-token scale
         P̂_ij = P̃_ij / s_P                  // per-token INT8 quantize P
         O_ij = diag(e^{m_i,j-1-m_ij})⁻¹ O_i,j-1 + P̂_ij × V̂_j × s_P × s_V  // INT8 MatMul for PV
       O_i = diag(l_i,T_n)⁻¹ O_i,T_n
       L_i = m_i,T_n + log(l_i,T_n)
  5. return O={O_i}, L={L_i}
  ```
  
  **反向（Algorithm 2）**：
  输入 {s_Q,Q̂_i}, {s_K,K̂_j}, {s_V,V̂_j}, O, {L_i}, dO ∈ R^{N×d}。
  ```
  1. D = rowsum(dO ∘ O)                     // δ = rowsum(dO∘O)
  2. for j=1..T_n:
       for i=1..T_m:
         S_ij = Q̂_i × K̂_j × s_Q × s_K      // recompute S from forward quantized Q,K
         P_ij = exp(S_ij - L_i)             // recompute P
         s_P, P̂_ij = ψ(P_ij)                // per-block INT8 quantize P
         s_dO, dÔ_i = ψ(dO_i)               // per-block INT8 quantize dO
         dV_j += P̂_ijᵀ × dÔ_i × s_P × s_dO  // INT8 MatMul: dV = Pᵀ dO
         dP_ij = dO_i × V_jᵀ                 // *** FP16, NOT quantized ***
         dS_ij = P_ij ∘ (dP_ij - D_i)        // softmax gradient
         s_dS, dŜ_ij = ψ(dS_ij)              // per-block INT8 quantize dS
         dQ_i += dŜ_ij × K̂_j × s_dS × s_K   // INT8 MatMul: dQ = dS × K
         dK_i += dŜ_ijᵀ × Q̂_i × s_dS × s_Q  // INT8 MatMul: dK = dSᵀ × Q
  3. return dQ, dK, dV
  ```
  
  关键设计：(a) dP 保持 FP16 不量化，避免了量化误差经 dS = P∘(dP−δ) 传播到 dQ,dK；(b) K-smoothing 在 kernel 入口执行，无需修改反向 pass（因为 dS 每行和为 0，dQ=dSK=dSK^{sm} 直接成立）；(c) QK-norm（RMSNorm on Q,K）控制激活动态范围，减少 INT8 量化步长。

## SLA2

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 SLA2（Sparse-Linear Attention with Learnable Routing and QAT），一种可训练的稀疏-线性混合注意力方法，用于加速 Diffusion Transformer（DiT）中的 attention 计算。核心包括三部分：(I) 可学习路由器 R(Q,K)：基于压缩后的 Q、K 通过可学习投影 proj_q、proj_k 得到路由分数，经 Top-k 选择哪些位置走稀疏 attention（M=1）、哪些走线性 attention（M=0）；(II) 直接 α-组合公式 O = α⊙O_s + (1-α)⊙O_l，其中 α ∈ R^{N×1} 可学习，消除了原 SLA 中稀疏分支的行归一化缩放失配问题，无需额外的 Proj(O_l) 补偿；(III) QAT（量化感知训练）：前向使用低比特 attention（INT8/FP8 量化 Q,K,P,V），反向保持 FP16 精度，使模型在训练中适应量化误差。
  实验比较：SLA2 vs Full Attention（FlashAttn2）、SLA、VSA、VMoBA，在 Wan2.1-T2V-1.3B-480P 和 Wan2.1-T2V-14B-720P 上，sparsity 85%/90%/95%/97% 下的视频生成质量（VBench 五维度 + Vision Reward）和计算量（FLOPs）。

- 硬件平台是什么，配置是什么。
  NVIDIA RTX 5090（具体 VRAM 量论文未明确说明，但 Wan2.1-14B-720P 超出单卡显存，评测时启用 sequential CPU offloading，报告延迟已排除 offload 开销）。FP16 精度训练和推理，低比特 attention 使用 INT8 或 FP8 量化。

- 模型是什么。数据集和bench分别是什么。
  模型：Wan2.1-T2V-1.3B（480P 分辨率）和 Wan2.1-T2V-14B（720P 分辨率），均为视频 DiT 模型。
  数据集：私有视频数据集，约 3000 个视频（每个约 5 秒），从公开来源收集。使用 Qwen3-VL-Flash 为每个视频生成 caption 作为 text-video pair，用于微调和评测。
  Benchmark 指标：VBench（Imaging Quality IQ、Overall Consistency OC、Aesthetic Quality AQ、Motion Smoothness MS、Subject Consistency SC）、Vision Reward (VR) 人类偏好评分、FLOPs、kernel 速度 C/t（C=4N²d）、端到端推理延迟。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  SLA2 代码未开源。论文作者 Jintao Zhang 的 SLA 系列代码在 https://github.com/thu-ml/SLA（含 SLA v1），SLA2 被同一仓库的 citation 引用但尚未包含代码。Baseline 方法（SLA、VSA、VMoBA）均使用官方开源实现。
  
  算法 pipeline（SLA2 前向计算，以单头 attention 为例）：

  **输入**：Q, K, V ∈ R^{N×d}，block size b_q, b_k，sparsity k%，可学习参数 proj_q, proj_k ∈ R^{d×d}, α ∈ R^{N/b_q×1}

  **Step 1 — 压缩与路由**：
  ```
  Q̄ = pool(Q) ∈ R^{N/b_q×d}      // mean pooling over b_q tokens
  K̄ = pool(K) ∈ R^{N/b_k×d}      // mean pooling over b_k tokens
  P_c = softmax(proj_q(Q̄) proj_k(K̄)ᵀ / √d)
  M_c = Top-k(k%, P_c)            // per-row top-k% → 1, rest → 0
  ```
  M_c ∈ {0,1}^{N/b_q × N/b_k}，展开得 M ∈ {0,1}^{N×N}。

  **Step 2 — 稀疏 attention（M=1 位置）**：
  ```
  S_sparse = QKᵀ/√d ⊙ M            // 仅计算 M=1 位置
  P = softmax(S_sparse)            // 仅在选中位置归一化
  O_s = P V                        // sparse attention 输出
  ```
  使用 FlashAttention 风格 block-wise 实现，跳过 M=0 位置的计算。

  **Step 3 — 线性 attention（M=0 位置）**：
  ```
  Q^φ = φ(Q), K^φ = φ(K)          // φ = softmax 作为 kernel 函数
  H = (K^φ)ᵀ((1-M)⊙V)              // 先算 KᵀV，O(Nd²) 而非 O(N²d)
  Z = rowsum((K^φ)ᵀ(1-M))          // 归一化因子
  O_l = Q^φ H / (Q^φ Z)            // 线性 attention 输出
  ```

  **Step 4 — α-组合**：
  ```
  O = α ⊙ O_s + (1-α) ⊙ O_l
  ```
  其中 α 在 Stage 1 用 MSE(FullAttn(Q,K,V), SLA2(Q,K,V)) 初始化，Stage 2 与扩散模型端到端微调。

  **QAT 前向（低比特 attention）**：
  ```
  Q̂, s_Q = quant(Q); K̂, s_K = quant(K)     // FP16→INT8/FP8
  S = dequant(Q̂ K̂ᵀ / √d, s_Q, s_K)
  P = softmax(S ⊙ M)
  P̂, s_P = quant(P); V̂, s_V = quant(V)
  O_s = dequant(P̂ V̂, s_P, s_V)
  ```
  量化方案遵循 SageAttention2++。反向传播全程 FP16，使用原始 Q,K,V 和 O_s 计算梯度。

  **训练两阶段**：
  - Stage 1：用各 attention 层各 diffusion timestep 的 Q,K,V 构造数据集 D，MSE loss 训练 R 和 α（使用可微分 SoftTop-k 替代 Top-k）
  - Stage 2：替换扩散模型 attention 为 SLA2，端到端微调全部参数 Θ 和 α（R 参数固定，使用硬 Top-k）

  **关键参数**：b_q=128, b_kv=64, k% ∈ {5%,4%,3%} 对应 sparsity 85%/90%/95%/97%, τ=0.1（SoftTop-k 温度）, 500 steps, batch 64 (1.3B) / 15 (14B)。

## Inference Time Context Sparsity

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是在推理时对 LLM 的注意力机制沿 context 维度施加稀疏化：decode 阶段每个 query 仅选取 KV cache 中 top-k 个 token 参与 attention 计算，而非全量 dense attention。稀疏选择算法包括：(1) Oracle top-k（精确 top-k 作为上界，消除近似索引器混淆因素）；(2) vAttention 随机索引选择（基于采样的随机稀疏化）；(3) Double Sparsity 索引器（8 通道 16-bit 精度量化）。稀疏模式为 per-token、per-query、per-head 级别，不强制块结构。
  实验比较：密集 attention（dense）vs 稀疏 attention 在 5×、10×、50×、100×、250×、500× 等稀疏度下的模型质量。评测 20 个模型、5 个模型家族（Qwen2.5、Qwen3.5 hybrid、Gemma3 hybrid、Ministral3、Llama3），覆盖 RULER-HARD-32K、LOFT-32K/128K、AIME2025（长生成）、SWE-Bench Django（agentic coding）四种任务类型。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 80GB HBM3 GPU，FP16 精度。kernel benchmark 在单 H100 上完成，GQA 配置 Hq=32、Hkv=8、D=128、page size 16、NHD layout、128K context。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5（0.5B/1.5B/3B/7B/14B/32B/72B）、Qwen3.5 hybrid（0.8B/2B/4B/9B/27B，含线性注意力层）、Gemma3 hybrid（1B/4B/12B/27B，滑动窗口注意力）、Ministral3（3B/8B/14B）、Llama3（8B/70B），共 20 个模型。
  数据集/Benchmark：(1) RULER-32K-HARD（6 个子任务：fwe, qa1, qa2, vt, nm2, nm3）；(2) LOFT（5 个数据集：hotpotqa, nq, musique, qampari, quest，32K 和 128K 两个 context 长度）；(3) AIME 2025（数学推理，允许最长 65K token 生成，实际平均 ~25K token）；(4) SWE-Bench Lite Django 子集（114 个真实 GitHub issue）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/skylight-org/sparse-attention-hub（Apache 2.0），基于 HuggingFace Transformers + Ray Tune。
  算法 pipeline（decode 阶段稀疏 attention）：

  ```
  # 输入: Q ∈ R^{Hq×d}, K_cache ∈ R^{N×Hkv×d}, V_cache ∈ R^{N×Hkv×d}
  # 稀疏 mask 由三部分 additive 组合: Sink + Local + TopK

  # 1. 构建稀疏 mask
  mask = zeros(N)                    # N = context length
  mask[0:128] = 1                    # Sink tokens (前128个永远保留)
  mask[pos-128:pos] = 1             # Local window (当前token前128个)

  # 2. 索引器选 top-k
  scores = indexer(Q, K_cache)       # Double Sparsity: 8×16-bit 量化通道计算近似分数
  topk_indices = topk(scores, k)     # k = N / sparsity_ratio, k << N
  mask[topk_indices] = 1

  # 3. 稀疏 attention (仅选中 token)
  K_sparse = K_cache[mask]           # shape: (k', Hkv, d), k' = |mask|
  V_sparse = V_cache[mask]
  attn_scores = Q @ K_sparse^T / sqrt(d)   # (Hq, k')
  attn_weights = softmax(attn_scores)
  output = attn_weights @ V_sparse          # (Hq, d)
  ```

  核心洞察：当 d << N 时，dense attention 的信息实际被 hidden dimension 瓶颈压缩（Theorem 1: 需要 d ≥ N-1 才能无损区分所有 attention 分布），因此极端稀疏并非近似而是更优目标。实验证明 hybrid 架构模型（Qwen3.5-27B）在 50× 稀疏下 RULER-HARD 和 AIME2025 质量无损，100× 稀疏下仍保持竞争力。

## FuseFlow

- 属于算法pipeline的实现是什么？实验比较什么？
  - **实现**：FuseFlow 提出的融合策略和数据流优化构成了稀疏 ML 推理的**算法级 acceleration pipeline**，核心包括：
    1. **交叉表达式核融合（EKF）算法**：通过 index substitution + POG 自动融合跨多个稀疏 Einsum 的迭代空间。该算法决定了哪些中间张量被物化到内存、哪些被流式传递——本质上是 sparse-aware 的 fusion-recomputation tradeoff 决策算法。支持 unfused、partially fused、fully fused 三种粒度。
    2. **Factored Iteration（分解迭代）vs Global Iteration（全局迭代）**：FuseFlow 选择 factored iteration 作为默认策略——将 n 维全局迭代空间分解为多个 pairwise 迭代子空间（每个二元操作一个），牺牲部分省略冗余计算的机会，但大幅降低 coordinate processing overhead（如图 5 所示）。这是稀疏特有的算法设计选择：在稠密计算中，global iteration 的坐标开销可忽略，但在高维稀疏张量中可能导致 coordinate explosion。
    3. **Dataflow Ordering 选择**：对每个稀疏 tensor algebra 表达式提供多种等效 dataflow order（如 Gustavson vs inner product），不同 order 对应不同的渐进复杂度。FuseFlow 的 POG 枚举所有不破坏融合的有效 order，用户选择最优。
    4. **Fusion Heuristic**：基于 FLOPs/bytes 的快速分析模型，根据 tensor 维度、稀疏度和 intersection rates 估算给定融合配置的计算和内存成本，用于在 exhaustive 仿真前 prune suboptimal 配置。
  - **实验比较**：
    - **Fusion 粒度对性能的影响**（图 12）：GPT-3 在 full fusion 下获 ~2.7× speedup，而 GCN 和 GraphSAGE 在 full fusion 下性能退化（因 nested matmul 的 recomputation 开销超过内存收益），SAE 在 partial fusion 下几乎无增益（~1.01×）但在 full fusion 下获 ~1.94×。
    - **Operational Intensity 分析**（图 14）：全融合增加 operational intensity（减少内存传输），但同时增加 FLOPs（因 recomputation），验证了 fusion-recomputation tradeoff 的存在。
    - **Sparsity 消融**（图 15）：partial fusion 的 speedup 随稀疏度增加而增加；structured sparsity patterns（power-law, block diagonal）因更好的 locality 优于 uniform random。
    - **Dataflow Order sweep**（图 18）：nested matmul 的最优 vs 最差 dataflow order 性能差 ~29×。每个 kernel 选择最优 order + 施加局部约束可将搜索空间缩减 68.5%-99.9%。

- 硬件平台是什么，配置是什么。
  - **模拟器**：Comal cycle-accurate dataflow simulator（基于 DAM framework [81]，Rust 编写），集成 HBM2 内存模拟（Ramulator 2.0 [48]）。
  - **FPGA 验证**：Xilinx VU9P (AWS F1)，通过 Vitis HLS 综合 SAMML 图并验证模拟器保真度（$R^2=0.991$）。
  - **实际芯片参考**：论文提及 Onyx [42]（12nm CGRA）和 Opal [10]（16nm CGRA SoC）作为 SAM/SAMML 编译目标的实际数据流加速器。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - GCN [35]：2-layer，隐藏维度 128
    - GraphSAGE [26]：2-layer
    - Sparse Autoencoder (SAE) [51]：3-layer
    - GPT-3 Small (125M params) with BigBird attention [79]：sequence length 1024
  - **数据集**：
    - Cora [76] (2708×1433, 99.7% sparse)
    - Cora_ML [7] (2995×2879, 99.8% sparse)
    - DBLP [7] (17716×1639, 99.6% sparse)
    - OGB-Collab [34] (235868×128, 99.9% sparse)
    - OGB-MAG [34] (1939743×128, 99.9% sparse)
    - ImageNet [16] (224×224, 50% sparse, SAE 用)
    - NIH-CXR [69] (1024×1024, 50% sparse, SAE 用)
    - LUNA16 [63] (512×512, 50% sparse, SAE 用)
    - IMDB [7] (GPT-3 BigBird 用, mask sparsity 53.9%-86.5%)
    - KarateClub [78] (dataflow ordering 消融实验用)
  - **稀疏类型**：zero-based lossless（GCN/GraphSAGE 的 adjacency matrix）、zero-based lossy（SAE 的 weight pruning）、zero-based lossy masked activation（GPT-3 BigBird attention mask）。
  - **Benchmark 指标**：latency in cycles (normalized speedup vs unfused baseline)、FLOPs count、bytes transferred、operational intensity、parallelization speedup、search space size。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：GitHub fuseflow-artifact 仓库 + Figshare DOI（reserved），Docker 部署，MIT License。
  - **算法 pipeline 伪代码**：
    ```
    算法：FuseFlow Cross-Expression Fusion (EKF)
    输入: Einsum expressions E = {E1, E2, ..., En} within a Fuse{} region,
          Mode orders M (per-tensor storage format constraints),
          User dataflow orders D (optional)
    输出: Fused Einsum subgraphs + Partial Order Graph (POG)

    POG = DirectedGraph()  // 偏序图，节点=索引变量，边=顺序约束

    for each expression E_i in E:
        // Step 1: Rename local reduction indices
        for each reduction index r in E_i:
            r' = fresh_index()
            replace r with r' in E_i
            // 添加该张量 view 的 mode order 约束
            for each tensor view v using r':
                for each pair (a, b) in v.mode_order where a precedes b:
                    POG.add_edge(a, b)

        // Step 2: Producer-Consumer fusion via index substitution
        for each tensor T in E_i where T = producer_result:
            consumer_E = find_expression_using(T)
            substitute(T.indices, consumer_E.corresponding_indices)

        // Step 3: Propagate order constraints
        for each (producer_idx, consumer_idx) mapping:
            POG.add_edge(producer_idx.outer, consumer_idx.inner)
            // 传递性：如果 i→k 且 k→j，则 POG 中 i→j

        // Step 4: Handle multiple tensor uses
        for each tensor T used in multiple expressions:
            views = create_distinct_views(T)
            for each v in views:
                v.annotate(T.mode_order)
            if detect_cycle(POG):
                // 物化 permuted copy 打破循环
                materialize_transposed_copy(T, conflicting_view)

    // 检测 POG 是否无环
    if POG.has_cycle():
        error("Fusion constraints unsatisfiable")
    else:
        valid_orders = POG.topological_sorts()
        return fused_einsums, valid_orders
    ```

    **Factored Iteration vs Global Iteration 的伪代码对比**（对应图 5）：
    ```
    // 论文示例表达式: D_{il} = A_{ik} B_{kj} C_{jl}

    // Global Iteration (Custard/Stardust 默认):
    //   4 维迭代空间，坐标处理开销随维度指数增长
    for i in I:
      for k in K:
        for j in J:
          for l in L:
            D[i,l] += A[i,k] * B[k,j] * C[j,l]

    // Factored Iteration (FuseFlow 默认):
    //   两个 3 维子空间交错执行，坐标开销限于二元操作
    for i in I:
      for k in K:
        for j in J:
          E[i,j] += A[i,k] * B[k,j]   // 子空间 1: i-k-j
      for j in J:
        for l in L:
          D[i,l] += E[i,j] * C[j,l]   // 子空间 2: i-j-l (E_{ij} 被流式传递)
    ```

    **Fusion Heuristic 原理**：
    ```
    输入: tensor_dimensions[], sparsity_percentages[], intersection_rates[]
    输出: estimated_FLOPs, estimated_bytes

    estimated_FLOPs = sum(compute_per_op * nonzero_ratio)
    estimated_bytes = sum(io_tensor_sizes * sparsity_factors)
    operational_intensity = estimated_FLOPs / estimated_bytes

    // 如果 operational_intensity 低于阈值或 FLOPs 显著增加（因 recomputation），
    // 标记该配置为 suboptimal 并 prune
    if operational_intensity < threshold or FLOPs_growth > limit:
        prune(configuration)
    ```: Curvature-Aware Gradient Estimation for Quantization-Aware Training

- 属于算法pipeline的实现是什么？实验比较什么？
  - **实现**：CAGE（Curvature-Aware Gradient Estimation）是一种新的 QAT 梯度估计方法。它在标准 Straight-Through Estimator (STE) 梯度基础上增加了一个曲率感知的修正项 `λ_t * (x_t - Q(x_t))`（即量化误差作为修正信号），从而对抗量化引起的损失上升。CAGE 有两个变体：**Coupled CAGE**（修正项加到梯度上再传入优化器）和 **Decoupled CAGE**（修正项直接加到优化器的更新量 Δ_t 上）。CAGE 是 **optimizer-agnostic**（支持 AdamW、Muon、Shampoo 等）和 **quantizer-agnostic**（支持任意量化器，论文默认使用 QuEST INT 量化器）。CAGE 包含一个 **silence period**（沉默期）机制：训练前 s 比例的步数 λ_t=0（仅 STE），之后 λ_t 线性 ramp-up。
  - **实验比较**：CAGE vs QuEST（ICML 2025，此前 SOTA QAT 方法）vs BF16 baseline，在 Llama 系列模型（30M/50M/100M/200M/430M/800M/1700M/3200M）上进行 QAT pretraining（W4A4、W3A3）和 QAT fine-tuning（Llama 3.1-8B W4A16、Llama-3.2-3B）。评估指标包括 validation loss、perplexity、RULER score。还比较了 Coupled vs Decoupled CAGE、silence ratio 和 λ 系数的影响。

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA H100 GPU。训练 100M 模型 W4A4 时 per-iteration 耗时约 101-105ms（CAGE_D overhead 极小，相比 QuEST 几乎无额外开销）。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Llama-style 模型，参数规模 30M、50M、100M、200M、430M、800M、1700M、3200M。Fine-tuning 实验使用 Llama 3.1-8B 和 Llama-3.2-3B（Tulu-SFT）。
  - **数据集/benchmark**：论文未明确说明预训练数据集的具体名称（仅提及 token count D 作为 scaling law 参数）。Fine-tuning 评估使用 RULER benchmark（long-context evaluation，平均分数越高越好）。Baseline 比较还包括 GPTQ（W4A16）和 RTN（W4A16）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：https://github.com/IST-DASLab/CAGE（MIT License）
  - **依赖**：Python 3.11, CUDA 12.6, PyTorch 2.6
  - **使用示例**：
    ```bash
    # W4A4 QAT pretraining with CAGE on 200M Llama model
    bash train.sh --model-size-prefix=200M --w-bits=4 --a-bits=4 --cage-lambda=15 --cage-silence-ratio=0.7
    ```
  - **算法伪代码（CAGE + AdamW）**：
    ```
    输入: 初始参数 x_0, 总步数 T, AdamW超参 (β1,β2,α,ω,ε),
          量化器 Q (如 QuEST INT), CAGE系数 λ, 沉默比例 s

    初始化: m_{-1}=0, v_{-1}=0

    for t = 0, 1, ..., T-1:
        r_t = (t+1)/T                          # 训练进度比例
        if r_t ≤ s:
            λ_t = 0                            # 沉默期: 仅STE
        else:
            λ_t = λ * (r_t - s) / (1 - s)      # 线性ramp-up

        采样 minibatch, 执行量化前向传播
        e_t = x_t - Q(x_t)                     # 量化误差 (无梯度)
        g_t = ∇̃f(x_t) + λ_t * e_t             # Coupled: 梯度 + 曲率修正

        # Decoupled weight decay
        x_t = (1 - α*ω) * x_t

        # AdamW 更新
        m_t = β1*m_{t-1} + (1-β1)*g_t
        v_t = β2*v_{t-1} + (1-β2)*g_t⊙g_t
        m̂_t = m_t / (1-β1^t)
        v̂_t = v_t / (1-β2^t)

        # Decoupled CAGE: 修正项直接作用于更新量
        Δ_t = m̂_t / (√v̂_t + ε) + λ_t * e_t
        # (Coupled CAGE: Δ_t = m̂_t / (√v̂_t + ε), 修正已在 g_t 中添加)

        x_{t+1} = x_t - α * Δ_t               # 更新master weights
    return x_T
    ```
  - **核心张量计算**：CAGE 的关键操作是 `g_t = STE_gradient + λ_t * (x_t - Q(x_t))`，其中 `x_t - Q(x_t)` 是浮点 master weight 与量化后权重的逐元素差值，作为量化误差信号直接修正梯度方向。该修正项的计算开销极低（≈ 一次逐元素减法+标量乘法），因此论文称其为 near-cost-free。
  - **关键超参**：`--cage-lambda`（默认 10，控制修正强度）、`--cage-silence-ratio`（默认 0.8，前 80% 训练步数不施加修正）、`--cage-schedule`（linear_ramp 或 constant）。
  - **最优结果**：CAGE W3A3 的 validation loss 低于 QuEST W4A4；在 Llama 3.1-8B W4A16 fine-tuning 上 RULER 平均分 73.2（QuEST 68.7, GPTQ 65.1, RTN 41.5）；CAGE 将 QAT fine-tuning 的压缩精度损失减半。

---

## FlashAttention-4

- 属于算法pipeline的实现是什么？实验比较什么？
  - **实现**：FlashAttention-4 是一种针对 NVIDIA Blackwell GPU 非对称硬件扩展的 attention 算法与 kernel 协同设计。核心算法创新包括：
    1. **软件指数函数模拟**：使用 FMA 单元上的多项式近似（degree-3，Sollya 优化系数，Horner 方法求值）结合 IEEE 754 位操作实现 $2^x$ 的软件模拟，部分替代硬件 MUFU.EX2（仅 16 ops/clock/SM），将指数吞吐量瓶颈分散到吞吐量更高的 FMA 单元。对每行 10-25% 的条目使用模拟，其余使用硬件 MUFU。BF16 精度下 degree-3 多项式与硬件 MUFU 误差无差异（量化误差 3.9×10⁻³ 主导）。
    2. **条件 softmax rescaling**：修改在线 softmax 算法，仅当行最大值增量 $m_j - m_{j-1} > \tau$（$\tau = \log_2(256) = 8.0$）时才执行 $O_j = e^{m_{j-1}-m_j} O_{j-1} + e^{S_j-m_j} V_j$，否则跳过 rescaling（$O_j = O_{j-1} + e^{S_j-m_{j-1}} V_j$）。最终通过 $m_{\text{final}}$ 和 $\ell_{\text{final}}$ 统一归一化来保证正确性。显著减少非 MMA 的逐元素乘法操作。
    3. **前向流水线重设计**：ping-pong 双 Q tile 调度，softmax warpgroup（128 threads/warpgroup，每线程一整行，消除 inter-warp shuffle）与 MMA warpgroup 重叠执行。通过 TMEM 解耦 P 传输，将 output rescaling 分离到独立的 "correction" warpgroup。
    4. **后向 2-CTA MMA 模式**：利用 Blackwell 的 CTA pair 协同 MMA（M=256 tile），通过 DSMEM 跨 CTA 交换半个 dS tile 重排归约轴，将 $dQ$ 的全局 atomic add 次数减半。
    5. **LPT (longest-processing-time-first) CTA 调度**：causal masking 下按 reverse mblock 顺序；varlen 下预处理排序 batches。MHA causal 获 4-8% FLOPS 增益，MQA-8 获 7-14% 增益。
  - **实验比较**：FlashAttention-4 vs cuDNN 9.13 vs Triton 3.6 vs FlashAttention-2 vs Gluon vs PyTorch。Forward pass: FA4 比 cuDNN 快 1.1-1.3×，比 Triton 快 2.1-2.7×。Backward pass: FA4 在各序列长度和 causal masking 设置下均一致优于 baseline。确定性 backward 达非确定性 1-CTA 的 75% 性能。

- 硬件平台是什么，配置是什么。
  - **GPU**：NVIDIA B200 180GB SXM6 (1000W)，Blackwell 架构。
  - 关键参数：Tensor core BF16 8192 ops/clock/SM（Hopper 2×），MUFU 指数单元 16 ops/clock/SM（不变），SMEM 读带宽 128 bytes/clock/SM（不变），TMEM 256KB/SM（新增）。
  - **软件栈**：CUDA 13.1, FlashAttention 2.8.3, Triton 3.6, PyTorch 2.10.0, CuTe-DSL 4.4.1。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：论文使用标准 attention 算子 benchmark，不针对特定模型训练。benchmark 覆盖常见配置：head dimension 64/128/(192,128)（后者对应 DeepSeek V3 架构），序列长度 1k-32k，hidden dimension 2048（16 或 32 heads）。支持 MHA、MQA、GQA 变体。
  - **数据集/bench**：纯算子级 microbenchmark（无下游任务数据集）。FLOPs 计算公式：Forward = $4 \times \text{seqlen}^2 \times \text{head\\_dim} \times \text{num\\_heads}$（causal 除以 2），Backward = Forward × 2.5。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute（permissive license）
  - **算法伪代码（条件 softmax rescaling + 软件指数模拟）**：
    ```
    输入: Q, K, V ∈ R^{N×d} (BF16), causal mask, τ = log₂(256) = 8.0
    参数: tile_M, tile_N (tile sizes), emulation_ratio = 0.1-0.25

    for each Q tile (M rows):
        m = -∞ (running max, size M)
        ℓ = 0  (running normalizer, size M)
        O = 0  (running output, M×d)
        for each KV tile (N columns):
            # Step 1: Compute S = Q @ K^T (MMA on tensor cores)
            S = Q_tile @ K_tile^T   # M×N, BF16 MMA

            # Step 2: Softmax with conditional rescaling
            m_new = rowmax(S)       # per-row maximum
            # Conditional rescaling:
            for each row i:
                if m_new[i] - m[i] > τ:   # need rescaling
                    scale = 2^(m[i] - m_new[i])          # = e^{m_old - m_new}
                    O[i,:] = scale * O[i,:]               # rescale previous output
                    P[i,:] = exp(S[i,:] - m_new[i])       # compute softmax numerator
                else:                                     # skip rescaling
                    P[i,:] = exp(S[i,:] - m[i])           # use old max
                m[i] = max(m[i], m_new[i])                 # update running max

            # Exponential computation (hybrid hardware + software):
            # - (1 - emulation_ratio) entries: use MUFU.EX2 hardware instruction
            # - emulation_ratio entries: software emulation via polynomial
            #   x_frac = x - floor(x)
            #   2^x_frac ≈ p₀ + p₁*x_frac + p₂*x_frac² + p₃*x_frac³  (Horner: FMA)
            #   result = (2^floor(x) as IEEE 754) | mantissa_bits(2^x_frac)

            # Step 3: Update normalizer (conditional)
            if rescaling occurred:
                ℓ = scale * ℓ + rowsum(P)   # rescale old normalizer
            else:
                ℓ = ℓ + rowsum(P)            # just add (old max unchanged)

            # Step 4: Compute output contribution
            O = O + P @ V_tile   # M×d, MMA (TS mode: A from TMEM, B from SMEM)

        # Final normalization after all KV tiles
        O = diag(1/ℓ) @ O       # renormalize by true normalizer
    return O
    ```
  - **Roofline 分析公式（前向）**：
    - $T_{\text{MMA}} = \frac{4MNd}{8192}$ cycles（MMA throughput 8192 ops/clock/SM）
    - $T_{\text{smem}} = \frac{3MNd}{8192}$ cycles（SS + TS MMA 操作数读取，128 bytes/cycle）
    - $T_{\text{exp}} = \frac{MN}{16}$ cycles（MUFU 16 ops/clock/SM）
    - 对 tile M=N=d=128：$T_{\text{MMA}}=1024$, $T_{\text{smem}}=768$, $T_{\text{exp}}=1024$（MMA + exp 为瓶颈）
    - 对 tile M=256,N=d=128：$T_{\text{MMA}}=2048$, $T_{\text{smem}}=1536$, $T_{\text{exp}}=2048$
  - **Roofline 分析公式（后向）**：
    - $T_{\text{MMA}} = \frac{10MNd}{8192}$ cycles（5 次 MMA）
    - $T_{\text{smem}} = \frac{4Md+3Nd+MN}{64} + \frac{MN}{64} + \frac{Md}{16}$ cycles
    - 对 M=N=d=128：$T_{\text{MMA}}=2560$, $T_{\text{smem}}=3328$, $T_{\text{exp}}=1024$（SMEM 为瓶颈，超出 MMA 约 30%）
    - 2-CTA 模式（M=256,N=d=128）：$T_{\text{smem}}=2688$（仅超出 MMA 约 5%）
  - **编译**：全部用 CuTe-DSL（嵌入 Python）编写，JIT 编译。Forward kernel 编译 2.5s（vs FA3 C++ 模板 55s，22× 加速），Backward kernel 1.4s（vs 45s，32× 加速）。

## MAC-Attention: Match-Amend-Complete Attention for Efficient Long-Context Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 MAC-Attention 算法——一种训练无关（training-free）、模型无关的 LLM 解码加速方案。核心思想是通过 Match-Amend-Complete 三阶段复用先前的 attention 计算结果，而非从 KV cache 重新计算完整 attention，从而将 decode 阶段的 attention 计算从 O(N) 降为常数复杂度（在匹配命中时）。
  具体三阶段：
  1. **Match**：对每个新 query Q_n，在一个大小为 κ（默认 512）的滑动窗口中用 pre-RoPE L2 距离找到最相似的历史 query Q_m，复用其缓存的 attention 输出 A_m。
  2. **Amend**：在匹配边界附近的一个小 band（宽度 r=256）上重新计算 attention，修正 softmax 质量集中在 decode 光标附近导致的误差。公式：A_n ≈ A_m ⊖ Attn(Q_m, K_{j~m}, V_{j~m}) ⊕ Attn(Q_n, K_{j~n}, V_{j~n})。
  3. **Complete**：用数值稳定的 log-sum-exp merge 将 amended 结果与 KV tail 上的 fresh attention 融合。
  实验比较：MAC-Attention vs Full Attention（FlashInfer baseline）、Quest、RocketKV、Multipole 在相同 KV 访问比例（1%, 5%, 10%, 20%）下的质量（LongBench v2）和延迟。

- 硬件平台是什么，配置是什么。
  NVIDIA Hopper GPU（H100 级别），BF16 精度。CUDA 13.0 环境。单 GPU 评测（SGLang 服务端 + MAC-Attention kernel）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama 3.1 家族（主要验证模型），Qwen3-30B-A3B-Instruct（LongBench v2 质量评估）。
  数据集/Benchmark：(1) LongBench v2（120K context）；(2) RULER（120K context）；(3) LongGenBench（16K 连续生成）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/YJHMITWEB/MAC-Attention.git（MLSys 2026，ACM AE Badge 认证）。
  算法 pipeline（MAC-Attention decode 阶段）：

  ```
  # 全局状态
  Q_cache = []        # 滑动窗口 query cache，大小 κ（默认 512）
  A_cache = []        # 滑动窗口 attention output cache

  # 每个 decode step
  def mac_attention_decode(Q_n, K_cache, V_cache, n):
      # === Stage 1: Match ===
      # 在滑动窗口内用 pre-RoPE L2 距离匹配最相似 query
      Q_n_pre_rope = Q_n_before_rope  # pre-RoPE query
      best_idx, best_score = argmin(||Q_n_pre_rope - Q_cache[i]||_2) for i in range(|Q_cache|)
      if best_score > τ (threshold=0.45):
          return fallback_full_attention(Q_n, K_cache, V_cache)  # miss → fallback
      Q_m, A_m = Q_cache[best_idx], A_cache[best_idx]
      m = n - κ + best_idx  # 匹配 token 在序列中的绝对位置

      # === Stage 2: Amend ===
      # 在匹配边界附近 rectification band j 上重新计算
      j = max(m - r, 0)  # r = 256 (rectification band)
      # 从 A_m 中移除旧 band attention
      A_m_corrected = A_m ⊖ attention_update(Q_m, K_{j~m}, V_{j~m})
      # 加入新 band attention
      A_new_band = attention_compute(Q_n, K_{j~n}, V_{j~n})
      A_prefix = A_m_corrected ⊕ A_new_band  # online softmax merge

      # === Stage 3: Complete ===
      # 与 KV tail 上 fresh attention 融合
      A_tail = attention_compute(Q_n, K_{m~n}, V_{m~n})
      A_n = logsumexp_merge(A_prefix, A_tail)

      # === Cache Update ===
      Q_cache.append(Q_n_pre_rope)
      A_cache.append(A_n)
      if |Q_cache| > κ:  Q_cache.pop(0); A_cache.pop(0)
      return A_n
  ```

  **FLOPs 减少比例**：r_skip = 1 - r/N（当 κ ≥ m；r 与 context 长度 N 无关）。实际匹配命中率 ~99.5%，跳过 ~98.9% KV cache 访问。
  **复杂度**：匹配命中时计算和带宽复杂度为常数级别，与 context 长度无关。

  **兼容性**：与 chunked prefill、continuous batching、speculative decoding、PD disaggregation、MHA/GQA 均兼容。

## PuzzleMoE

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 PuzzleMoE 的稀疏专家合并（Sparse Expert Merging）算法——一种训练无关（training-free）的 MoE 模型压缩方法。核心算法包括两部分：(1) **Pairwise Dual-Mask Expert Merging**：对每一对 expert 的权重矩阵 W_i, W_j ∈ R^{d×h}，首先基于元素级幅度相似性计算 similarity mask M^{sim}（对称百分比差异 Δ ≤ τ_sim=0.4），标识可安全合并的共享权重；然后基于 activation-aware saliency 计算互补的 saliency masks M_i^{sal}, M_j^{sal}（使用 Wanda 指标 A = |W| ⊙ ‖X‖₂），标识各 expert 独有且重要的权重。最终通过 M_i = M_i^{sal} ∨ M^{sim} 和 M_j = M_j^{sal} ∨ M^{sim} 两个掩码，执行逐元素稀疏合并：相似条目取幅度平均、不相似条目从更 salient 的 expert 选取。同时存储每个 expert 的符号矩阵 S_i, S_j，推理时重建为 Ŵ_i = (-1)^{S_i} ⊙ M_i ⊙ W_{merged}。(2) **Bit-packed Encoding**：利用 Bfloat16 权重指数域仅使用 5-bit（值域 112-128）的观察，将指数偏移后释放 3 个冗余 bit，用于嵌入 binary mask bit 和 sign bit，消除额外 mask 存储开销。自定义 CUDA GEMV kernel 在数据加载路径上 on-the-fly 解码 mask/sign，无需物化解码后矩阵。合并策略采用随机 pairwise 分组（默认合并 2 个 expert），单次前向 pass 计算 saliency，无需迭代搜索。
  实验比较：(1) PuzzleMoE vs expert dropping（NAEE, STUN）和 expert merging（D2, HC-SMoE, Sub-MoE）及 LLM pruning（Wanda 2:4 semi-structured）；(2) 两种压缩率（25% 和 50% sparsity，即 expert 数量降至 75% 和 50%）；(3) 7 个 benchmark zero-shot accuracy + perplexity；(4) 数学推理任务（GSM8K 8-shot, Math-500, AIME24, AIME25）；(5) 压缩时间对比；(6) 内存和推理加速对比；(7) 消融：校准数据集（C4 vs MATH）、合并 expert 数量（2 vs 3）、分组策略（随机 vs 搜索）、相似阈值 τ_sim、重要性评分函数（activation-aware vs magnitude-only）、与量化结合（+3-bit AWQ → 4.8× 总压缩）。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-80GB GPU × 2（Mixtral-8x7B 全模型需 2 GPU，压缩后可单 GPU），A100-40GB GPU × 2（Qwen3-MoE 全模型需 2 GPU，压缩后可单 GPU）。Bfloat16 精度推理。压缩实验在 2×A100-80GB 上完成。预填充长度 1024，解码长度 512。

- 模型是什么。数据集和bench分别是什么。
  模型：Mixtral-8x7B-v0.1（8 experts, top-2 routing）、Deepseek-MoE-16B（64 experts, top-6 routing）、Qwen1.5-MoE-A2.7B、Qwen3-MoE-30B-A3B。
  数据集/Benchmark：(1) 校准集：C4 数据集 128 samples（seq_len=2048）；(2) Language Modeling：WikiText-2 perplexity（seq_len=2048）；(3) 下游 zero-shot：ARC-c, ARC-e, HellaSwag, PIQA, BoolQ, Winogrande, MMLU（7 个 benchmark）；(4) 数学推理：GSM8K（8-shot）、Math-500、AIME24、AIME25。结果报告 16 个不同随机种子（expert 分组）的平均值。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/Supercomputing-System-AI-Lab/PuzzleMoE
  算法 pipeline（Pairwise Dual-Mask Sparse Expert Merging）：
  ```
  # 输入: MoE 模型含 N 个 experts {E_1,...,E_N}，校准数据集 D
  # 参数: τ_sim = 0.4（相似阈值），压缩率 25% 或 50%

  # === Phase 1: Calibration（单次前向 pass）===
  for each batch in D:
      前向传播，收集每个 expert 的输入激活 X_i

  # === Phase 2: Pairwise Expert Merging ===
  for each MoE layer:
      # 随机分组 experts 为 N/2 对（配对时考虑压缩率）
      pairs = random_group(experts, num_pairs)

      for each pair (E_i, E_j):
          W_i, W_j ∈ R^{d×h}  # expert weight matrices
          X_i, X_j            # 校准激活

          # Step 1: Similarity-based Mask（标识共享知识）
          Δ = ||W_i| - |W_j|| / (|W_i| + |W_j|)     # 逐元素对称百分比差异
          M^{sim} = 1_{Δ ≤ 0.4}                      # 相似阈值

          # Step 2: Saliency-based Mask（标识独有重要权重）
          A_i = |W_i| ⊙ ‖X_i‖₂     # Wanda 指标
          A_j = |W_j| ⊙ ‖X_j‖₂
          M_i^{sal} = 1_{A_i ≥ A_j}                  # 互补 mask
          M_j^{sal} = 1 − M_i^{sal}

          # Step 3: Sparse Expert Merging
          M_i = M_i^{sal} ∨ M^{sim}                   # dual mask 合并
          M_j = M_j^{sal} ∨ M^{sim}
          # 逐元素合并：
          W_{merged} = M^{sim} ⊙ (|W_i|+|W_j|)/2 + (1−M^{sim}) ⊙ (M_i^{sal}⊙|W_i| + M_j^{sal}⊙|W_j|)

          # Step 4: 符号存储
          S_i = 1_{W_i < 0},  S_j = 1_{W_j < 0}

          # Step 5: Bit-packing（将 M_i, M_j, S_i, S_j 嵌入 W_{merged} 的 Bfloat16 指数位）
          # 指数偏移：所有指数 −112 → 5-bit 范围，释放 3 个冗余 bit
          # 打包：bit[13] = mask_i, bit[15] = sign_i（对 expert i）
          #        bit[12] = mask_j, bit[14] = sign_j（对 expert j）

  # === Phase 3: Inference（on-the-fly 解码）===
  def decode_weight(W_packed, expert_pos):  # expert_pos ∈ {0,1}
      mask_bit = (W_packed >> (13 - expert_pos)) & 1
      if mask_bit == 0:
          return 0.0  # pruned weight
      sign_bit = (W_packed >> (15 - expert_pos)) & 1
      exp = (W_packed & 0x0F80) + (112 << 7)    # 重建 Bfloat16 指数
      W_decoded = (sign_bit << 15) | exp | (W_packed & 0x007F)
      return W_decoded.view(bfloat16)
  # 自定义 CUDA GEMV kernel: 在数据加载路径上内联 decode，无需物化解码矩阵
  ```

  **核心设计洞察**：
  1. 为什么 pairwise 而非 k≥3 合并？k 个 expert 联合合并时每个位置有 (2^k−1) 种选择，组合爆炸不可行。Pairwise 合并 O(1) 闭式解 + 线性时间复杂度。
  2. 为什么相似性用对称百分比差异而非直接差值？|W_i| − |W_j| 的符号差异（opposite signs）会导致伪罚分，百分比差异 Δ 仅度量幅度相似性（与符号无关），sign 单独存储和恢复。
  3. 为什么随机分组足够好？随机分组与搜索分组的平均准确率仅差 0.3pp（72.6 vs 72.9），因为 PuzzleMoE 的力量来自逐元素稀疏合并本身而非分组策略。
  4. τ_sim=0.4 的选取：0.3-0.5 范围内性能最优，太小欠用相似性、太大过度合并。
  5. 与量化结合：50% 合并 + 3-bit 量化 + per-group scale 0.125 bit → 平均 3.35 bit/weight（原始 Bfloat16 16-bit），4.8× 总压缩。Compressed mask 仅需 log₂3≈1.58 bit（三种状态：属于 W_i / W_j / 两者共享）。

  **关键精度结果**（50% sparsity）：
  | 模型 | Vanilla | PuzzleMoE | HC-SMoE | Wanda 2:4 |
  |------|---------|-----------|---------|-----------|
  | Mixtral-8x7B | 74.1 | 72.6 | 63.8 | 68.7 |
  | Deepseek-MoE | 62.6 | 62.1 | 42.6 | 58.9 |
  | Qwen1.5-MoE | 65.9 | 65.4 | 41.9 | 63.0 |
  | Qwen3-MoE | 72.6 | 71.2 | 37.2 | 68.9 |

  PuzzleMoE 在 50% 压缩率下分数下降 0.5-2.1pp，远超其他方法（HC-SMoE 下降 10-35pp）。

## MixLLM: LLM Quantization with Global Mixed-Precision between Output and Embeddings

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是全局混合精度量化（Global Mixed-Precision Quantization）：对权重沿 output channel 维度施加全局统一的 mixed-precision——高 salience 的 output channel 使用 8-bit 量化（~10%）、其余使用 4-bit 量化（~90%），同时激活使用 8-bit 量化（W4.4A8）。核心算法包含：
  1. **全局 Salience 搜索**：基于 Fisher Information Matrix 的二阶 Taylor 展开，在单 pass 中计算所有线性层所有 output channel 对最终 loss 的贡献度 `S_c = 1/|D| * Σ|g_d^T·Δ + 0.5·(g_d^T·Δ)²|`，全局排序后选取 top-10% 通道分配 8-bit。
  2. **Weight 量化**：输出通道级别混合精度——4-bit asymmetric group-wise（group=128）+ 部分 8-bit symmetric group-wise（group=128），混合精度结构为 output channel-wise 而非 layer-wise 或 input channel-wise，简化了 GPU 系统开发。
  3. **Activation 量化**：8-bit symmetric group-wise（group=128）。
  实验比较：(1) 不同 bit-width 配置（W4A8→W4.4A8→W4.8A8→W6A8→W8A8）vs float16 baseline；(2) vs SmoothQuant W8A8、QuaRot W4A4/W4A8、QServe W4A8 等 weight-activation 量化方案；(3) vs GPTQ W4/W5、AWQ W4 等 weight-only 量化。评测 perplexity（Wikitext2、C4）和 6 个下游任务（BBH、GPQA、MMLU-Pro、MuSR、ARC-Challenge、HellaSwag）。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 80GB GPU，CUDA 12.1，PyTorch 2.4.1，transformers 4.45.2。≤8B 模型用单 A100，70B 模型用 4×A100。FP16 精度 baseline。Salience 搜索在 Wikitext2 128 样本（seq_len=2048）校准集上进行。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama 3.1（8B, 70B）、Llama 3.2（1B）、Qwen2.5（0.5B, 1.5B, 7B, 32B）、Mistral 7B v0.3、LLaMA 2（7B, 13B）。
  数据集/Benchmark：(1) 校准集：Wikitext2 128 samples（seq_len=2048）；(2) Perplexity：Wikitext2、C4；(3) 下游任务（LM-Eval）：BBH、GPQA、MMLU-Pro、MuSR、ARC-Challenge、HellaSwag，Llama 3.1 评测使用 lm-evaluation-harness commit e5af196。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/microsoft/MixLLM（MIT License），43.7% C++、36.0% Python、18.0% CUDA。Maven 依赖包：mixllm 核心量化包 + mixllm/test/test_kernel.py 内核测试 + mixllm/evaluation/run.sh 一站式校准→量化→fake inference 流程。
  
  算法 pipeline（全局混合精度量化，单 pass salience 搜索）：
  ```
  # 输入: 所有线性层权重 W_ℓ ∈ R^{out_cℓ × in_cℓ}，校准集 D
  
  # === Phase 1: 全局 Salience 计算（Algorithm 1）===
  for each linear layer ℓ:
      for each output channel c in [0, out_cℓ):
          # 在该通道注入量化扰动，计算 Loss 变化
          # 使用 Fisher Information Matrix 近似 Hessian
          # S_c = 1/|D| Σ_{d∈D} |g_d^T·(c_q - c_0) + 0.5·(g_d^T·(c_q - c_0))²|
          # 其中 Δ = c_q - c_0 是量化前后该通道的差值
          # g_d 是该通道参数关于 loss 的梯度（在 FP16 权重上计算）
          S[ℓ, c] = salience_score(W_ℓ, c, D)
  
  # 全局排序所有通道的 salience
  all_channels = [(ℓ, c, S[ℓ,c]) for ℓ, c in all_output_channels]
  sort_descending(all_channels, key=s)
  
  # 选取全局 top P% 通道分配 8-bit（P≈10%，即 W4.4A8）
  top_k = int(len(all_channels) * P / 100)
  high_salience_set = set(all_channels[:top_k])
  
  # === Phase 2: 量化执行 ===
  for each linear layer ℓ:
      W_ℓ_quant = []
      for each output channel c:
          if (ℓ, c) in high_salience_set:
              # 8-bit symmetric per-group quantization (group=128)
              s = max(|W[ℓ,c]|) / 127.0  # symmetric scale
              W_q = round(W[ℓ,c] / s), clamped to [-127, 127]
          else:
              # 4-bit asymmetric per-group quantization (group=128)
              z = min(W[ℓ,c,group]); s = (max(W[ℓ,c,group]) - z) / 15.0
              W_q = round((W[ℓ,c] - z) / s), clamped to [0, 15]
          W_ℓ_quant.append((W_q, s, z))
  
  # Activation: 8-bit symmetric per-group quantization (group=128)
  # A_q = round(A / s_a), s_a = max(|A|) / 127.0
  ```

  **核心设计洞察**：MatMul 性能受较大的 weight 张量约束而非较小的 activation 张量——weight 从 8-bit 降到 4-bit 使计算强度提升 ~80%，activation 从 8-bit 降到 4-bit 仅 ~5.88%。因此 W4.4A8 是最优 Pareto 点。

  **Salience 分布特征**（Table 4）：v_proj 和 down_proj 层 8-bit 比例极高（71.22% 和 53.82%），gate_proj 层极低（0.73%），表明 salience 高度集中在特定层类型。单 pass 搜索与迭代搜索效果相当，70B 模型搜索仅需 55 分钟，7B/8B 模型 7 分钟。

  **关键精度结果**（Llama 3.1 70B，W4.4A8）：WikiText2 PPL 增量 <0.2 vs float16（vs SOTA ~0.5）；MMLU-Pro +0.93 over SOTA 三种流行模型的平均。

## Tilus

- 属于算法pipeline的实现是什么？实验比较什么？
  - **实现**：Tilus 提出了基于**代数布局系统**的低精度矩阵乘法计算范式，通过以下机制实现高效的低精度推理：
    1. **代数布局系统**：用 primitive layouts（local/spatial）的 Kronecker product（⊗）组合表示任意寄存器张量布局。布局支持除法（inverse operation），使寄存器张量可在不同 dtype 和 layout 间无代价 reinterpretation——只要 per-thread bit 数一致即可。
    2. **低精度权重加载算法**：通过预处理变换权重全局内存布局（e.g., i6[BK,BN] → u8[BK×BN×6/8]），将非对齐的低精度元素紧凑打包为字节对齐格式，使 LoadGlobal 可高效执行 coalesced memory access。通用规则：给定每线程 n bytes 和 T threads，reinterpret 为 layout `local(n₂).spatial(T).local(n₁)`，其中 n₁=gcd(n,16), n₂=n/gcd(n₁,16)。
    3. **寄存器级类型转换（Vectorized Casting）**：利用 CUDA PRMT（permute bytes in 32-bit register）、LOP3（arbitrary 3-input logical operation）和 bitwise 指令，在寄存器内完成低精度→标准精度（如 f16）的向量化类型转换，无需线程间通信或 shared memory。
    4. **任意位宽支持**：支持 1-8 bit 的 signed integer（int2-int8）、unsigned integer（uint1-uint8）和浮点（float3-float8，任意 exponent/mantissa 分布如 e4m3, e3m3, e3m2, e2m2, e2m1, e1m1）。
    5. **统一量化 kernel 模板**：单一 program template 通过参数化 tile sizes 覆盖所有量化类型，无需为每种位宽手工编写 kernel。
  - **实验比较**：
    - Tilus vs cuBLAS FP16（标准精度 baseline）
    - Tilus vs Triton（编译器生成的 kernel，缺少低精度原生支持和布局控制）
    - Tilus vs Ladder（编译器生成的 kernel，受限于 2 的幂次位宽 + 缺少 software pipelining）
    - Tilus vs QuantLLM（手工 kernel，仅支持 FP 量化，不支持 sub-channel 量化粒度）
    - Tilus vs Marlin（手工 kernel，仅支持 4-bit signed integer，不支持 Hopper GPU）
    - 端到端：Tilus 量化 kernel 集成 vLLM vs vLLM FP16 vs Ladder uint4，在 Gemma-2-9B/QWen2.5-32B/Llama-3.3-70B 上对比 prefill 和 decode 延迟。

- 硬件平台是什么，配置是什么。
  - NVIDIA L40S GPU（48 GiB, Ada Lovelace），driver 565.57.01, CUDA 12.6.3。扩展验证：NVIDIA A100（Ampere）和 H100（Hopper）。
  - 激活类型：float16（主实验）、bfloat16、int8 均支持。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Gemma-2-9B, QWen2.5-32B, Llama-3.3-70B-Instruct。关注其矩阵乘法层（attention 和 FFN 中的 linear 层）。
  - **数据集**：使用 dummy inputs 和 dummy weights（纯系统性能测试，不依赖真实数据内容）。模型 meta-information 从 Hugging Face Hub 自动获取。
  - **Benchmark 指标**：算子级——speedup vs cuBLAS FP16、latency (ms)；端到端——prefill latency（TTFT, 2048 prompt tokens）、decode latency（per-token, batch 1/16）。
  - **量化方案**：weight-only quantization（激活保持 FP16/BF16）。支持 per-channel 和 sub-channel 量化粒度。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：https://github.com/NVIDIA/tilus。Artifact: https://github.com/yaoyaoding/tilus-artifacts, Apache 2.0。Docker 镜像 ~21 GiB。
  - **算法 pipeline（FP16 × INT6 矩阵乘法，一个 thread block）**：
    ```
    # 输入: A ∈ FP16[M,K], B ∈ INT6[K,N]
    # 输出: C ∈ FP16[M,N] = A × B
    # 每个 thread block 计算 C 的一个 BM×BN tile
    # 参数: BM=16, BN=8, BK=16, 32 threads/block
    
    # ---- 预处理（host 端，kernel 启动前）----
    # Transform B 的全局内存布局:
    #   i6[K, N] → u8[K/BK, N/BN, BK×BN×6/8]
    #   保证每个 thread (共32个) 持有 n=24 bits = 3×u8 = 4×i6
    
    # ---- Kernel 执行 ----
    bi, bj = BlockIndices()                         # thread block 索引
    
    # 创建全局内存视图（给定地址、dtype、shape）
    A_global = ViewGlobal(A_ptr, f16, [M, K])
    B_global = ViewGlobal(B_transformed_ptr, u8, [K/BK, N/BN, BK*BN*6/8])
    C_global = ViewGlobal(C_ptr, f16, [M, N])
    
    # 寄存器张量布局定义（Kronecker product 表示）
    # 对于 A 操作数 (mma.m16n8k16 的 A 布局):
    #   local(2,1) ⊗ spatial(8,4) ⊗ local(1,2)
    #   等价于: column(2,2).spatial(8,4).local(1,2)
    acc = AllocateRegister(f32, [BM, BN], 
           layout=column(2,2).spatial(8,4).local(1,2))
    
    for k in range(0, K, BK):
        # Step 1: 加载 A tile (硬件友好的 FP16 加载)
        a_tile = LoadGlobal(A_global, 
                  layout=column(2,2).spatial(8,4).local(1,2),
                  offset=[bi*BM:, k:])
        
        # Step 2: 加载 B tile (紧凑的 u8 格式, coalesced access)
        b_tile = LoadGlobal(B_global,
                  layout=local(3).spatial(32),
                  offset=[k/BK, bj, 0:])
        
        # Step 3: 无代价布局 reinterpretation
        #   u8[local(3).spatial(32)] → i6[local(2,1).column_spatial(4,8).local(2,1)]
        #   条件: 两个布局 per-thread bits 相同 (24 bits)
        b_tile = View(b_tile, dtype=i6,
                  layout=local(2,1).column_spatial(4,8).local(2,1))
        
        # Step 4: PRMT/LOP3 向量化 casting i6→f16
        b_tile = Cast(b_tile, f16)
        
        # Step 5: Tensor Core mma (f16×f16→f32 accumulate)
        acc = Dot(a_tile, b_tile, acc)    # acc += a_tile × b_tile
    
    # 输出: Cast f32→f16, Store to global
    result = Cast(acc, f16)
    StoreGlobal(result, C_global, offset=[bi*BM:, bj*BN:])
    ```
  - **低精度 loading 通用公式**：对于任意低精度张量 B[dtype, shape]，每个线程持有 n bytes 数据，T 个线程。设 n₁=gcd(n,16), n₂=n/gcd(n₁,16)，则 reinterpret 为 `u8, layout=local(n₂).spatial(T).local(n₁)` 可实现 coalesced memory access。例如 n=3 bytes (24 bits), T=32: n₁=gcd(3,16)=1, n₂=3，layout=local(3).spatial(32)，对应图 9 中的 transform program。
  - **向量化 casting 实现**（以 i6→f16 为例）：利用 PRMT 指令在 32-bit 寄存器内按字节置换/提取/拼接低精度字段，利用 LOP3 指令执行任意 3 输入逻辑运算实现符号扩展和位域操作，结合 bitwise AND/SHIFT/OR 完成完整的类型转换——全部在寄存器内完成，线程间无通信。
