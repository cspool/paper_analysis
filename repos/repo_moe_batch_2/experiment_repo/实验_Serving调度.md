## LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - LExI 在推理服务层面的核心实现是**修改 vLLM 推理框架中 MoE 层的 top-k 路由参数**，为每一层静态设置不同的 active expert 数量（而非所有层使用相同的 top-k）。具体而言：LExI 的 Stage 1 通过 Monte Carlo 采样计算每层在不同 top-k 下的 Frobenius 范数扰动损失，Stage 2 通过进化搜索在总 active expert budget B 约束下找到最优的逐层 k_j 分配。得到的 k* = (k_1, ..., k_L) 直接应用于 vLLM 的 FusedMoE 模块，替换原有的固定 top-k 路由，使得在推理时每层激活不同数量的 expert。
  - 实验比较：
    - LExI vs Baseline（固定 top-k）vs Inter-Expert Pruning (NAEE) vs Intra-Expert Pruning (MoE-I²)
    - 指标：Throughput（tokens/s，end-to-end latency 换算）、Accuracy/F1/Perplexity
    - 在多个 active expert budget B 下的 Pareto trade-off 对比（B=100, 120, 150, 180 等）

- 硬件平台是什么，配置是什么。
  - **NVIDIA H100 80GB GPUs**
  - 大多数模型 4 GPUs（Mixtral-8x7B, Qwen1.5-MoE, OLMoE, MiniCPM），DeepSeek-V2-Lite 和 DeepSeekVL2-Tiny 使用 2 GPUs
  - Tensor Parallelism 跨 GPU 部署
  - Batch size = 16 推理

- 开源Serving框架是什么。修改了什么。
  - **vLLM**（Kwon et al. 2023）：高性能 LLM 推理框架，原生支持 MoE 模型通过 **FusedMoE** 模块（融合 expert 计算和路由以提升效率）
  - LExI 的修改：
    - 在模型加载后、推理执行前，LExI 根据进化搜索得到的 k* 修改每个 MoE 层的 top-k 参数
    - 具体操作：对每个 MoE layer j，调用 `set_topk(model.moe_layers[j], k_j)` 将路由器的 TopK 选择数量从统一的 k_base 改为 k_j
    - LExI 本身不修改 vLLM 的调度逻辑、内存管理（PagedAttention）或 kernel 实现，仅改变 MoE 层的路由配置参数
    - 推理时的 token 路由流程不变：input token → router(gate) → TopK(k_j) → 激活 k_j 个 expert → 加权求和 → output

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **未开源**。论文未提供代码仓库或任何开源链接。
  - vLLM 框架在 LExI 优化下的推理全过程：
    ```
    1. 模型加载阶段
       model = load_moe_model("Mixtral-8x7B-Instruct")  # 从 HuggingFace 加载
       // 原始所有层 top-k = 2

    2. LExI 离线优化（一次执行）
       k_star = LExI_optimize(model, budget=B)  # Stage 1+2
       // k_star = [2, 1, 2, 2, ..., 1]  — 32层的逐层 top-k
       for layer_j, k_j in enumerate(k_star):
           model.moe_layers[layer_j].topk = k_j  // 修改路由参数

    3. 推理执行（vLLM 在线服务）
       // Batch 请求到达
       prompts = ["What is AI?", "Explain...", ...]  # batch_size=16
       tokens = tokenizer(prompts)

       // Prefill 阶段：并行处理所有 prompt tokens
       for layer_j in range(L):
           hidden_states = attention(layer_j, tokens)  // Self-Attention
           // MoE 层：路由 + 专家计算
           router_logits = gate(hidden_states)  // [B, L, N_experts]
           topk_indices, topk_weights = TopK(router_logits, k=k_j)  // 使用层特定 top-k
           // FusedMoE Kernel (H100 Tensor Cores)
           expert_outputs = []
           for e in topk_indices:
               expert_outputs.append(expert[e](hidden_states))  // FFN: W1→Act→W2
           output = sum(topk_weights[i] * expert_outputs[i])
           tokens = output

       // Decode 阶段：自回归生成
       for step in range(max_new_tokens):
           // 逐 token 经过所有层（每层用其 k_j 个 expert）
           ...

    4. 硬件执行
       - 每个 token 在每个 MoE 层仅激活 k_j 个 expert（而非固定的 2 或 4 个）
       - 减少的总 expert 计算量 ≈ Σ_j (k_base - k_j) / (L × k_base)
       - H100 Tensor Cores 执行 expert FFN GEMM（FusedMoE batch 操作）
       - Communication: 更少的 active experts = 更少的 all-reduce/broadcast 通信量
       - Memory bandwidth: 更少的 expert 参数需要从 HBM 加载到 SM
     ```
  - LExI 在 vLLM 中的作用：通过减少敏感度低的层的 active expert 数量，直接减少那些层的 FFN 计算量、inter-GPU 通信量和 memory bandwidth 使用，从而提升整体推理吞吐量。与 expert pruning 不同，LExI 不删除任何 expert 参数，因此模型在需要时仍可激活更多 expert（通过改变 budget B 快速调整）。

## IFMoE: An Inference Framework Design for Fine-grained MoE

- 属于Serving调度的实现是什么？实验比较什么？
  - IFMoE 提出针对 fine-grained MoE 模型的推理框架，包含两个核心实现：
    1. **并行机制重设计（EP+TP Hybrid Parallelism）**：传统 Expert Parallelism（EP）在推理时每台机器复制全部非 expert 参数（Attention、Normalization、Shared Expert），导致内存膨胀，限制了 batch size 和 context length。IFMoE 采用 EP+TP 混合并行：Expert 参数仍用 EP 分布，Shared 参数（Attention、Norm、Shared Expert）使用 Tensor Parallelism（TP）切分，避免非 expert 参数的每机全量复制。通信上，用 double All-Gather 替代传统 All-to-All 操作，因为在单节点内推理场景下 All-Gather 通信开销不高于 All-to-All。
    2. **基于 Self-Draft 的 Speculative Decoding**：观察到 fine-grained MoE 用更少 expert 也能保持较好性能，因此用 MoE 模型自身（激活更少 expert，decode_topk Dk=2）作为 draft model，快速生成 α 个 token，然后用全量 expert（encode_topk Ek=6）重新计算 KV-cache 完成 verification。不同于传统 speculative decoding，IFMoE 接受 draft model 的全部输出 token，仅更新 KV-cache。
  - 实验比较：
    - Baselines：Full model（原始 Qwen2-57B-A14B-Instruct / Deepseek-Lite-Chat 全量 expert 推理）
    - IFMoE vs Full model 在 latency 和 throughput 上的对比
    - 下游性能评估：XSum（摘要）、GSM8K（数学）、TruthfulQA（真实性）、IFEval（指令遵循）
    - 结果：IFMoE 在 benchmark 上取得 >30% 推理速度提升和 >30% 吞吐量提升，下游性能与全量模型接近（lossless 近似）

- 硬件平台是什么，配置是什么。
  - **Qwen2-57B-A14B-Instruct**：4× NVIDIA A6000 GPUs
  - **Deepseek-Lite-Chat**：2× NVIDIA A6000 GPUs
  - 节点内 GPU 间通信（单节点多卡推理场景），无跨节点通信需求
  - 论文未明确说明 CPU、内存、互联类型等具体配置

- 开源Serving框架是什么。修改了什么。
  - IFMoE **未开源**（论文 Checklist 明确标注 "IFMoE is still under develop with future features"）。
  - 论文未明确说明基于哪个开源 Serving 框架构建（如 vLLM、TGI 等），以原型系统实现。
  - 核心修改：
    1. **并行策略切换**：从纯 EP 切换为 EP+TP 混合。Shared 参数（Attention、Norm、Shared Expert）从 EP 的每卡全量复制改为 TP 切分。Expert 参数保持 EP 分布，各机器持有不同 expert。
    2. **通信原语替换**：将传统 MoE 的 All-to-All dispatch/combine 替换为 double All-Gather 操作，适应 EP+TP 混合并行的通信模式。
    3. **Decoding 流程改造**：实现 Algorithm 1 的 draft-decode + KV-cache revision 流程。Draft 阶段用 decode_topk Dk=2 激活少量 expert 快速生成 token，每 α=10 步后执行一次 encode（Ek=6 全量 expert）回填 KV-cache。
    4. **GroupedGEMM Kernel 选择**：由于 PyTorch 与 CUDA 12.5 版本兼容性问题，选用 **Cutlass GroupedGEMM** 实现（而非 cuBLAS GroupedGEMM 或 Triton GroupedGEMM）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：未开源。论文 NeurIPS Checklist 回答为 "No"，理由为 "IFMoE is still under develop with future features"。
  - **使用例子与全过程**（基于论文描述还原）：
    1. **输入**：用户请求以 batch 形式到达，每个请求包含 prompt tokens。
    2. **Prefill 阶段**：所有 machines 接收相同输入 tokens（因 EP+TP 混合模式下共享参数已 TP 切分）。Attention 和 Norm 层通过 TP 在所有 machines 上并行计算。Expert 层：每台 machine 上的 router 计算 token-to-expert 分配，通过 double All-Gather 收集各 machine 所需的 expert 输出。由于单节点内通信带宽充足（NVLink/PCIe），double All-Gather 不会成为瓶颈。
    3. **Decode 阶段（IFMoE Draft）**：Decode 时仅激活 Dk=2 个 expert（而非全量 Ek=6）。每个 decode step：router 选 top-2 experts → GroupedGEMM（Cutlass 实现）并行计算各 expert 输出 → combine → 生成 1 个 token。连续执行 α=10 步，所有 draft token 追加到 buffer。
    4. **KV-cache Revision**：每 α 步后，对 buffer 中所有 token 用 Ek=6 全量 experts 重新做一次 encode forward，更新 KV-cache 中对应位置的 key/value。此步骤确保后续 decode 的 attention 计算基于"全量 expert 应产生的 KV"。
    5. **输出**：生成的 token 序列返回给用户。IFMoE 内存节省使更大 batch size 成为可能（Qwen2 可达 batch size 256，Deepseek-Lite 可达 200）。

## FineMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving

- 属于Serving调度的实现是什么？实验比较什么？
  - FineMoE 提出**细粒度专家卸载系统**，包含三大核心组件：
    1. **Expert Map Store**：记录 iteration-level 的专家概率分布（expert map），相比 MoE-Infinity 的 request-level Expert Activation Matrix，粒度更细。每张 expert map 存储所有 MoE 层的 gate network 输出的概率分布 P_l^{(i)} ∈ R^J（而非仅记录 binary 激活或 hit count）。
    2. **Expert Map Searcher**：在每次推理迭代前，通过两种相似度搜索匹配历史 expert map：(a) **Semantic-based search**：提取 embedding layer 输出作为语义 embedding，与 Expert Map Store 中历史 semantic embedding 计算 cosine similarity，选择最相似的 iteration 的 expert map 指导前 d 层的 expert prefetching；(b) **Trajectory-based search**：对第 l ∈ [d+1, L] 层，收集前 (l-d) 层的 expert probability trajectory，与历史 expert map 对应层计算 cosine similarity，选择最优匹配指导该层 expert prefetching。
    3. **Expert Cache**：基于 LFU + expert map probability 的联合优先级进行 expert caching 和 eviction。Expert prefetching priority = p_{l,j} / (l - l_{now})，eviction priority = 1 / (p_{l,j} * freq_{l,j})。异步 Publisher-Subscriber 架构将 map searching 和 expert prefetching 与推理过程解耦。
    - **Similarity-aware 动态 expert 选择**：根据 similarity score 动态计算 threshold δ_l = Clip(1-score, 0, 1)，高 confidence（高 score）时 prefetch 较少 experts 减少内存，低 confidence 时 prefetch 较多 experts 防止 miss。
  - 实验比较：
    - Baselines：MoE-Infinity（request-level activation matrix + synchronous prefetch）、ProMoE（stride-based speculative prefetch + per-layer predictor）、Mixtral-Offloading（layer-wise speculative prefetch + LRU cache）、DeepSpeed-Inference（layer-wise offloading, no prefetch）
    - Offline：TTFT 和 TPOT 对比（prefill + decode 分离）
    - Online：CDF of end-to-end request latency（空 Expert Map Store，Azure 推理 trace 驱动）
    - 结果：FineMoE 平均降低 TTFT 74%/67%/56%/53%，降低 TPOT 46%/38%/27%/22%（vs DeepSpeed/Mixtral-Offloading/ProMoE/MoE-Infinity），提升 expert hit rate 14%/37%/68%（vs Mixtral-Offloading/ProMoE/MoE-Infinity）

- 硬件平台是什么，配置是什么。
  - **主测试台**：6× NVIDIA GeForce RTX 3090 24GB GPU，NVLink 互联，PCIe 4.0 32GB/s CPU-GPU 带宽，AMD Ryzen Threadripper PRO 3955WX 32 核，480 GB CPU 内存
  - **高配测试台**：NVIDIA A100 80GB HBM2e，2 TB/s 峰值内存带宽（单卡，无 EP）
  - Expert parallelism (EP) 将 experts 分布到多 GPU

- 开源Serving框架是什么。修改了什么。
  - 基于 **HuggingFace Transformers** + **MoE-Infinity 代码库**（https://github.com/TorchMoE/MoE-Infinity）
  - FineMoE 自身的 prototype 未发现独立开源仓库，论文基于 MoE-Infinity 实现
  - 核心修改：
    1. **新增 Expert Map Store**（Python, PyTorch + NumPy）：以 ndarray 存储 semantic embeddings 和 expert maps，转换为 tensor 进行 similarity 计算
    2. **新增 Expert Map Searcher**（Python, PyTorch）：实现 pairwise cosine similarity 计算（semantic + trajectory），redundancy score 计算用于 deduplication
    3. **修改 Expert Cache**（C++, CUDA Runtime API）：基于 MoE-Infinity 的 expert management 实现，增加 similarity-aware prefetching priority 和 eviction priority 逻辑
    4. **异步架构**：Publisher-Subscriber 模式解耦 map searching/prefetching 与推理过程

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：FineMoE 原型基于 MoE-Infinity 代码库（https://github.com/TorchMoE/MoE-Infinity），自身未发现独立开源仓库
  - **推理全过程**：
    1. **输入**：用户请求 prompt 到达，进入 Step 1 —— FineMoE 收集推理 context（semantic embedding + previous expert trajectory）
    2. **Step 2 - Expert Map Search**：Expert Map Searcher 接收 context，对前 d 层执行 semantic-based search（cosine similarity 匹配 semantic embedding），对第 l ∈ [d+1, L] 层执行 trajectory-based search（cosine similarity 匹配前 (l-d) 层 expert probability trajectory），从 Expert Map Store（容量 1K maps）中选择最优 expert map
    3. **Step 3 - Expert Prefetching & Offloading**：根据 similarity score 计算 δ_l = Clip(1-score, 0, 1)，从 searched expert map 中按概率从高到低选取 experts 直到 Σp ≥ δ_l。Expert Cache 按 prefetching priority = p_{l,j}/(l-l_{now}) 排序 prefetch 任务，从 CPU 异步传输 expert weights 到 GPU
    4. **Step 4 - Expert Serving**：每个 MoE layer：gate network 选 top-K experts → 若 expert weights 在 GPU cache 中（expert hit）直接计算 → 若不在（expert miss）on-demand 从 CPU 加载 → computation forward。Expert Cache 达到 limit 时按 eviction priority = 1/(p_{l,j} * freq_{l,j}) 踢出低优先级 experts 回 CPU
    5. **Step 5 - Expert Map Update**：每 iteration 后，新 expert map 写入 Expert Map Store。达到容量上限时计算 redundancy score RDY = (d/L)*score^{sem} + ((L-d)/L)*score^{traj}，剔除冗余 maps 保持多样性
    6. **Multi-GPU EP**：Experts 按 round-robin 分配到不同 GPU，Expert Cache 通过 CUDA Runtime API 管理 per-GPU 的 expert 加载/卸载。异步线程池在 GPU space 调度 prefetch 和 on-demand loading 任务

## HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap

- 属于Serving调度的实现是什么？实验比较什么？
  - HierMoE 修改 Megatron-LM 训练框架的 AlltoAll 通信调度，实现拓扑感知的分层 token 去重和 expert 交换：
    1. **HierD-AlltoAll（分层去重 AlltoAll）**：替代 Megatron-LM 原生的标准 AlltoAll 和 Tutel-2DH 的二维分层 AlltoAll。根据 GPU 集群的实际分层拓扑（Node/QPI/NVLink/Intra-GPU 四层），自动选择最优层级维度 d* 执行分层 AlltoAll。在每层执行 token 去重（按 bitwise OR 合并同一 expert group 内的 token 选择），消除因多个 expert 位于同一 GPU 导致的 token 重复传输。高层（Inter-Node, 低带宽 IB）通信量大幅减少，更多通信转移到高带宽 Intra-node 链路（NVLink）。
    2. **HierD-ES（分层 Expert Swap）**：在 HierD-AlltoAll 基础上，通过交换 expert 在 GPU 间的位置来平衡各 hierarchical group 的通信负载。每 iteration 或每 N iterations 计算交换任意两 expert 后的通信时间估计矩阵，选择使总通信时间最小化的 expert pair 进行交换。
  - 实验比较：
    - Baselines：Megatron-LM（标准 AlltoAll）、Tutel-2DH（二维分层 AlltoAll）、SmartMoE（expert placement）
    - 消融对比：HD2-MoE（仅 2D 去重）、HD2-MoE-Smart（2D 去重+SmartMoE swap）、HD-MoE（HierD-AlltoAll 无 swap）、HierMoE（完整方案）
    - AlltoAll 通信时间加速比：1.55×-3.32× over baselines
    - 端到端训练加速比：1.18×-1.27× over Megatron-LM

- 硬件平台是什么，配置是什么。
  - **32-GPU 集群**：4 nodes × 8 NVIDIA RTX A6000-48G GPUs
  - 每节点：Dual Intel Xeon Platinum 8358 @ 2.60GHz，512GB DDR4
  - GPU 互联：NVLink 112.5GB/s (4× link per GPU)，PCIe 4.0 x16
  - 跨节点互联：Mellanox MT28908 InfiniBand @ 200Gb/s (ConnectX-6)
  - 四层拓扑结构：Inter-Node (IB) → Inter-QPI → Inter-NVLink → Intra-NVLink
  - 软件：Ubuntu 20.04，CUDA 12.1，PyTorch 2.1.2，NCCL 2.18.5

- 开源Serving框架是什么。修改了什么。
  - 框架：**Megatron-LM** (https://github.com/NVIDIA/Megatron-LM/)，NVIDIA 的大规模 LLM 分布式训练框架，原生支持 MoE 模型的 Expert Parallelism（EP）训练。
  - HierMoE 本身未公开独立开源仓库，在 Megatron-LM 之上以原型系统实现。
  - 核心修改：
    1. **AlltoAll 通信原语替换**：将 Megatron-LM MoE layer 中用于 token dispatch 和 combine 的标准 AlltoAll（NCCL AlltoAll）替换为分层去重 AlltoAll（HierD-AlltoAll）。新增逻辑在每层 AlltoAll 前计算去重 token 分布，根据当前 iteration 的路由结果 I_route 在各层级维度上执行 token deduplication。
    2. **性能模型参数采集**：利用 NCCL collective primitives + nccl-tests (https://github.com/NVIDIA/nccl-tests) 在集群启动时测量 7 种 AlltoAll 通信变体的 α（启动延迟）和 β（每字节传输时间）参数，通过最小二乘法拟合线性模型 t = α + n · β。r² 值均在 0.997 以上。参数采集在训练前一次性完成（<300s 测量 + <10ms 拟合）。
    3. **Expert Swap Manager**：新增 HierD-ES 模块，在每 iteration 根据 token routing 结果计算交换每对 expert 的估计通信时间矩阵 Q_d*（增量更新方式，O(D·T·K·E)），选择最优 expert pair 交换。Expert 交换执行时间仅占端到端时间的 ~1%。
    4. **最优维度选择器**：在每 MoE layer 的 AlltoAll dispatch 前，计算 d=1 到 d=D（集群拓扑层数）各维度的估计通信时间 td，选择 td 最小的维度 d* 作为当前 iteration 的 HierD-AlltoAll 维度。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - HierMoE 原型未公开独立仓库，基于 Megatron-LM 实现。
  - HierMoE MoE 训练全流程（单 MoE layer，32 GPUs EP=32）：

  ① **输入**：batch tokens → embedding → hidden states x ∈ R^{T×M}（T tokens, M embedding dim）。所有 non-expert 权重（Attention, LayerNorm）按 DP 分布在各 GPU 上。Experts 按 EP 分布：每 GPU 持有 E/32 个 expert 的完整参数。

  ② **Router/Gating**（GPU 本地计算）：对每个 token 计算 softmax gating + Top-K 选择 → routing mask I_route ∈ R^{T×E}（boolean，标记 token i 是否选择 expert j）。Gating 计算量小，在 GPU 上本地完成。

  ③ **HierD-AlltoAll 维度选择**（CPU 侧控制逻辑）：
     - 从 I_route 计算各层级的去重 token 分布：对 d=1..D，将 E 个 expert 聚合到 U[d] 个 group，去重（OR）后统计每组 token 数
     - 代入性能模型公式 (3)-(5) 计算各维度的估计通信时间 td
     - 选择 d* = argmin td
     - 复杂度 O(D·T·K)，T=1024·batch，K=8，D=4 → 微秒级

  ④ **HierD-AlltoAll Dispatch**（GPU 间 NCCL 通信）：
     - 按选定的 d* 维度执行分层 AlltoAll
     - 例 d*=3: Inter-Node AlltoAll (IB, 200Gb/s, U[1]=4 groups) → 去重后跨节点传输
       → Inter-QPI AlltoAll (QPI, U[2]=8 groups) → 去重后跨 QPI domain 传输
       → Intra-QPI (NVLink) AlltoAll (112.5GB/s, Intra-level-2, U[3]=G=32) → GPU 级分发
     - 每层传输前执行 token 去重：同一 group 内多 expert 被选中 → 只传一份 token 副本

  ⑤ **Expert FFN 计算**（GPU 本地计算）：
     - 每个 GPU 对收到的 tokens 执行本地 experts 的 FFN 前向计算
     - W_gate GEMM → activation → W_up GEMM → element-wise → W_down GEMM
     - cuBLAS GEMM kernel 在 GPU SM 上执行

  ⑥ **HierD-AlltoAll Combine**（GPU 间 NCCL 通信）：
     - 逆向分层 AlltoAll，将 expert 输出按原 token 位置合并回各 GPU
     - 同样使用 HierD-AlltoAll 选定的 d* 维度

  ⑦ **HierD-ES Expert Swap**（每 iteration 可选，~1% 时间开销）：
     - CPU 侧：从 I_route 增量更新 Z 矩阵 (E×E×G)，计算 Q_d*[r,c] 估计时间矩阵
     - 用 smooth-max (γ=10) 平滑后选择 (r*,c*) = argmin Q_d*[r,c]
     - GPU 侧：通过 NCCL P2P Send/Recv 交换 expert r* 和 c* 的参数+优化器状态

  ⑧ **Backward**：expert FFN backward → HierD-AlltoAll combine backward (等同于 dispatch) → HierD-AlltoAll dispatch backward (等同于 combine) → Attention backward

  ⑨ **输出**：完成一层 MoE layer 的前向+反向，token hidden states 传递至下一层

- 属于Serving调度的实现是什么？实验比较什么？
  - HOBBIT 在 Llama.cpp 之上构建了一个混合精度 Expert Offloading 推理系统，核心 Serving 调度实现包括三个层次：
    1. **Token-level Dynamic Expert Loading (Section 3.2)**：根据 gating output ||G(x)|| 动态评估 expert 重要性，计算 unimportance degree score s_{e_i} = Σ_{j=0}^{i-1} ||G(x)_{e_j}||，用双阈值 T1=0.6/T2=0.9 将 cache-miss expert 分为三组 — 高精度 (FP16/INT8)、低精度 (INT4/INT2)、跳过。低精度 expert 加载量减少最高 4×。
    2. **Layer-level Adaptive Expert Prefetching (Section 3.3)**：利用 MoE 层间 gating input 的高余弦相似度（相邻层 top-1 expert 预测准确率平均 96%），通过 Stacking Computer 一次性批量计算所有后续层的 gating output，实现自适应预取。结合混合精度预取来降低错误预测的惩罚。
    3. **Sequence-level Multidimensional Expert Caching (Section 3.4)**：提出 LHU (Least High Precision Frequently Used) 策略，结合 LRU + LFU + LHU + FLD 四种策略的加权线性组合管理混合精度 expert cache，最小化 cache miss penalty（高精度 miss 代价为 C，低精度 miss 代价为 (B_l/B_h)·C）。
  - 实现量：在 Llama.cpp 基础上增加 8,000 行 C++/C 代码。
  - 实验比较：Transformers (TF)、DeepSpeed-Inference (DS)、Llama.cpp (LL)、MoE-Offloading (MO)、MoE-Infinity (MI)、Fiddler (FD) 共 6 个 baseline。三组硬件对比配置（见表 2）。指标：prefill latency (s)、decoding speed (tokens/s)。四种 I/O 长度组合：[16,32], [16,128], [128,32], [128,128]，batch_size=1。

- 硬件平台是什么，配置是什么。
  - **RTX 4090**（edge server）：24GB GPU memory，256GB CPU memory，64 CPU cores，PCIe 4.0 (32GB/s theoretical)，Samsung NVMe SSD 980 PRO (7,000 MB/s theoretical, ~3,000 MB/s practical)。
  - **Jetson AGX Orin**（end device）：32GB unified memory（CPU/GPU 共享），12 CPU cores。SSD 同上。
  - 三组测试配置：
    - Jetson AGX Orin + INT8 模型：HB vs LL, MI
    - RTX 4090 + FP16 模型：HB vs TF, DS, MO, MI
    - RTX 4090 + CPU + FP16 模型：HB vs LL, FD

- 开源Serving框架是什么。修改了什么。
  - 开源框架：**Llama.cpp** (https://github.com/ggerganov/llama.cpp)。
  - 核心修改：
    1. **权重分布修改**：将所有 non-expert 权重 + 部分多精度 expert 置于 GPU memory，全部 expert 权重存储于 CPU memory。这与 Llama.cpp 原生的 "足够多完整层放 GPU + 剩余层放 CPU" 模式不同，针对 MoE 稀疏激活特性优化。
    2. **两种计算模式**：
       - GPU-centric：主线程在 GPU 上计算，scheduler 线程从 CPU/SSD 加载所需精度 expert 到 GPU。
       - CPU-GPU cooperative：expert cache miss 时，主线程发送 expert 输入到 CPU helper 线程计算并返回结果。
    3. **Dynamic Expert Loader**：Expert Scorer 基于 ||G(x)|| 动态计算重要度分数，生成不同精度加载任务入 Task Queue，Expert Scheduler 通过 read() 系统调用异步加载。
    4. **Adaptive Expert Predictor**：Stacking Computer（stack + matmul + top-k）一次性计算所有后续层 gating，自适应选择需预取的 expert。
    5. **Multidimensional Cache Manager**：Policy Performer 维护 LRU/LFU/LHU 优先级记录，按加权和公式 evict 最低优先级 expert。高/低精度 cache 分离管理。新 sequence 开始时重置记录。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未提供 HOBBIT 完整开源仓库链接（"we implement HOBBIT on top of Llama.cpp with 8,000 additional lines of C++/C code"）。
  - HOBBIT MoE 推理全流程（GPU-centric 模式，单 token 解码为例）：

  ① **输入**：token → embedding → hidden state x，所有 non-expert 权重（Attention + LayerNorm 等）常驻 GPU VRAM。
  ② **第 i 层 MoE 计算**：
     - Router/Gating 在 GPU 上计算，得到 top-K expert IDs 和 gate weights ||G(x)||。
     - Adaptive Expert Predictor 的 Stacking Computer 利用当前 gating input 一次性计算后续层的预测 expert IDs。
  ③ **Cache 检查**：Multidimensional Cache Manager 检查 on-demand experts 和 prediction experts 是否在 expert cache (GPU VRAM) 中。
  ④ **Expert 加载（cache miss）**：
     - Expert Scorer 按公式 s_{e_i} = Σ_{j=0}^{i-1} ||G(x)_{e_j}|| 计算专家不重要度分数。
     - s_{e_i} ≤ 0.6 → 加载高精度 expert (FP16/INT8) 从 CPU memory via PCIe read()
     - 0.6 < s_{e_i} ≤ 0.9 → 加载低精度 expert (INT4/INT2)，传输量减少 4×
     - s_{e_i} > 0.9 → 跳过该 expert
     - Expert Scheduler 异步执行加载任务。
  ⑤ **Expert 替换**：Cache Manager 按加权优先级公式 p_t = w_lru·p_t^lru + w_lfu·p_t^lfu + w_lhu·p_t^lhu + w_fld·p_t^fld 选择最低优先级 expert evict，写入新 expert。高/低精度 cache 分别管理。
  ⑥ **计算**：GPU cores 对所有 on-demand experts 执行 FFN 计算：y = Σ G(x)_{e_i} · E_{e_i}(x)。
  ⑦ **推进**：hidden state 传递至第 i+1 层，prediction experts 已被预取到 cache 中。
  ⑧ **输出**：最后一层 logits → softmax → 采样 → next token。

## FloE: On-the-Fly MoE Inference on Memory-constrained GPU

- 属于Serving调度的实现是什么？实验比较什么？
  - FloE 实现了一个面向内存受限 GPU 的 on-the-fly MoE 推理 Serving 系统，核心调度优化包括：
    1. **Expert Hybrid Compression（Section 3.2）**：将 expert 三组投影矩阵在 DRAM 中压缩存储（gate/down 做上下文稀疏化，up 做 INT2 量化），减少 PCIe 传输量。每 expert 压缩比 9.3×（~300MB → ~32MB）。
    2. **Inter-Expert Sparsity Predictor（Section 3.3.1）**：学习型 MLP 预测器，在当前层 i 计算时预测下一层 i+1 激活的 expert，实现预取。参数规模随层深动态调整（单层 MLP 32K → 双层 MLP 2M）。平均 precision 0.88。
    3. **Intra-Expert Sparsity Predictor（Section 3.3.2）**：参数免费的复用型预测器，用当前层 hidden state 与下一层 W_up 做矩阵乘，预估 up projection 输出激活分布，预计算稀疏掩码。平均 recall 0.95。
    4. **Compact Asynchronous Transfer（Section 3.4.2）**：将 gate 投影的列和 down 投影的行在 DRAM 中紧凑排列（co-locate），提升 chunk 大小从 d_hidden×num_bytes 到 2×d_hidden×num_bytes；使用 AVX-512 SIMD 指令 + 多线程打包到 pinned memory，跨多 CUDA stream 异步传输。
  - 实验比较：DeepSpeed-MII（ZeRO-Infinity offloading，FP16）、Mixtral-Offloading（expert 预测+缓存+INT3 量化）、Fiddler（CPU-GPU 协同计算）、Mixtral-GPU（HQQ INT2 全量 GPU 驻留，作为延迟下界参考）。指标：端到端 TPS（tokens per second），单 expert 计算延迟，传输带宽利用率。

- 硬件平台是什么，配置是什么。
  - GeForce RTX 3090（24GB VRAM），64核 CPU @2.3GHz，256GB DRAM，PCIe 4.0 ×16（峰值带宽 ~32GB/s）。
  - 限制 VRAM 使用量从 12GB 到 24GB 进行消融实验。

- 开源Serving框架是什么。修改了什么。
  - 论文未明确给出 FloE 完整代码仓库。基于 PyTorch 构建，使用 Triton 实现 sparse kernel。核心修改点：
    - expert 权重在 DRAM 中的紧凑布局（co-locate gate 列和 down 行）。
    - 自定义的 pinned memory 管理 + AVX-512 SIMD 多线程异步传输模块。
    - 学习型 inter-expert 预测器（SGD 训练，<1min 收敛）。
    - Triton-based sparse GEMV kernel（替代 PyTorch dense GEMV）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文发表于 ICML 2025。未明确给出开源链接。
  - FloE 推理全流程（单 token 解码为例）：
    1. **输入**：token → embedding → hidden state x（1×4096），非 expert 权重常驻 VRAM。
    2. **第 i 层 MoE 计算**：
       a. Router/Attention 计算在 GPU 上完成，hidden state 传递到 MoE 层。
       b. Inter-expert predictor（在层 i-1 已运行）已预取层 i 的压缩 expert 权重到 VRAM cache。
       c. Intra-expert predictor（在层 i-1 已运行）已预计算 up projection 激活的稀疏掩码。
    3. **Sparse expert 计算**：GPU 执行 sparse GEMV kernel，仅加载选中通道的 gate 列和 down 列。
    4. **预取下一层**：当前层 hidden state 输入 inter-expert predictor（预测层 i+1 激活 expert）+ intra-expert predictor（用层 i+1 复用的 W_up 预估稀疏分布），触发 compact async transfer 从 DRAM 传输层 i+1 压缩 expert 权重。
    5. **输出**：logits → next token。整个过程 transfer 与 computation 流水化，PCIe 传输被 GPU 计算隐藏。
  - 端到端结果：在 RTX 3090 + 12GB VRAM 约束下，FloE 速度达到 Mixtral-GPU（全量 INT2 驻留 GPU）的 91%，对比 DeepSpeed-MII 加速 48.7×。

## Fast Inference of Mixture-of-Experts Language Models with Offloading

- 属于Serving调度的实现是什么？实验比较什么？
  - 论文构建了 MoE 专用的 expert offloading 调度系统，替代 naive offloading（HuggingFace accelerate 的 device_map="auto"），实现更高效的 batch size 1 交互式推理调度。核心调度策略包括：
    1. **Expert LRU Cache 调度**：每个 MoE 层维护 k 个最近使用 expert 的 GPU 缓存，避免每 token 都从 host RAM 重新加载。k=2（12GB GPU）或 k=4（16GB GPU）。
    2. **Speculative Expert Prefetching**：在当前层 expert 加载完成后立即启动投机预取——将下一层 MoE gate 应用于当前层 hidden states 预测下一层最可能使用的 expert，在后台异步加载。
    3. **内存分割调度**：当 host RAM 无法容纳完整模型时（如 Google Colab），expert 在 host RAM 和 GPU memory 之间按 LRU 策略动态换入换出，换出时回写到 host RAM。
    4. **异步多 buffer 架构**：分配 b=4 个共享 device buffer 用于异步拷贝和预取，所有 MoE 层复用，减小内存足迹。
  - 实验比较：
    - Full algorithm vs w/o expert pre-loading vs w/o LRU cache & pre-loading vs Naive offloading (accelerate)
    - 在 T4/RTX 3060/RTX 3080 Mobile/A100 上测 tokens/sec，batch size=1

- 硬件平台是什么，配置是什么。
  - T4 (Google Colab free-tier): 16GB VRAM, PCIe Gen.3, host-to-device 8-16GB/s
  - RTX 3080 Mobile (gaming laptop): 16GB VRAM, PCIe Gen.4
  - RTX 3060 (midrange desktop): 12GB VRAM, PCIe Gen.3
  - A100-80GB-SXM: 用于对比参考（可无 offloading 运行）
  - 约束：GPU VRAM 仅容纳 non-expert layers + k 个缓存 expert，全部 expert 参数需存储于 host RAM

- 开源Serving框架是什么。修改了什么。
  - **基线框架**：HuggingFace accelerate（naive offloading with device_map="auto"），按层整体加载/卸载到 GPU
  - **论文自建 offloading 系统**：不修改已有 serving 框架，而是基于 PyTorch 构建专用的 MoE offloading 调度器，替代 accelerate 的默认 offloading
  - 修改/新增内容：
    - **Per-expert offloading**（替代 per-layer offloading）：将每个 MoE 层的 8 个 expert 独立 offload，仅加载 top-2 所需的 expert 到 GPU
    - **LRU cache 管理**：在 GPU 侧维护 per-layer expert cache，跟踪使用顺序
    - **投机预取逻辑**：使用当前层 hidden states 推测下一层 expert 选择，异步启动 host-to-device 传输
    - **内存管理**：expert 参数连续 pinned memory 分配 + b=4 个共享 device buffer

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：代码开源在 https://github.com/dvmazur/mixtral-offloading
  - **Serving 框架执行全过程（以 Mixtral-8x7B-Instruct 在 T4 16GB + 2-bit experts 为例）**：

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 模型加载阶段                                                  │
    │    - Attention/embedding/norm 层: 4-bit HQQ 量化, 常驻 GPU       │
    │    - Experts (45.1B params): 2-bit HQQ 量化, 常驻 pinned RAM    │
    │    - GPU 侧预分配: k×32_layers 个 expert slot + 4 个共享 buffer │
    │    - 初始化 LRU cache: 每层随机加载 k 个 expert 到 GPU           │
    │           ↓                                                     │
    │ 2. 用户输入 prompt tokens [T₁, T₂, ..., Tₙ]                      │
    │    Prompt 处理 (prefill): 逐层计算, 每层 expert 加载一次          │
    │    (与生成阶段的逐 token 加载不同, prefill 相对高效)               │
    │           ↓                                                     │
    │ 3. Token 生成循环 (autoregressive decode, batch=1)               │
    │    for each new token:                                          │
    │      for layer l in 0..31:                                      │
    │        ┌─ Attention block ──────────────────────────────────┐   │
    │        │  已常驻 GPU, 直接计算, 输出 h_attn                  │   │
    │        └────────────────────────────────────────────────────┘   │
    │        ┌─ MoE Gate ────────────────────────────────────────┐   │
    │        │  gate_scores = W_gate[l] @ h_attn  (常驻GPU)      │   │
    │        │  top2_experts = topk(gate_scores, 2)               │   │
    │        └────────────────────────────────────────────────────┘   │
    │        ┌─ Expert Loading (GPU cache check) ────────────────┐   │
    │        │  for e in top2_experts:                            │   │
    │        │    if e in GPU_cache[l]:                           │   │
    │        │      expert_weights = GPU_cache_buf[e]  // 命中    │   │
    │        │      mark_recent(e)                                │   │
    │        │    else:                                           │   │
    │        │      evict = LRU_evict(C_l)  // cache miss         │   │
    │        │      copy GPU_cache[evict] → host_pinned[evict]   │   │
    │        │      copy host_pinned[e] → GPU_cache_slot          │   │
    │        │      (单次 contiguous mem copy, PCIe 8-16GB/s)      │   │
    │        └────────────────────────────────────────────────────┘   │
    │        ┌─ Speculative Prefetch (后台异步) ─────────────────┐   │
    │        │  # 用当前 h_attn 预测下一层 expert                │   │
    │        │  pred_gate = W_gate[l+1] @ h_attn                  │   │
    │        │  pred_top1, pred_top2 = topk(pred_gate, 2)        │   │
    │        │  async_copy host_pinned[pred_top1] → shared_buf   │   │
    │        │  async_copy host_pinned[pred_top2] → shared_buf   │   │
    │        │  (在独立的 CUDA stream 上执行, 与当前层计算重叠)    │   │
    │        └────────────────────────────────────────────────────┘   │
    │        ┌─ Expert FFN Computation ──────────────────────────┐   │
    │        │  out = 0                                          │   │
    │        │  for e, w in zip(top2_experts, gate_weights):     │   │
    │        │    out += w * SiLU(W_gate_e @ h) * (W_up_e @ h)  │   │
    │        └────────────────────────────────────────────────────┘   │
    │      → 下一 token 生成                                         │
    │           ↓                                                     │
    │ 4. 输出: generated tokens, 2-4 tokens/sec                       │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **关键性能数据（Table 2, 2-bit experts）**：
    | Hardware | Full algo | w/o pre-load | w/o cache & pre-load | Naive (accelerate) |
    |----------|-----------|-------------|---------------------|-------------------|
    | A100 | 3.06 tok/s | 2.92 tok/s | 2.27 tok/s | 1.39 tok/s |
    | 3080 Mobile | 2.66 tok/s | 2.23 tok/s | 1.76 tok/s | 1.06 tok/s |
    | RTX 3060 | 2.28 tok/s | 2.05 tok/s | 1.55 tok/s | 0.92 tok/s |
    | T4 (Colab) | 2.09 tok/s | 1.57 tok/s | 1.17 tok/s | 0.66 tok/s |

    从 Naive offloading 到 Full algorithm 加速约 2.2×（T4）到 3.2×（RTX 3060, 3-bit）。Pre-loading 在 RTX 3060 上效果最显著（因 k=2 的较小 LRU cache）。

## HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - HybriMoE 在 kTransformers 之上实现了一套面向资源受限环境的混合 CPU-GPU MoE 推理调度系统，核心调度优化包括：
    1. **Dynamic Intra-Layer Hybrid Scheduling（Section IV-B）**：引入三条优先级规则简化 expert-to-hardware 映射问题——GPU 优先计算已缓存的高负载 expert（降序），CPU 优先计算未缓存的低负载 expert（升序），PCIe 传输优先移动高负载未缓存 expert 到 GPU。将调度问题形式化为 `argmin max(CPU_TIME(cpu_expert), GPU_TIME(gpu_expert))` 的分配问题。执行前通过仿真阶段迭代填充 CPU/GPU/PCIe 时间线来评估调度策略，选择最小化延迟的配置。
    2. **Impact-Driven Inter-Layer Prefetching（Section IV-C）**：利用残差连接导致相邻层 hidden state 高度相似的特点，复用后续层的 gating 信息预测 next-3-layers 的 expert 激活，通过仿真评估预取每个 expert 对整体调度效率的潜在影响（impact），贪心选择收益最高的 expert 进行预取。
    3. **Score-Aware Caching / MRS Replacement Policy（Section IV-D）**：提出 Minus Recent Score (MRS) 替换策略，利用 expert routing score 作为缓存优先级信号。公式 S = α × TopP(s) + (1-α) × S，仅累积 top-p 个 expert 的 score（p 通常为 2× 激活 expert 数）。利用高 score expert 在后续 iteration 中更可能被重用的观察。
  - 实验比较：llama.cpp（静态按层映射到 CPU/GPU）、AdapMoE（SOTA GPU-centric MoE 调度，自适应 prefetching+caching）、kTransformers（SOTA CPU-GPU hybrid MoE 调度，按历史激活频率静态映射）。指标：TTFT（prefill 阶段）、TBT（decode 阶段）。消融实验对比 Scheduling、Prefetching、Caching 各组件的独立贡献。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA RTX A6000
  - CPU：Intel Xeon Gold 5220R，限制使用 10 cores（模拟边缘部署场景）
  - 通过调整 GPU expert cache ratio（25%, 50%, 75%）评估不同硬件配置下的性能和扩展性

- 开源Serving框架是什么。修改了什么。
  - 基于 **kTransformers** (https://github.com/kvcache-ai/ktransformers) 和 **llama.cpp** kernels。
  - kTransformers 提供灵活的基础设施用于 kernel injection 和混合 CPU-GPU 执行。
  - 核心修改：
    1. **Hybrid Scheduler**：在 kTransformers 的 expert 执行路径中插入优先级规则驱动的调度逻辑，运行时动态分配 expert 到 CPU 或 GPU。仿真阶段在 warmup 中完成，收集 CPU/GPU 处理速度和数据传输延迟。
    2. **Parallel Execution Engine**：利用 fine-grained CUDA stream 调度实现 CPU、GPU、PCIe transfer 三者的并行执行。修改 C++ kernels 直接处理 expert 计算任务分配，消除 Python 开销。
    3. **Prefetching Module**：在每层计算时利用后续层 gating 信息预测并预取 expert，与当前层计算并行。
    4. **MRS Cache Manager**：替换 kTransformers 原有的 LFU 缓存策略为 score-aware MRS 策略。
    5. **Marlin Quantization**：集成 llama.cpp 的 Marlin 4-bit 量化 kernel 提升计算效率和降低内存使用。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：代码开源在 https://github.com/PKU-SEC-Lab/HybriMoE
  - **HybriMoE MoE 推理全流程（单 MoE layer，以 Mixtral-8x7B 为例）**：

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Warmup 阶段                                                    │
│    - 测量 CPU expert computation speed (per token latency)        │
│    - 测量 GPU expert computation speed                            │
│    - 测量 PCIe transfer latency (CPU↔GPU expert weight copy)      │
│    - 初始化 MRS cache: 随机加载 k 个 expert 到 GPU cache          │
│           ↓                                                       │
│ 2. 用户输入 prompt tokens [T₁, T₂, ..., Tₙ]                       │
│    Prefill + Autoregressive decode loop:                           │
│           ↓                                                       │
│ 3. 每层 MoE 执行                                                  │
│    ┌─ Attention Block ────────────────────────────────────────┐   │
│    │  Non-expert 权重常驻 GPU，直接在 GPU 计算                  │   │
│    └──────────────────────────────────────────────────────────┘   │
│    ┌─ MoE Gating/Router ──────────────────────────────────────┐   │
│    │  gate_logits = W_gate @ h  (常驻 GPU, 轻量级)             │   │
│    │  topk_experts, gate_weights = topk(softmax(logits), K)   │   │
│    └──────────────────────────────────────────────────────────┘   │
│    ┌─ Hybrid Scheduling Decision ─────────────────────────────┐   │
│    │  1. 检查每个 activated expert 是否在 GPU cache 中          │   │
│    │  2. 构建 GPU Queue: 已缓存 expert，按 load 降序排列        │   │
│    │  3. 构建 CPU Queue: 未缓存 expert，按 load 升序排列        │   │
│    │  4. Simulation Phase: 迭代填充 CPU/GPU/PCIe 时间线          │   │
│    │     while 未完成所有 expert:                               │   │
│    │       选最早完成的 timeline → 执行对应操作:                │   │
│    │         - GPU: 从 GPU Queue 取最高 load cached expert      │   │
│    │         - CPU: 从 CPU Queue 取最低 load uncached expert    │   │
│    │         - PCIe: 从 CPU Queue 取最高 load uncached expert   │   │
│    │           → 传输完成后插入 GPU Queue (按 load 降序)        │   │
│    │  5. 选择 min max(CPU_TIME, GPU_TIME) 的调度方案            │   │
│    └──────────────────────────────────────────────────────────┘   │
│    ┌─ Parallel Execution (3-way: CPU + GPU + PCIe) ───────────┐   │
│    │  CUDA Stream 0 (GPU compute):                              │   │
│    │    for cached expert e (sorted by load desc):              │   │
│    │      Marlin 4-bit quantized GEMM:                          │   │
│    │        gate_out = SiLU(W_gate_4bit @ h)                    │   │
│    │        up_out = W_up_4bit @ h                              │   │
│    │        out += gate_weight * gate_out * up_out @ W_down_4bit│   │
│    │  CUDA Stream 1 (PCIe transfer):                            │   │
│    │    for high-load uncached expert:                          │   │
│    │      cudaMemcpyAsync(CPU_weight → GPU_buffer, PCIe)        │   │
│    │  CPU Thread Pool (CPU compute):                            │   │
│    │    for low-load uncached expert:                           │   │
│    │      llama.cpp C++ kernel:                                 │   │
│    │        CPU GEMM → expert FFN output                        │   │
│    │    (CPU 端首 expert 计算慢，后续 expert 因 cache 利用更快)  │   │
│    └──────────────────────────────────────────────────────────┘   │
│    ┌─ Impact-Driven Prefetching (与当前层计算并行) ────────────┐   │
│    │  1. 读取 layer l+1, l+2, l+3 的 gating weights             │   │
│    │  2. 计算预测的 expert activation:                          │   │
│    │     pred_experts_{l+i} = topk(W_gate_{l+i} @ h_l, K)       │   │
│    │  3. 对每个候选 prefetch expert:                            │   │
│    │     模拟 "若预取该 expert" 对调度效率的影响                 │   │
│    │     (复用 Section IV-B 的仿真逻辑)                          │   │
│    │  4. 贪心选择 impact 最高的 expert → async 预取             │   │
│    └──────────────────────────────────────────────────────────┘   │
│    ┌─ MRS Cache Update (每 iteration 结束) ───────────────────┐   │
│    │  1. 获取当前 iteration 的 routing scores s                 │   │
│    │  2. 更新 priority scores:                                  │   │
│    │     S = α × TopP(s) + (1-α) × S                            │   │
│    │     (仅 top-p=2K 个 expert score 被累积)                    │   │
│    │  3. 若需 evict: 选择 S 最低的 expert 从 GPU cache 移除    │   │
│    │  4. 将新激活的 expert 写入 GPU cache                       │   │
│    └──────────────────────────────────────────────────────────┘   │
│           ↓                                                       │
│ 4. Expert FFN 聚合输出                                            │
│    out = Σ gate_weights[e] × ExpertFFN_e(h)                       │
│    (CPU 计算的结果通过 PCIe 拷回 GPU)                              │
│           ↓                                                       │
│ 5. 输出: next token logits → 采样 → 下一个 token                  │
└─────────────────────────────────────────────────────────────────┘
```

  - **关键技术原理**：
    - **优先级规则的设计动机**：GPU 优先高负载缓存 expert（减少传输开销），CPU 优先低负载未缓存 expert（CPU 延迟与 load 线性相关），PCIe 优先高负载未缓存 expert（最大化 GPU 利用率）
    - **仿真调度原理**：warmup 阶段测量 CPU_TIME_per_expert、GPU_TIME_per_expert、TRANSFER_TIME → 运行时基于实际 expert activation 构建优先级队列 → 迭代式仿真（贪心 fill timelines）→ 输出最优 expert-to-device 分配方案
    - **MRS vs LRU/LFU**：LRU/LFU 不考虑 MoE expert 的 routing score 预测信号。MRS 利用"高 score expert 更可能在下一 iteration 被重用"的观察（图 3b），通过指数移动平均累积 score 信号

  - **关键性能数据**：
    | Stage | Model | Cache Ratio | HybriMoE Speedup vs kTransformers |
    |-------|-------|-------------|-----------------------------------|
    | Prefill (avg) | All models | 25%-75% | 1.33× |
    | Decode (avg) | All models | 25%-75% | 1.70× |

  - **Ablation Study (Qwen2, 25% cache ratio)**:
    | Technique | Prefill Latency(s) | Prefill Speedup | Decode Latency(s) | Decode Speedup |
    |-----------|-------------------|-----------------|-------------------|----------------|
    | Baseline (kTransformers) | 1.47 | — | 0.21 | — |
    | +Scheduling | 1.17 | 1.26× | 0.14 | 1.46× |
    | +Prefetching | 1.39 | 1.06× | 0.18 | 1.15× |
    | +Caching | — | — | 0.15 | 1.38× |
    | All | 1.13 | 1.31× | 0.11 | 1.86× |

  - **MRS Cache Hit Rate vs LRU (Figure 9)**:
    | Model | 25% Cache (MRS vs LRU) | 75% Cache (MRS vs LRU) |
    |-------|------------------------|------------------------|
    | Mixtral | 36.2% vs 30.2% (+6%) | 83.3% vs 80.6% |
    | DeepSeek | 52.7% vs 47.7% (+5%) | — |
    | Qwen2 | 52.8% vs 45.0% (+7.8%) | — |

## HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - HAP 在 DeepSpeed-FastGen 上构建了一套面向 MoE 推理的动态混合并行策略自动选择系统。核心调度优化包括：
    1. **Module Decomposition（Section III-B）**：将 MoE 模型分解为 Attention 模块和 Expert 模块两个独立计算单元，各配备专用的推理延迟仿真模型（基于 FLOPs 的计算仿真模型 + 基于数据量的通信仿真模型），使用随机森林回归拟合仿真系数 η 和 ρ，计算仿真误差 <10%，通信仿真误差 <5%。
    2. **ILP-based Hybrid Parallel Strategy Search（Section III-C）**：构建 Attention 模块（DP/TP/DP+TP 混合）和 Expert 模块（EP/TP/EP+TP 混合）的并行策略搜索空间，将最小化端到端推理延迟问题形式化为整数线性规划（ILP），通过 Python PuLP 库求解最优混合并行配置。搜索在典型 8-GPU 单机配置下 <1 秒完成。
    3. **Dynamic Parallelism Transition Strategy（Section III-D）**：prefill 和 decode 阶段使用不同并行策略时，维护 INT4（per-group 量化）备份权重于 CPU memory，通过多 stream 异步流水线上传并反量化恢复为 BF16 精度。过渡策略根据仿真在 weight redistribution（集合通信）与 quantized upload+dequant 之间选择开销更低的方案。Per-group 量化保持 MMLU 67.7%（与原版一致）、GSM8K 58.0%（vs 原版 58.3%）。
  - 实验比较：HAP-based inference vs TP-based inference（baseline），端到端延迟对比。四种推理场景：短上下文约束输出（256 in + 64 out）、短上下文扩展输出（256 in + 2048 out）、长上下文约束输出（4096 in + 64 out）、长上下文扩展输出（4096 in + 2048 out）。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100 80GB（节点内 NVLink，高带宽），4×A100 及 8×A100 配置。
  - NVIDIA A6000 48GB（节点内 PCIe，低带宽），4×A6000 配置。
  - NVIDIA V100（节点内 PCIe），8×V100 配置。
  - 单节点多 GPU 部署场景。

- 开源Serving框架是什么。修改了什么。
  - **Serving 框架**：DeepSpeed-FastGen（https://github.com/microsoft/DeepSpeed），基于 MII 和 DeepSpeed-Inference 的高吞吐文本生成框架，原生支持 TP 等静态并行策略。
  - **修改内容**：
    1. **并行策略搜索引擎**：在 DeepSpeed-FastGen 的模型加载/初始化阶段集成 ILP 求解器（Python PuLP），基于硬件规格（GPU 数、显存、带宽）和模型配置（hidden dim、expert 数、层数）自动搜索最优混合并行策略。
    2. **动态策略切换机制**：在 prefill→decode 过渡点插入策略切换逻辑——若 Expert 模块在 prefill 和 decode 使用不同策略，触发 INT4 量化权重上传 + GPU 反量化，或 weight redistribution via AllGather/AllToAll。
    3. **计算/通信仿真模型校准**：initialization 阶段运行 microbenchmark 收集计算和通信延迟数据，训练随机森林回归模型以精确估计各策略组合的延迟。
    4. **内存约束感知**：ILP 约束中包含 KV cache、attention 权重、expert 权重、activation 的 per-device 内存占用约束，对 EP 采用保守上限估计（2× TP activation footprint）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：论文未明确给出 HAP 独立开源仓库。基于开源 DeepSpeed-FastGen 构建。
  - **HAP Serving 框架执行全过程（以 Mixtral-8x7B 在 4×A6000、4096-token context + 64-token generation 为例）**：
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 初始化阶段                                                    │
    │    HAP 读取模型配置 (Mixtral-8x7B: 32 layers, hidden=4096,      │
    │    experts=8, MoE_inter=14336) 和硬件配置 (4×A6000, PCIe)       │
    │    → Microbenchmark 收集计算/通信延迟数据                        │
    │    → 训练随机森林回归仿真模型 (计算 η, 通信 ρ)                    │
    │    → ILP 求解器 (PuLP) 搜索最优混合并行策略:                      │
    │      Attention 模块: DP=4 (避免 TP 的 AllReduce 通信开销)        │
    │      Expert 模块 prefill: EP=4 (低通信量)                        │
    │      Expert 模块 decode: TP=4 (避免 EP 负载不均衡)               │
    │    → 准备 INT4 量化 expert 权重备份于 CPU memory                  │
    │           ↓                                                      │
    │ 2. Prefill 阶段 (4096 tokens 并行)                               │
    │    for layer l in 0..31:                                         │
    │      ┌─ Attention (DP=4, 各 GPU 独立计算) ───────────────────┐  │
    │      │  各 GPU 持有完整 attention 权重                         │  │
    │      │  处理 1/4 batch tokens, 无通信                          │  │
    │      └──────────────────────────────────────────────────────┘  │
    │      ┌─ MoE Gate (各 GPU 复制执行) ──────────────────────────┐  │
    │      │  gate_logits = W_gate @ h  [1×(64×4096) → 1×8]       │  │
    │      │  top2_experts = topk(softmax(logits), 2)              │  │
    │      └──────────────────────────────────────────────────────┘  │
    │      ┌─ Expert FFN (EP=4, 每 GPU 2 experts) ─────────────────┐  │
    │      │  GPU0: experts 0,1; GPU1: experts 2,3; etc.          │  │
    │      │  All-to-All dispatch: tokens 路由到 expert 所在 GPU   │  │
    │      │  各 GPU 计算本地 expert FFN (SwiGLU)                   │  │
    │      │  All-to-All combine: 聚合输出                         │  │
    │      └──────────────────────────────────────────────────────┘  │
    │           ↓                                                      │
    │ 3. Prefill→Decode 过渡（动态策略切换）                            │
    │    Expert 模块策略: EP=4 → TP=4                                 │
    │    HAP 仿真评估过渡开销:                                          │
    │      T_reshard (AllGather+AllToAll) vs T_upload+T_dequant       │
    │    → 选择 INT4 量化权重上传方案 (更低开销):                       │
    │      CPU→GPU async copy INT4 权重 (multi-stream)                │
    │      → GPU 端 per-group dequant 恢复 BF16                       │
    │      → 过渡开销与 prefill 计算重叠（T_dequant < T_attn+T_comm） │
    │           ↓                                                      │
    │ 4. Decode 阶段 (逐 token autoregressive, 64 tokens)              │
    │    for each new token:                                           │
    │      ┌─ Attention (DP=4, 同 prefill) ───────────────────────┐  │
    │      │  各 GPU 独立计算，batch=1 per GPU                      │  │
    │      └──────────────────────────────────────────────────────┘  │
    │      ┌─ MoE Gate (各 GPU 复制执行) ──────────────────────────┐  │
    │      └──────────────────────────────────────────────────────┘  │
    │      ┌─ Expert FFN (TP=4, 各 GPU 持有完整 expert 的 1/4) ───┐  │
    │      │  Expert 权重沿中间维度切分 (14336→3584 per GPU)       │  │
    │      │  各 GPU 计算部分输出 → AllReduce 聚合                  │  │
    │      │  decode 阶段通信量小 (单 token)，TP 负载均衡优势明显   │  │
    │      └──────────────────────────────────────────────────────┘  │
    │           ↓                                                      │
    │ 5. 输出: 64 个 generated tokens                                  │
    │    端到端延迟 vs TP baseline: 1.68× speedup on A6000            │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **关键设计原理**：
    - 长上下文 prefill 场景：通信是瓶颈（TP 的 AllReduce 通信量大），HAP 为 Attention 选 DP（无通信）、Expert 选 EP（All-to-All 通信量低于 TP 的 AllReduce）
    - Decode 场景：计算是瓶颈（单 token），EP 的负载不均衡会浪费 GPU 计算资源，HAP 为 Expert 切换为 TP
    - 短上下文场景：TP 在多数配置下已接近最优，HAP 搜索后可能仍选 TP，实现不低于 baseline 的延迟

    **关键性能数据**：
    | Scenario | Model | Hardware | HAP Speedup vs TP |
    |----------|-------|----------|-------------------|
    | 256in+64out | Mixtral-8x7B | 4×A100 | 1.16× |
    | 256in+64out | Qwen1.5-MoE | 4×A6000 | 1.37× |
    | 4096in+64out | Mixtral-8x7B | 4×A6000 | up to 1.68× |
    | 4096in+64out | Mixtral-8x7B | 4×A100 | up to 1.77× |
    | 4096in+64out | Qwen2-57B | 4×A100 | up to 1.52× |
    | 2048in+64out | Mixtral-8x7B | 8×V100 | 1.57× |
    | 4096in+2048out | Mixtral-8x7B | 4×A100 | up to 1.13× |

    HAP 优势最大出现在长上下文+约束输出场景（prefill 主导延迟，通信瓶颈严重），短上下文+扩展输出场景加速最小（decode 主导延迟，TP 已是最优）。

## FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion

- 属于Serving调度的实现是什么？实验比较什么？
  - FUSCO 实现了一个 MoE-friendly 通信库，通过融合数据变换（data transformation）与通信（communication），替代框架中原有的 NCCL all-to-all 操作。核心实现包括三部分：
    1. **Data-Fused Communication Engine (dComm)**：引入 Segment Descriptor 抽象，将 MoE token 建模为逻辑 segments，在通信路径上直接完成数据重排（rearrangement），消除通信前后的显式 permute/repack 步骤。
    2. **Communication Planner**：基于 MoE router 的 token-expert 分配结果，构建两级 descriptor（Node-Level Forwarding + Expert-Level Distribution），实现层次化路由（hierarchical routing），在目的节点指定一个 forwarder GPU 接收跨节点数据后经 intra-node 链路分发，消除重复跨节点传输（token deduplication）。
    3. **Online Load Balancer**：将各节点的 GPU 按负载排序后贪心分组为 communication groups，每组包含每个节点的一个 GPU，组内 GPU 互为 forwarding endpoints，通过 circular shift 使高负载 GPU 分布到不同组，缓解跨节点流量倾斜。
  - 实验比较 FUSCO 与 NCCL（通用集合通信库）和 DeepEP（SOTA MoE 通信库，基于 NVSHMEM）在三种流量模式下的通信延迟，以及在 Megatron-LM 训练和 SGLang 推理上的端到端性能。

- 硬件平台是什么，配置是什么。
  - 8 节点集群，每节点配置：
    - CPU：2x Intel Xeon Platinum 8558（48 核/socket，192 线程/节点）
    - GPU：8x NVIDIA H100 80GB HBM3
    - 节点内互联：NVLink（每 GPU 18 条 NVLink link，理论聚合带宽约 480 GB/s per GPU）
    - 节点间互联：10x 400 Gbps Mellanox ConnectX-7 NIC（RoCE）
    - NIC-GPU 互联：PCIe 桥接
  - 软件环境：Linux kernel 5.15.0, Ubuntu 24.04, NVIDIA driver 535.183.06, CUDA 12.9, NCCL 2.26.3, PyTorch 2.7.0

- 开源Serving框架是什么。修改了什么。
  - **训练框架**：Megatron-LM —— 将 FUSCO 替换 Megatron-LM 中 MoE 层的 all-to-all 操作，通过扩展的 PyTorch distributed backend 调用 dComm primitive。
  - **推理框架**：SGLang —— 使用 prefill-decode disaggregation 配置，在 MoE 模型的 prefill 阶段用 FUSCO 替换 all-to-all 操作。
  - **修改内容**：约 500 行 Python 适配层（thin adaptation layer），桥接框架的 token-routing 路径与 FUSCO 的 planner 和 dComm primitive，无需修改模型逻辑或 expert kernel。FUSCO 本身在 NCCL transport layer 之上实现（约 2000 行 C++/CUDA），复用 NCCL 的设备注册、连接管理和 transport 层。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：论文声明 "Our code and data will be made publicly available"，截至分析时未在 web search 中发现公开代码仓库。FUSCO 基于 NCCL（https://developer.nvidia.com/nccl）构建，复用其 transport 层。
  - **Serving框架执行全过程（以 SGLang 推理为例，EP=64，seqlen=16k）**：
    ```
    ┌─────────────────────────────────────────────────────┐
    │ 1. SGLang Prefill 阶段                              │
    │    输入：用户 prompt tokens [T₁, T₂, ..., T₁₆ₖ]     │
    │    MoE Router 计算 token→expert 分配 (top-k=8)       │
    │           ↓                                          │
    │ 2. FUSCO Communication Planner                      │
    │    读取 token-expert 矩阵 A (T×K)                      │
    │    构建两级 descriptor：                               │
    │      - Node-Level: 每个 destination node 仅一份拷贝   │
    │      - Expert-Level: node 内各 GPU→expert 的精确偏移  │
    │    Online Balancer 按 greedy circular-shift 分组      │
    │           ↓                                          │
    │ 3. FUSCO dComm Engine 执行                           │
    │    ┌─ Sender GPU ─────────────────────────────┐      │
    │    │ Slice₀: [desc→gather segments→ring buf]  │      │
    │    │ Slice₁: [desc→gather segments→ring buf]  │      │
    │    │ ...   ← GPU memory copy + layout transform│     │
    │    └──────────────────────────────────────────┘      │
    │              ↓ RDMA (RoCE, 400Gbps)                   │
    │    ┌─ Receiver (Forwarder) GPU ───────────────┐      │
    │    │ desc→scatter to receive buffer            │      │
    │    │ NVLink P2P → distribute to expert GPUs    │      │
    │    └──────────────────────────────────────────┘      │
    │           ↓                                          │
    │ 4. Expert FFN Computation                           │
    │    各 GPU 上的 expert 直接消费已排列好的 token buffer  │
    │           ↓                                          │
    │ 5. dComm 反向 (Combine)                              │
    │    对称的 descriptor 驱动的 gather+all-to-all         │
    │           ↓                                          │
    │ 6. SGLang 继续 decode                                │
    │    输出：first-token generation (TTFT)               │
    └─────────────────────────────────────────────────────┘
    ```
    
    **训练全过程（Megatron-LM，EP=64）**：
    ```
    Forward:
    Input tokens → Attention → MoE Gate (top-k routing)
      → FUSCO dispatch (dComm: segment descriptor → pipelined GPU-to-ringbuf→RDMA)
      → Expert FFN (各GPU直接计算，无额外重排)
      → FUSCO combine (dComm: 反向 descriptor 驱动)
      → Output
    
    Backward:
    Gradient → FUSCO combine (反向 dispatch) → Expert backward
      → FUSCO dispatch (反向 combine) → Attention backward
    ```
    
    FUSCO 在框架中作为 `send/recv/allgather` 级别的 primitive 暴露，调用方式类似：
    ```python
    # 伪代码：在 Megatron-LM MoE 层中使用 FUSCO
    # 传统 NCCL 方式:
    # tokens = permute(tokens, routing_indices)    # 显式重排
    # tokens = all_to_all(tokens)                   # 通信
    # tokens = permute(tokens, expert_indices)      # 再次重排
    # expert_output = expert_ffn(tokens)
    # ... 对称的反向操作
    
    # FUSCO 方式:
    descriptors = fusco_planner.build_plan(token_expert_matrix)
    tokens = fusco_dcomm.dispatch_with_fusion(tokens, descriptors)  # 一步完成
    expert_output = expert_ffn(tokens)
    tokens = fusco_dcomm.combine_with_fusion(expert_output, descriptors)
    ```

## Faster MoE LLM Inference for Extremely Large Models

- 属于Serving调度的实现是什么？实验比较什么？
  - 论文基于 sglang serving 框架对 fine-grained MoE 模型（DeepSeek-V2-Lite、DeepSeek-V3）进行推理效率分析，核心调度层面的实现和实验包括：
    1. **MoE batch effect 分析（Section 4）**：通过 PyTorch + torch.compile 模拟实验，分析 MoE 层在不同 sequence length 下的 latency 和 throughput，对比 FFN，量化 MoE 因额外 expert 参数加载导致的"弱化批次效应"（weakened batch effect）。
    2. **Expert skipping 效率实验（Section 5）**：在 sglang 上修改 expert 激活逻辑，使每层仅激活 na（2 到原始值）个 expert，测量不同并发度（2-768）下的 throughput 变化和加速比。
    3. **Expert pruning 效率实验（Section 6）**：在 sglang 上减少总 expert 数量 ne（从 64 降到 8-48），测量不同并发度（2-784）下的 throughput 和加速比。
    4. **Expert parallelism 效率分析（Section 4.2）**：理论分析 EP vs TP 的通信开销——EP 在 fine-grained MoE 中通过 group-constrained routing 可将跨节点通信从 2(nd-1)Ld 降至 2naLd。
  - 实验比较：
    - Section 4：MoE vs FFN 在不同 sequence length L 下的 per-token latency 和 AI
    - Section 5：不同 na（2-6/2-8）在不同并发度下的 throughput（token/s）和 speedup ratio
    - Section 6：不同 ne（8-64）在不同并发度下的 throughput 和 speedup ratio（up to 2.3×）
    - Section 5.2：不同 inter-layer expert 分配策略（ascending/descending/peak/valley）下的 benchmark 性能

- 硬件平台是什么，配置是什么。
  - **Section 4 模拟实验**：1× NVIDIA Tesla A800 80G PCI-e, Intel Xeon Silver 4314 CPU @ 2.40GHz (24 cores), 15×16GB ECC DDR4@2666MHz
  - **DeepSeek-V2-Lite (Section 5 & 6)**：2× NVIDIA Tesla A800 80G PCI-e, Intel Xeon Silver 4314 CPU @ 2.40GHz (24 cores), 15×16GB ECC DDR4@2666MHz
  - **DeepSeek-V3 (Section 5 & 6)**：8× NVIDIA Tesla H200 141G SXM5, Intel Xeon Platinum 8558 CPU @ 2.10GHz (48×2 cores), 32×64GB ECC DDR4@2666MHz
  - 效率测试约束：固定 1024 input tokens + 1024 output tokens

- 开源Serving框架是什么。修改了什么。
  - **Serving 框架**：sglang build v0.4.4 post 1 (commit ad4e58bf67ec833ff4d036af5129ec6e1633efc4)
  - **Profiling 工具**：sglang.bench
  - **修改内容**：
    1. **Expert skipping 修改**：在 sglang 的 MoE expert 调度中，将所有 MoE 层的激活 expert 数 na 从默认值（V2-Lite=6, V3=8）统一降低到 2 至原始值之间的某个值。修改涉及 MoE layer 的 top-k 选择逻辑——在 router gate 输出后，将 topk 的 k 参数替换为缩减后的 na。
    2. **Expert pruning 修改**：在模型加载阶段，根据选择策略（random/structured/activate count/soft count）从 ne 个 expert 中选择 ne' 个保留，其余不加载到显存。修改涉及模型权重加载路径——仅加载选中的 expert 参数。
    3. **Section 4 模拟**：使用 PyTorch + torch.compile + HuggingFace Transformers (MixtralModel)，实现 MoE 和 FFN 的 latency 模拟，不修改 sglang。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：论文未提供独立开源仓库。使用开源框架 sglang (https://github.com/sgl-project/sglang) 和 PyTorch、HuggingFace Transformers。
  - **Serving 框架执行全过程（以 DeepSeek-V2-Lite, expert skipping na=2, 并发度=512, 2×A800 为例）**：

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 模型加载与配置                                                │
    │    sglang 加载 DeepSeek-V2-Lite (16B, 64 experts, na 从 6→2)     │
    │    - Attention/Embedding/Norm: 常驻 GPU 0                        │
    │    - Shared Expert (ds=10944): 常驻 GPU 0                        │
    │    - Routed Experts (64×de=1408): 分布在 2×A800 (EP=2)           │
    │    - 每层仅激活 na=2 个 routed expert (从 6 降至 2)               │
    │           ↓                                                      │
    │ 2. 请求到达与调度                                                │
    │    sglang scheduler 接收 512 个并发请求                          │
    │    每个请求: 1024 input tokens + 1024 output tokens              │
    │    Continuous batching: 动态合并请求                              │
    │           ↓                                                      │
    │ 3. Prefill 阶段 (1024 tokens 并行处理)                           │
    │    for each MoE layer:                                           │
    │      ┌─ Attention (MLA) ───────────────────────────────────┐     │
    │      │  常驻 GPU, 1024 tokens 并行计算                       │     │
    │      │  KV cache 写入 sglang 的 RadixAttention 管理         │     │
    │      └────────────────────────────────────────────────────┘     │
    │      ┌─ MoE Gate ──────────────────────────────────────────┐    │
    │      │  gate_logits = W_r @ h  (1024×64 tensor)             │    │
    │      │  topk_indices = topk(sigmoid(gate_logits), k=2)      │    │
    │      │  (原 k=6, 现改为 k=2, 减少 expert 加载和计算)         │    │
    │      └────────────────────────────────────────────────────┘     │
    │      ┌─ Expert FFN (仅 top-2, EP=2) ───────────────────────┐    │
    │      │  GPU0: expert e₀,e₁ 的 FP16 权重                      │    │
    │      │  GPU1: expert e₂,e₃ 的 FP16 权重                      │    │
    │      │  各 GPU 仅计算分配给自己的 expert                      │    │
    │      │  all-reduce 聚合结果                                   │    │
    │      └────────────────────────────────────────────────────┘     │
    │      ┌─ Shared Expert (常驻 GPU) ──────────────────────────┐    │
    │      │  out += SharedExpert(h)  (ds=10944, 不参与 routing)   │    │
    │      └────────────────────────────────────────────────────┘     │
    │           ↓                                                      │
    │ 4. Decode 阶段 (逐 token autoregressive)                        │
    │    for each new token (共 1024 tokens):                         │
    │      同 prefill 的 MoE 流程，但 batch=1 per request              │
    │      sglang 的 continuous batching 将多请求的 decode 合并批处理    │
    │      na=2 时：memory I/O 减少 (仅需加载 2 个 expert 参数)         │
    │           ↓                                                      │
    │ 5. 输出: generated tokens + throughput metric (token/s)          │
    │    结果 (Table 8): na=2 相比 na=6 在 concurrency=512 时           │
    │    throughput 从 9379→10954 tok/s (16.8% 提升)                   │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **expert pruning 执行全过程（ne=64→16, concurrency=512, 2×A800）**：
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Pre-inference Expert Selection                                │
    │    Soft Count 方法：记录每个 expert 在 calibration data 上的      │
    │    激活次数，选 top-ne' 个最常用 expert 保留。                      │
    │    从 64 个 expert 中选 16 个，其余 48 个不加载。                  │
    │           ↓                                                      │
    │ 2. 模型加载（仅加载选中的 expert）                                │
    │    sglang 加载 DeepSeek-V2-Lite，但每层仅 ne'=16 个 expert        │
    │    总参数量从 16B 降至约 16×(16/64)×(routed_expert_ratio)         │
    │    GPU 显存占用减少，但 FLOPS 不变（单 token 仍需 na=6 个 expert    │
    │    计算，只是可选的 expert pool 缩小）                              │
    │           ↓                                                      │
    │ 3. 推理执行                                                      │
    │    相同 sglang 推理流程，但 expert pool 从 64→16                  │
    │    每个 expert 的计算强度 (compute intensity) 提高                 │
    │    → 低并发时 memory I/O 瓶颈缓解 → up to 2.3× speedup            │
    │    高并发时（192+）throughput 可能下降（sglang 内部策略 bug？）      │
    │           ↓                                                      │
    │ 4. 性能影响                                                      │
    │    ne'=48 (25% 减少): 最佳方法 soft count, Avg 64.2 vs 66.0       │
    │    ne'=32 (50% 减少): 最佳方法 soft count, Avg 57.8 vs 66.0       │
    │    ne'=16 (75% 减少): 最佳方法 soft count, Avg 47.8 vs 66.0       │
    │    随机选择 ne'=16/32 几乎丧失语言能力                             │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **关键数据（Table 8, Figure 3a, DeepSeek-V2-Lite）**：
    | Concurrency | na=6 (baseline) | na=5 | na=4 | na=3 | na=2 | Speedup (na=2 vs 6) |
    |-------------|-----------------|------|------|------|------|----------------------|
    | 2 | 479 | 511 | 544 | 583 | 631 | 1.32× |
    | 32 | 2345 | 2412 | 2529 | 2716 | 3069 | 1.31× |
    | 128 | 5591 | 5660 | 5812 | 5960 | 6126 | 1.10× |
    | 512 | 9379 | 9783 | 9950 | 10102 | 10954 | 1.17× |
    | 768 | 9453 | 9694 | 10043 | 10249 | 10968 | 1.16× |

## Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

- 属于Serving调度的实现是什么？实验比较什么？
  - Fiddler 实现了一个面向资源受限环境的 MoE 推理 CPU-GPU 协同调度系统。核心调度策略包括：
    1. **动态执行策略选择（Algorithm 1）**：每个 expert 根据输入 token 数量 s 在三种策略中动态选择——(a) 若 expert 权重已在 GPU memory，直接在 GPU 执行；(b) 若 `cpu_lat(s) > gpu_lat(s) + trans_lat()`，则从 CPU memory 拷贝权重到 GPU 并在 GPU 执行；(c) 否则从 GPU memory 拷贝 activation 到 CPU memory 并在 CPU 执行。决策基于 latency model：GPU 延迟恒定（受限于参数加载），CPU 延迟随输入量线性增长。
    2. **Expert 热门度导向的 GPU 放置**：离线 profiling 统计各 expert 激活频率，按热门度降序将尽可能多的 expert 放入 GPU memory，最大化 GPU cache hit rate。在 Env1（56/256 expert on GPU）下 hit rate 从随机 21.9% 提升至 25.2%，Env2（125/256）从 48.8% 升至 53.0%。
    3. **AVX512_BF16 CPU 专用计算 kernel**：利用 Intel AVX512_BF16 指令集实现 CPU 端 expert 计算 kernel，PyTorch 原生不支持该指令集。
  - 实验比较：
    - Fiddler vs DeepSpeed-MII (ZeRO-Infinity) vs Mixtral-Offloading vs llama.cpp
    - 场景 a：不同输入/输出长度的 end-to-end 单 batch 推理吞吐
    - 场景 b：长上下文 prefill 的 TTFT（512-4096 input tokens）
    - 场景 c：beam search（width 4-16）end-to-end 延迟

- 硬件平台是什么，配置是什么。
  - Environment 1：NVIDIA Quadro RTX 6000（24576MiB VRAM），Intel Xeon Gold 6126（48 cores），PCIe Gen3 x16（32GB/s）
  - Environment 2：NVIDIA RTX 6000 Ada（49140MiB VRAM），Intel Xeon Platinum 8480+（112 cores），PCIe Gen4 x16（64GB/s）
  - 约束：两块 GPU 的显存均无法容纳 Mixtral-8x7B 全部 90GB+ 参数，Env1 仅可放 56/256 expert，Env2 可放 125/256 expert

- 开源Serving框架是什么。修改了什么。
  - Fiddler 基于 PyTorch 构建，不修改已有开源 serving 框架，而是自建 CPU-GPU 协同调度系统。
  - 核心修改/新增：
    - **Per-expert 执行策略决策**：替代 per-layer 统一执行，对每个 expert 独立判断 CPU/GPU 执行策略
    - **Expert 权重分配**：initialization 阶段将 non-expert 层+热门 expert 放 GPU memory，其余 expert 放 CPU memory
    - **Latency model 校准**：initialization 阶段测量 cpu_lat(s)、gpu_lat(s)、trans_lat() 为运行时决策提供参数
    - **CPU AVX512_BF16 kernel**：自定义 C++ kernel 替代 PyTorch 默认 CPU GEMM

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：代码开源在 https://github.com/efeslab/fiddler
  - **Serving 框架执行全过程（以 Mixtral-8x7B 16-bit 在 Environment 1, single-batch inference 为例）**：

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 初始化阶段                                                    │
    │    - 加载 Mixtral-8x7B (47B params, 16-bit, >90GB)               │
    │    - Non-expert 层 (Attention/Embedding/Norm): 常驻 GPU (~2B)     │
    │    - Expert 层 (32 layers × 8 experts = 256 experts):             │
    │      热门度 profiling (ShareGPT calibration) → 按热门度排序      │
    │      GPU 放置 top-56 热门 expert，其余 200 个 expert 放 CPU RAM   │
    │    - 测量 microbenchmark: cpu_lat(s), gpu_lat(s), trans_lat()    │
    │           ↓                                                       │
    │ 2. 用户输入 prompt tokens [T₁, T₂, ..., Tₙ]                       │
    │    Prefill + autoregressive decode loop:                          │
    │           ↓                                                       │
    │ 3. 每层 MoE 执行 (Forward)                                        │
    │    for each layer l in 0..31:                                     │
    │      ┌─ Attention block ────────────────────────────────────┐    │
    │      │  Weights 常驻 GPU, 直接在 GPU 计算                     │    │
    │      └──────────────────────────────────────────────────────┘    │
    │      ┌─ MoE Gate ───────────────────────────────────────────┐   │
    │      │  gate_scores = W_gate[l] @ h  (常驻 GPU)              │   │
    │      │  top2_experts, gate_weights = topk(softmax(scores),2) │   │
    │      │  infl_size[j] = count(tokens routed to expert j)      │   │
    │      └──────────────────────────────────────────────────────┘   │
    │      ┌─ Fiddler Algorithm 1: Per-expert 执行决策 ────────────┐  │
    │      │  for j = 1 to 8:                                       │  │
    │      │    s = inp_size[j]                                     │  │
    │      │    if s == 0: continue                                 │  │
    │      │    if is_at_gpu(l, j):                                 │  │
    │      │      → GPU 直接执行 (无数据传输)                        │  │
    │      │    elif cpu_lat(s) > gpu_lat(s) + trans_lat():         │  │
    │      │      → CPU→GPU copy expert weight (300MB/expert, PCIe) │  │
    │      │      → GPU 执行 expert FFN                             │  │
    │      │    else:                                               │  │
    │      │      → GPU→CPU copy activation (s×4096 floats, PCIe)   │  │
    │      │      → CPU AVX512_BF16 执行 expert FFN                 │  │
    │      │      → CPU→GPU copy output activation                  │  │
    │      └──────────────────────────────────────────────────────┘   │
    │      ┌─ Expert FFN 聚合 ────────────────────────────────────┐   │
    │      │  out = Σ gate_weights[j] * SiLU(W_gate_e @ h)         │   │
    │      │       * (W_up_e @ h)   // 各 expert 独立计算后加权加和  │   │
    │      └──────────────────────────────────────────────────────┘   │
    │           ↓                                                       │
    │ 4. 输出: generated tokens                                         │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **关键性能数据**：
    | Scenario | Fiddler vs Best Baseline | Env1 | Env2 |
    |----------|--------------------------|------|------|
    | Single batch (avg) | 1.26× vs llama.cpp | — | — |
    | Long prefill TTFT (avg) | 1.07× vs DeepSpeed-MII | — | — |
    | Beam search (avg) | 11.57× vs llama.cpp | — | — |

    **三种执行策略的 latency model**：
    - Strategy (a): latency = gpu_lat(s) ≈ constant（GPU 直接从显存执行）
    - Strategy (b): latency = gpu_lat(s) + trans_lat() ≈ constant + weight_transfer（GPU 计算+PCIe 传权重，约 2-5× 计算时间）
    - Strategy (c): latency = cpu_lat(s) + negligible_act_copy ≈ linear in s（CPU 计算，activation 拷贝 <1% 总延迟）
    - 决策阈值：当 s 较小时 cpu_lat(s) < gpu_lat(s) + trans_lat() 选 (c)；当 s 较大时选 (b)

## FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

- 属于Serving调度的实现是什么？实验比较什么？
  - FarSkip-Collective 在推理侧修改 vLLM 和 SGLang 两个开源推理框架，实现 MoE 模型中 all-reduce 通信与计算的重叠。具体修改包括：
    1. **MoE 层 all-reduce 异步化**：vLLM/SGLang 中 MoE 推理使用 all-reduce（而非 all-to-all）来聚合各 EP rank 上的专家输出。FarSkip 将 all-reduce 改为 async_op 模式运行，仅在下一个 MoE 计算前同步，利用 FarSkip 架构中 MLP 输入不依赖当前 attention 输出的特性，将 all-reduce 与 attention 子块计算重叠。
    2. **Attention 层 RowParallelLinear all-reduce 异步化**：修改 attention 输出投影层（RowParallelLinear），将内部 all-reduce 改为 async_op 模式，同步调用仅在下一次 attention 层之前执行。
    3. **HIP/CUDA-graphs 集成**：使用 graph-compatible 通信 API 调用（PyNCCL direct Python binding），使重叠机制与 CUDA graphs 兼容，支持 prefill 和 decode 阶段的 fused kernel。
    4. **MLA（Multi-head Latent Attention）特殊处理**：针对 DeepSeek 模型的 MLA prefill 和 decode 分别使用不同的 fused kernel，每种情况单独实现 async all-reduce 调用。
  - 实验比较 FarSkip-Collective 推理 vs 常规推理在 TTFT（Time-To-First-Token）和解码阶段的加速比。

- 硬件平台是什么，配置是什么。
  - vLLM 推理：1× AMD MI300X 8GPU 机器；FP8 量化 + fused-MoE forward kernel。
  - SGLang 推理单节点：TP=8, EP=8 配置。
  - SGLang 推理多节点：2 节点系统，TP=16, EP=16，8×400Gbs NIC 互联。
  - 推理配置：prefill 阶段 BS=2（per-device），EP=8, TP=8；decode 阶段 BS=1024（多节点 large-batch 设置）。

- 开源Serving框架是什么。修改了什么。
  - **推理框架**：vLLM [19] 和 SGLang [45]——现代 LLM 推理引擎，支持 TP、EP、PP，用于 MoE 模型（如 DeepSeek）的分布式推理。
  - **修改内容**：
    1. **MoE 层**：将 EP 相关的 all-reduce（原本用于聚合各 rank 的专家输出）从同步模式改为 async_op 模式，同步点延迟到下一个 MoE 计算之前。这利用了 FarSkip 架构中 MLP 子块输入使用 outdated activation（$o_{k-1}$）不依赖最新 attention 输出的特性。
    2. **Attention 层**：修改 RowParallelLinear 输出投影层中的 all-reduce 为 async_op 模式，同步延迟到下一个 attention 层之前。对于 MLA 的 prefill 和 decode fused kernel 分别处理。
    3. **CUDA Graphs 兼容**：使用 graph-compatible 通信 API（PyNCCL）替代标准 torch.dist 调用，确保在 CUDA graph capture 场景下异步通信正常工作。
    4. **设计原则**：所有修改在 PyTorch API 层面完成（torch.dist async_op + torch.cuda.Stream），避免 low-level kernel 或 Triton 修改，保持硬件无关性和框架兼容性。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：论文声明 "We plan to open-source our implementation and modified model checkpoints and provide easy integration with the upstream frameworks"，截至分析时未在 web search 中发现公开代码仓库。
  - **推理框架执行全过程（以 Llama-4-Scout 109B 在 vLLM 上，EP=8, TP=8, MI300X 8GPU 为例）**：

    ```
    ┌─────────────────────────────────────────────────────────┐
    │ 1. vLLM 接收推理请求                                     │
    │    输入：用户 prompt tokens [T₁, T₂, ..., Tₙ]            │
    │    vLLM scheduler 分配请求到 GPU，管理 KV cache           │
    │           ↓                                              │
    │ 2. Attention 子块（TP=8, 列并行 Q/K/V + 行并行 O）       │
    │    ┌─ Q/K/V projection (ColumnParallelLinear) ──────────┐│
    │    │  各 TP rank 独立计算，无通信                         ││
    │    └────────────────────────────────────────────────────┘│
    │    ┌─ Core Attention (fused MLA kernel) ────────────────┐│
    │    │  各 TP rank 独立计算 attention scores + output       ││
    │    └────────────────────────────────────────────────────┘│
    │    ┌─ Output projection (RowParallelLinear) ───────────┐ │
    │    │  **FarSkip 修改**: all_reduce(async_op=True)       │ │
    │    │  启动异步 all-reduce，立即返回，不等待完成           │ │
    │    │  返回 partial output (本 rank 计算结果)             │ │
    │    └────────────────────────────────────────────────────┘│
    │           ↓                                              │
    │ 3. MoE 子块（EP=8, 各 rank 持有 E/8 个 expert 权重）     │
    │    ┌─ Gating/Router (各 rank 复制执行) ─────────────────┐│
    │    │  Router(token) → top-k expert indices              ││
    │    │  各 rank 独立计算，无通信                            ││
    │    └────────────────────────────────────────────────────┘│
    │    ┌─ Expert FFN (各 rank 本地 expert 计算) ────────────┐│
    │    │  输入：replicated activations（vLLM EP 方式）       ││
    │    │  各 rank 仅计算自己持有的 experts                    ││
    │    │  fused-MoE forward kernel, FP8 量化                 ││
    │    └────────────────────────────────────────────────────┘│
    │    ┌─ All-Reduce 聚合 (EP 间) ──────────────────────────┐│
    │    │  **FarSkip 修改**: all_reduce(async_op=True)       ││
    │    │  聚合各 rank 的专家输出                              ││
    │    │  异步启动，返回 partial 结果                         ││
    │    └────────────────────────────────────────────────────┘│
    │           ↓                                              │
    │ 4. 下一层 Attention 子块执行时                             │
    │    ┌─ 同步上一层的 all-reduce ──────────────────────────┐│
    │    │  Wait(all_reduce_handle)  // 此时通信已被重叠       ││
    │    │  获取完整 activation 用于残差加和                    ││
    │    └────────────────────────────────────────────────────┘│
    │    // FarSkip 利用 mlp-in_k = o_{k-1} (outdated)         │
    │    // 使得 MLP 输入不需要等待 attention all-reduce 完成   │
    │    // attention 计算可与上一层的 all-reduce 重叠          │
    │           ↓                                              │
    │ 5. 输出：first token (TTFT) 或 next token (TBT)          │
    └─────────────────────────────────────────────────────────┘
    ```

    **vLLM EP 的 all-reduce 方式说明**：
    不同于训练中使用 all-to-all Dispatch+Combine，vLLM/SGLang 的 MoE EP 实现将 activation 在各 rank 上复制，仅 expert 权重按 EP 分布。各 rank 计算自己的 experts 后，通过 all-reduce 聚合结果（而非 all-to-all）。这种方式消除了 Dispatch/Combine 的 permutation 开销，但 all-reduce 仍然是阻塞的。FarSkip 将此 all-reduce 异步化并与计算重叠。

    **CUDA Stream 层面的执行调度**：
    ```python
    # 伪代码：FarSkip 在 vLLM 中 MoE 层的实现
    # 主计算 stream (default stream)
    with torch.cuda.stream(compute_stream):
        gate_out = router(hidden_states)
        expert_out = fused_moe(hidden_states, gate_out, expert_weights)
    
    # 异步 all-reduce 在独立 stream 上运行
    with torch.cuda.stream(comm_stream):
        # async_op=True: 启动后立即返回 handle
        all_reduce_handle = torch.dist.all_reduce(
            expert_out, async_op=True
        )
    
    # 不等待 all-reduce，立即进入下一层的 attention 计算
    # （FarSkip 架构保证 attention 输入不需要完整的 expert_out）
    attn_out = attention(hidden_states)  # 与 all-reduce 重叠
    
    # 在需要完整 expert_out 之前同步
    all_reduce_handle.wait()
    final_out = hidden_states + attn_out + expert_out
    ```

    **性能结果**：
    - Llama-4-Scout (109B): all-reduce 重叠率 95.3%, TTFT 加速 12.2%-18.5%
    - DeepSeek-V2 (235B): all-reduce 重叠率 97.6%, TTFT 加速 8.2%-16.8%
    - DeepSeek-V3 (671B) SGLang: TTFT 加速 up to 1.34× (TP=8, EP=8)
    - 多节点 decode (TP=16, EP=16, BS=1024): 显著且一致的 TBT 加速（Fig. 7）

## HarMoEny: Efficient Multi-GPU Inference of MoE Models

- 属于Serving调度的实现是什么？实验比较什么？
  - HarMoEny 实现了一个面向多 GPU MoE 推理的动态负载均衡 Serving 调度系统，核心调度优化包括：
    1. **Dynamic Token Rebalancing（Section 4.2, Algorithm 2）**：每个 batch 中所有 GPU 交换轻量级 metadata（token-to-expert 分配摘要，约 4kB），构建全局 token 分布视图。贪心调度算法确定性地识别最过载 GPU g_max 和最大贡献源 GPU g_from，将 token 从过载 GPU 重路由到欠载 GPU g_min，逐步迭代直至负载平衡。引入 token threshold q 控制最小 offload 粒度。
    2. **Asynchronous Expert Prefetching（Section 4.3）**：当 token rebalancing 将 expert 重分配到未持有该 expert 的 GPU 时，通过独立 CUDA stream 异步从 system memory 预取 expert 权重，直接覆写已完成的 expert 内存位置（无需先写回 system memory），overwrite 加速 5.5×（11ms→2ms on V100）。
  - 实现量：1115 行 PyTorch 代码，使用 NVIDIA CUDA stream 实现异步 expert 加载。MoE 层实现为 PyTorch nn.Module，可应用于任意 PyTorch 模型。
  - 实验比较：DeepSpeed（Tutel enabled, round-robin EP）、FastMoE、FasterMoE（dynamic shadowing）、ExFlow（integer programming expert placement）共 4 个 baseline。
  - 指标：Throughput（tokens/s）、Mean TTFT（time-to-first-token）。ablation study 额外包含 time breakdown（CUDA Events 细粒度分析）和不同 load balancing policies 对比（Round-robin、ExFlow policy、Even Split）。

- 硬件平台是什么，配置是什么。
  - NVIDIA DGX1 机器，8× NVIDIA V100 GPU（每 GPU 32GB VRAM），NVLink 互联，500 GB system memory。

- 开源Serving框架是什么。修改了什么。
  - HarMoEny 直接基于 PyTorch 实现（不基于已有 serving 框架修改），开源在 https://github.com/sacs-epfl/HarMoEny。
  - 核心修改/新增（作为 PyTorch nn.Module 插入现有模型）：
    1. **MoE Layer 重写**：Algorithm 1 定义的 6 步 MoE forward 流程——token routing → metadata exchange → token scheduling (rebalancing) → scatter tokens → expert processing + async loading → gather tokens。替代标准 MoE 层的 "router → all-to-all dispatch → expert FFN → all-to-all combine" 流程。
    2. **Token Scheduler**：Algorithm 2 的贪心负载均衡调度器，在 metadata exchange 后同步计算出全局最优 token-to-GPU schedule S。
    3. **Expert Prefetching 协议**：异步 expert 权重传输——当一个 expert 完成计算后，立即检查是否有下一个需要运行但未加载的 expert，通过独立 CUDA stream 异步 prefetch 权重直接覆写已完成 expert 的内存。
    4. **Configurable Router Skew**：可配置的人工 expert 流行度偏斜机制（参数 α ∈ [0,1]），支持可控的 token 分布倾斜实验。
  - 需要 Gurobi license 来运行 ExFlow baseline 对比实验（ExFlow 使用 integer programming）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：代码完整开源在 https://github.com/sacs-epfl/HarMoEny（212 commits, 82.5% Python, 16.5% Shell, 1.0% Dockerfile）。包含 Docker 支持、EC2 setup 脚本、experiments 目录下的可执行实验脚本。
  - **HarMoEny MoE 推理全流程（以 Switch128, 8×V100, batch inference 为例）**：

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 模型加载与配置                                                │
│    PyTorch 加载 Switch128 (12 MoE layers, 128 experts/layer)      │
│    - Self-Attention + Router: 复制到所有 8 GPU (data parallelism) │
│    - Experts: 初始 round-robin 分布到 8 GPU (EP=8)               │
│    - 每 GPU 持有 16 个 expert (128/8)，每 expert 18MB             │
│    HarMoEny 通过 replace_moe_layer() 注入自定义 MoE 层            │
│           ↓                                                      │
│ 2. 用户输入 batch tokens [B tokens]                              │
│           ↓                                                      │
│ 3. 每层 MoE forward (Algorithm 1, 6 steps)                       │
│    ┌─ Step 1: Token Routing ─────────────────────────────────┐   │
│    │  各 GPU 独立计算 self-attention → Router(W_gate @ h)     │   │
│    │  → m_expert: token-to-expert assignment tensor           │   │
│    │  (各 GPU 复制执行, 无通信)                                │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Step 2: Metadata Exchange ────────────────────────────┐   │
│    │  SENDMETADATATOGPUs(m_expert)                           │   │
│    │  每 GPU 广播本地 token-expert 分配 (~4KB metadata)       │   │
│    │  → m_all: 全局 token-to-expert assignment                │   │
│    │  (negligible overhead)                                   │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Step 3: Token Scheduling (Algorithm 2) ───────────────┐   │
│    │  S_initial = INITIALASSIGN(m_all)                       │   │
│    │  S = REBALANCE(S_initial):                               │   │
│    │    t_avg = total_tokens / |G|                            │   │
│    │    while any GPU has tokens > t_avg:                     │   │
│    │      g_max = most overloaded GPU                        │   │
│    │      g_from = GPU contributing most tokens to g_max     │   │
│    │      e_max = expert from g_from sending most to g_max   │   │
│    │      t_move = tokens to transfer                         │   │
│    │      if t_move < q: stop (insufficient to amortize)     │   │
│    │      g_min = least loaded GPU                           │   │
│    │      transfer min(t_move, t_avg - t_g[g_min]) tokens    │   │
│    │      from e_max on g_from → g_min                        │   │
│    │  → S: rebalanced 3D schedule [src_GPU, expert, dst_GPU] │   │
│    │  (各 GPU 独立并行计算, 因 metadata 全局一致, 结果相同)    │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Step 4: Scatter Tokens ───────────────────────────────┐   │
│    │  SENDTOKENSTOGPUS(x, m_expert, S)                       │   │
│    │  All-to-all communication: 各 GPU 按 rebalanced S 发送  │   │
│    │  token 到目标 GPU → receive x' from all other GPUs      │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Step 5: Expert Processing + Async Loading ────────────┐   │
│    │  for each expert e assigned to this GPU:                │   │
│    │    if e not in GPU memory:                              │   │
│    │      async CUDA stream: copy e weights from sys mem     │   │
│    │      overwrite completed expert's memory (5.5× faster)  │   │
│    │      18MB / (PCIe bandwidth) ≈ 2ms (V100)              │   │
│    │    compute: x''_e = e(x'_e)  // expert FFN forward      │   │
│    │    (async prefetch overlaps with current expert compute)│   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Step 6: Gather Tokens ────────────────────────────────┐   │
│    │  SENDTOKENSBACKTOGPUs(S, x'')                           │   │
│    │  All-to-all communication: 各 GPU 返回处理后的 token    │   │
│    │  → RECONSTRUCT(S, y, m_all): 恢复原始 token 顺序       │   │
│    └────────────────────────────────────────────────────────┘   │
│ 4. 输出: next token logits → generated tokens                   │
│    90% skew workload: 186 tok/s (vs 106 tok/s ExFlow, +75%)     │
│    无 skew workload: 213 tok/s (vs ~200 tok/s baselines)        │
└─────────────────────────────────────────────────────────────────┘
```

  - **Token threshold q 的数学推导（Section 4.4）**：
    q 由硬件规格决定，与动态 workload 无关：q > φ·d_type / (2·β)
    其中 φ = GPU FLOPS, d_type = 元素字节数, β = PCIe 带宽。
    物理含义：确保 expert 计算时间 > expert 加载时间，使 prefetch 可被计算掩盖。

  - **关键性能数据（Table 1 model specs）**：
    | Model | MoE Layers | Experts | Expert Size |
    |-------|-----------|---------|-------------|
    | Switch128 | 12 (alternating) | 128 | 18 MB |
    | Qwen 1.5 MoE | 24 | 60 | 33 MB |

  - **Throughput 对比（Switch128, 90% skew, Constant dataset）**：
    | System | Throughput | HarMoEny Speedup |
    |--------|-----------|-----------------|
    | HarMoEny | 186 tok/s | 1.0× |
    | ExFlow | 109 tok/s | 1.7× |
    | FasterMoE | 109 tok/s | 1.7× |
    | FastMoE | 124 tok/s | 1.5× |
    | DeepSpeed | 20 tok/s | 9.1× |

  - **Ablation: time breakdown（Switch128, 90% skew, MoE layer 1）**：
    - No rebalancing: mean latency 289ms, GPU idle 82.6%
    - Rebalancing only: mean latency 149.5ms (-48.3%), GPU idle 2.6%
    - Rebalancing + async prefetch: mean latency 136.6ms (-8.6% over sync)
    - Scheduler overhead: 30.8% of mean latency (Switch128), 20.3% (Qwen)

  - **Real-world datasets throughput (Switch128)**：
    HarMoEny: 201 tok/s (steady across datasets)
    FasterMoE/FastMoE: 92-98% of HarMoEny throughput
    ExFlow: inconsistent due to inability to adapt to dynamic skew
    DeepSpeed: very low due to input padding strategy

## GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - GRACE-MoE 在 Megablocks 上构建了一套面向多节点 SMoE 推理的 Serving 调度系统，核心调度/系统优化包括：
    1. **Hierarchical Sparse Communication (HSC, Section 5)**：替换 flat global All-to-All 为 physically global but logically sparse 的两阶段通信方案——Stage 1 跨节点路由（每个 GPU 与远端节点 peer GPU 通信，将 token 转发到目标节点，dest node 相同的 token 仅传输一次）；Stage 2 节点内重分发（GPU 间 token 经 NVLink 传输到 expert 所在 GPU）。跨节点使用单一 global communication group + zero-padding 实现 logically sparse point-to-point transfers，保留 sparse communication 的带宽优势并利用 global collective 的 implicit barrier 做 soft synchronization。跨节点通信与节点内 routing decision computation 通过细粒度 pipelining 重叠。
    2. **Offline-Online Coordinated Scheduling**：Offline 阶段（profiling → grouping → replication）生成 expert placement plan 和 replica map；Online 阶段（HSC + topology-aware routing）按 plan 执行 token dispatch 和 replica selection。Offline 结果可跨 dataset 复用（cross-dataset placement 最差 latency 增加 ≤4.52%），避免频繁 re-profiling。
    3. **Multi-Node Multi-GPU Synchronization Reduction**：通过 HSC 的 implicit barrier 机制消除 explicit global barrier，结合 locality-aware routing 优先使用本地/节点内副本减少跨节点通信，缓解 straggler effect 和 synchronization overhead。
  - 实验比较：(1) 端到端 inference latency 和 MoE layer time：GRACE-MoE vs Tutel, Megablocks, vLLM, C2R, Occult；(2) 六种组件增量配置的通信/负载/延迟指标分解（Table 1, Figure 5）；(3) Cross-dataset transfer generalizability；(4) Lighter workloads（batch=64/128）下的稳定性。

- 硬件平台是什么，配置是什么。
  - 2 节点，每节点 4× NVIDIA A100-SXM4 GPU (80GB)。节点内 NVLink（12 links/GPU, 50 GB/s per direction）。节点间 25 Gbps Ethernet（模拟实际有限跨节点带宽）。
  - 软件：Megablocks (Gale et al. 2023) + PyTorch 2.5 + Triton 3.1，支持 data parallelism + expert parallelism。

- 开源Serving框架是什么。修改了什么。
  - 基于 Megablocks（https://github.com/databricks/megablocks），一个基于 block-sparse matrix multiplication 的 MoE 计算框架。
  - GRACE-MoE 修改/新增内容：
    - **HSC 通信模块**：替换 Megablocks 原有的 flat All-to-All 为 hierarchical sparse communication 实现。Cross-node 部分使用 global collective group + zero-padding sparse point-to-point；Intra-node 使用节点内高带宽链路做 token redistribution。Cross-node 通信与 intra-node routing computation fine-grained pipelining。
    - **Offline Profiling + Grouping 模块**：基于 spectral clustering 实现 hierarchical grouping（跨节点 fully non-uniform + 节点内 controlled non-uniform），生成 expert placement map。
    - **Dynamic Replication 模块**：基于 load skew factor ρ 计算每层 replica 数，选择 heaviest group 中 hot experts 复制到 underutilized GPUs。
    - **Online Routing 模块**：Topology-aware routing with locality preference（三级优先 + WRR with load prediction），集成到 Megablocks 的 token dispatch 路径中。
    - **数据并行 + Expert 并行**：保持 Megablocks 原有 DP+EP 能力，在其上构建 HSC + routing。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文声明 "code will be released upon acceptance"，截至分析时未发现公开仓库。基于开源 Megablocks。GRACE-MoE 作为 Megablocks 的上层调度优化，不修改底层 expert kernel。
  - Serving 框架执行全过程（以 OLMoE 6.92B, 2 nodes×4 GPUs/node, batch=512, prefill=64, decode=32 为例）：

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Offline 准备阶段                                             │
│    - 在 calibration data (WikiText-2) 上 profiling              │
│    - 记录每层每个 expert 的共激活频率 → affinity matrix A[l]    │
│    - Hierarchical Grouping:                                     │
│      Cross-node (fully non-uniform): 64 experts→2 node groups   │
│      Intra-node (controlled non-uniform, r=0.15): 每组→4 GPU groups│
│    - Dynamic Replication: 每层根据 ρ 确定副本数和 hot experts   │
│    - 生成 expert placement plan + replica map                    │
│           ↓                                                      │
│ 2. 模型加载                                                     │
│    - Megablocks 加载 MoE 模型权重到各 GPU                        │
│    - 按 placement plan 分布 expert，按 replica map 复制 hot     │
│      expert 权重到 secondary GPUs                               │
│    - Attention/Embedding/Norm 常驻各 GPU (DP 复制)               │
│           ↓                                                      │
│ 3. 用户输入 batch tokens [B=512, S=64]                          │
│    Prefill + Decode loop:                                        │
│           ↓                                                      │
│ 4. 每层 MoE 执行（以单层为例）                                   │
│    ┌─ Router/Gating ────────────────────────────────────────┐   │
│    │  gate_logits = W_gate @ h  [512×64, 64 experts]        │   │
│    │  topk_indices, topk_weights = topk(softmax(logits), k) │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ HSC Stage 1: Cross-node Token Forwarding ─────────────┐   │
│    │  每个 GPU 扫描所有 expert indices                        │   │
│    │  对 dest node 相同的 token：聚合为单次 cross-node send  │   │
│    │  Global collective group + zero-padding → sparse P2P    │   │
│    │  跨节点 traffic 仅含必要的 token data（去重后）          │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ HSC Stage 2: Intra-node Redistribution (与 Stage 1     │   │
│    │  routing decision computation pipelined) ───────────────│   │
│    │  节点内各 GPU 经 NVLink (50GB/s×12) P2P 传输 tokens     │   │
│    │  Token 精准路由到 expert 所在 GPU                        │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Locality-Aware Routing (on replica selection) ────────┐   │
│    │  for each token → expert e with replicas:               │   │
│    │    if token GPU has replica: use local (intra-GPU)      │   │
│    │    elif intra-node replica exists: WRR w/ load pred     │   │
│    │    else: cross-node WRR fallback                        │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Expert FFN Computation ───────────────────────────────┐   │
│    │  Megablocks block-sparse matmul:                        │   │
│    │  gate_out = SiLU(W_gate @ x) * (W_up @ x)              │   │
│    │  expert_out = gate_out @ W_down                        │   │
│    │  output = sum(topk_weights * expert_out)                │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ HSC Combine (对称反向) ───────────────────────────────┐   │
│    │  Intra-node gather → cross-node combine (global group)  │   │
│    │  Token output 返回到原 GPU，reassemble 序列              │   │
│    └────────────────────────────────────────────────────────┘   │
│           ↓                                                      │
│ 5. 输出: generated tokens                                       │
│    End-to-end latency 相比 Occult 降低 up to 78.55%              │
│    4.66× speedup (OLMoE, 2 nodes×4 GPUs)                        │
└─────────────────────────────────────────────────────────────────┘
```

  - HSC 的关键设计优势：flat global All-to-All 需要 strict synchronization across all ranks，heterogeneous 集群中受最慢链路限制（straggler effect）。HSC 通过 global collective 的 implicit barrier 做 soft sync + logically sparse transfer，消除 explicit barrier 开销。Cross-node traffic 通过 token deduplication（同一 dest node 多 token 聚合单次传输）进一步减少。
  - Component analysis (Table 1, vs Occult baseline):
    - Occult: uniform grouping + flat All-to-All
    - Occult+HSC: All-to-All time −35.19%, GPU idle −49.88%
    - HG+HSC: All-to-All time −48.33%, cross-node traffic −50.67%, GPU load std +90.03%
    - +DR+WRR: GPU idle −26.86%, GPU load std +31.92%
    - +DR+TAR (full GRACE-MoE): All-to-All time −50.57%, cross-node traffic −52.11%, GPU idle −25.66%

## HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission

- 属于Serving调度的实现是什么？实验比较什么？
  - HybridEP 修改 Tutel 分布式 MoE 训练框架的 Expert Parallelism (EP) 通信模式，实现跨数据中心场景下的高效 EP 扩展，核心实现包括三部分：
    1. **Stream-Based Modeling（流建模）**：将 MoE 训练过程解耦为计算流（GeMM 建模：Attention + FFN + Expert）和通信流（A2A + AG），通过重叠建模分析两流之间的关系，推导出以最小化训练延迟为目标的优化问题，求解最优的 A2A/AG 混合比例 p。当 p=1 时退化为标准 EP（纯 A2A），当 p=0 时全部使用 AG 通信。模型根据配置自动判断三种 case（Case 1: 混合 A2A+AG，Case 2.1: 混合 A2A+AG，Case 2.2: 仅 AG）。
    2. **Domain-Based Partition（域分区）**：定义 Expert Domain（专家域）为一组仅使用 AG 通信的 DC 集合。遵循"域内用 AG、域间用 A2A"的规则，通过 Multilevel Description（多层级架构抽象，用 scaling factor SF^i 表示层级扩展）→ Location Renumbering（GPU 全局编号按 PyTorch 格式重映射到多层级位置）→ Topology Construction（Algorithm 1: 通信拓扑构建，对每对 GPU 逐层判断 AG/A2A/None）三步将通信模式映射到 GPU 级别的具体通信拓扑，确保与现有层级硬件架构兼容。
    3. **Parameter-Efficient Migration（参数高效迁移）**：通过 SR-Based Expert Compression 和 Asynchronous Communicator 实现轻量级专家迁移。压缩分为 SREncode（共享+残差分离 → Top-k 稀疏压缩）和 SRDecode（恢复+加法 fused）两阶段。异步通信器分 Initialization 阶段（SREncode 与上一 iteration optimizer step 融合）和 Asyn-comm 阶段（AG 通信与 pre-expert computation 重叠）。压缩比最高 50× 不损失精度。
  - 实验比较：
    - Baselines：Tutel、FasterMoE、SmartMoE（均针对 MoE HPC 环境优化）
    - 建模验证：computation/communication latency 估计精度、最优 p 搜索（4 种 case 各 4 个候选 p）
    - 端到端加速比：不同 data traffic (6-192MB) 和不同 expert size (2-32MB) 下的 iteration 时间
    - 消融实验：Domain-Based Partition（baseline）vs +Parameter-Efficient Migration（完整 HybridEP）
    - 迁移分析：SR 压缩 loss 对比、时间分解（SREncode/SRDecode overhead 及 fusion 效果）
    - 特性对比：EP vs HybridEP 的通信流量（input-dependent vs fixed-bound traffic）和频率（A2A→AG 转换）
    - 大规模仿真：SimAI 模拟 1000 DCs 下的加速比（固定 S_ED vs 固定 p）

- 硬件平台是什么，配置是什么。
  - **Cluster-S**（单 DC）：8 × NVIDIA A800 GPU in a single node
  - **Cluster-M**（2 DCs）：16 × NVIDIA A800 GPU on 2 nodes（每个 node 视为一个 DC）
  - **Cluster-L**（4 DCs）：32 × NVIDIA A800 GPU on 4 nodes
  - 节点内互联：PCIe 3.0 x16 (128 Gbps)
  - 跨节点（DC 间）互联：Ethernet (10 Gbps)，模拟跨 DC 低带宽场景
  - 软件环境：Ubuntu 18.04, CUDA 11.3, cuDNN 7.6, NCCL 2.10
  - 大规模仿真平台：SimAI (USENIX NSDI 2025)
  - 优化器：Adam (lr=1e-4)，PyTorch DDP (All-Reduce 同步梯度)

- 开源Serving框架是什么。修改了什么。
  - 框架：**Tutel** (https://github.com/microsoft/tutel)，Microsoft 的自适应 MoE 分布式训练框架，支持多维分层 All-to-All 和动态 expert 分配。
  - HybridEP 代码未公开开源。论文未提供开源链接，在 Tutel + PyTorch v1.12.1 之上以原型系统实现。
  - 核心修改：
    1. **通信模式替换**：将 Tutel 的纯 A2A 通信替换为混合 A2A+AG 通信。在训练前通过 Stream-Based Modeling 根据环境配置（B, G, P_E, D, Lat_comp^PE）计算最优 p 值，决定 A2A/AG 比例。
    2. **Domain-Based Partition 新增**：新增 Expert Domain 管理层（Multilevel Description + Location Renumbering + Topology Construction Algorithm 1），将 model 输出的通信比例映射到 GPU 级通信拓扑。
    3. **Parameter-Efficient Migration 新增**：SR-Based Expert Compression（shared expert + residual Top-k 压缩，value-index 稀疏传输格式）+ Asynchronous Communicator（Send Queue/Recv Queue 管理，与 optimizer step 和 pre-expert computation 重叠执行）。
    4. **模型驱动配置**：训练前根据模型和环境参数自动计算最优 p 值和 expert domain 大小 S_ED^l。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - HybridEP 代码未公开开源。论文未提供开源链接。
  - HybridEP MoE 训练全流程（以 Cluster-L: 4 DCs × 8 GPUs, EP=32, p=0.5 为例）：

  ① **训练前：环境配置采集与建模求解**
     - 输入：G=32 GPUs, B_intra=128Gbps, B_inter=10Gbps, P_E, D, Lat_comp^PE
     - Stream-Based Modeling 求解：Formula (1)-(12) → 判断 2D - G·P_E 正负 → 确定 case → 输出最优 p
     - 根据 p 计算 S_ED：p = (G - S_ED) / G → S_ED = G(1-p)

  ② **训练前：Domain-Based Partition 构建拓扑**
     - Multilevel Description：4 DCs × 8 GPUs → 2 levels，SF^0=4, SF^1=8
     - 根据 optimal p 设定 S_ED^0, S_ED^1
     - Location Renumbering (Eq.13)：32 个 GPU 全局索引 m → (x_0, x_1)
     - Topology Construction (Algorithm 1)：对每对 GPU (m,n)，逐层判断通信类型——同域内且同 offset→AG；不同域且同 offset→A2A；否则 None

  ③ **训练 iteration 开始（Forward Pass）**
     - Token batch (B tokens) → Embedding → pre-expert 层（Attention + FFN），按 DP 分布在各 GPU
     - **Asyn-comm 阶段**（与 pre-expert computation 重叠）：
       - Send Queue 弹出上一 iteration SREncode 的压缩 expert 残差
       - NCCL All-Gather：域内 GPU 间收集压缩后的 expert 参数
       - Recv Queue 接收 → SRDecode 恢复完整 expert = shared_expert + decompress(residual)

  ④ **Gate Network + Expert Dispatch**
     - Gate 计算 routing weights → Top-K 选择 activated experts（每个 token 选 K 个 expert）
     - **域间 A2A**（仅对 p 比例的数据）：token data 按 routing result 通过 NCCL All-to-All 发送到对应 expert 所在 GPU
     - **域内 AG**（对 1-p 比例的数据）：对应的 token 不再通过 A2A 传输，因为 expert 已通过 AG 收集到本地

  ⑤ **Expert FFN 计算（GPU 本地）**
     - 每 GPU 对收到的 tokens 执行本地 experts 的 FFN 前向
     - W_gate GEMM → SiLU activation → W_up GEMM → element-wise multiply → W_down GEMM
     - cuBLAS GEMM kernel 在 NVIDIA A800 Tensor Cores 上执行

  ⑥ **Expert Combine（A2A 逆向）**
     - 域间 A2A 通信将 expert 输出按原 token 位置合并回各 GPU

  ⑦ **Parameter-Efficient Migration（与 optimizer step 融合）**
     - SREncode：计算 expert 残差 = expert - shared_expert → Top-k 压缩 → 稀疏 value-index 格式 → 存入 Send Queue
     - 与当前 iteration 的 optimizer step（Adam 参数更新）融合执行，减少 ~30% 的编码 overhead

  ⑧ **Backward Pass**
     - Expert FFN backward → A2A combine backward → A2A dispatch backward → Attention backward
     - Shared expert 梯度通过 All-Reduce 在 backward 阶段同步

  ⑨ **输出**：完成一层的 forward+backward，所有 experts 和 shared expert 参数更新完毕，Send Queue 已为下一 iteration 准备压缩后的 expert 残差，进入下一 iteration

## JANUS: Disaggregating Attention and Experts for Scalable MoE Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - JANUS 在 SGLang 上实现了面向 MoE 的解耦式推理系统，核心 Serving 调度实现包括三个层次：
    1. **Attention-MoE 解耦架构（Section 3.2-3.3）**：将 Attention 层和 MoE 层部署到独立的 GPU 子集群（attention nodes / MoE nodes），支持各层类型独立配置并行度。通过 Adaptive Two-Phase Communication 实现低延迟跨子集群数据交换——Phase 1: 同节点多个实例通过 NVLink 聚合中间激活；Phase 2: 聚合后的大块数据通过 GPUDirect RDMA 跨节点传输。根据资源配置和流量负载自适应选择两种传输模式：Case-1（直接点对点传输）或 Case-2（一对一中继 + 节点内 NVLink 多播）。Gating 放置在 MoE 侧以简化通信、避免 per-expert tensor packing。
    2. **Activated-Expert-Balanced Scheduling (AEBS, Section 3.4)**：轻量级 GPU kernel 在每 MoE 层执行激活专家调度——收集当前 batch 的 top-k 路由结果 → 将单副本 expert 固定分配到唯一持有实例 → 多副本 expert 贪心分配到当前负载最低的实例 → 重写路由结果为物理副本 ID → dispatch token。**无 CPU-GPU 同步、无跨 GPU 协调**，调度开销在微秒级。
    3. **Fine-Grained SLO-Aware Resource Scaling（Section 3.5）**：基于 Roofline 和 Little's Law 构建 TPOT 性能模型（Eq. 1-3），使用 Monte Carlo 估计 a_max（最大激活 expert 数），通过 bounded binary search 求解稳态 batch size，枚举 (n_a, n_e) 搜索空间选择满足 SLO 的最小 GPU 配置。同时优化 expert placement 以避免高频共激活 expert 被放置在同一实例（min-max co-activation load，Appendix B Algorithm 3）。
  - 实现量：~4K 行 Python + ~300 行 CUDA/C++，基于 SGLang。
  - 实验比较：
    - Baselines：SGLang（monolithic TP/EP）、MegaScale-Infer（解耦式，随机 expert 调度 + attention 侧 gating + 粗粒度 scaling）、xDeepServe（解耦式，EPLB 调度 + attention 侧 gating + 4 GPU 粒度 scaling）
    - 指标：TPOT（per-token SLO）、per-GPU Throughput（TPG）
    - 数据集：ShareGPT（avg 16 in + 256 out）、BurstGPT（合成动态到达）
    - 结果：JANUS 相比 SGLang/MegaScale-Infer/xDeepServe 分别提升 per-GPU throughput 最高 4.7×/2.2×/3.3×，满足 TPOT SLO

- 硬件平台是什么，配置是什么。
  - 最多 4 节点 GPU 集群：
    - 每节点：128 CPU cores，2TB host memory，8× NVIDIA H100 (80GB)
    - GPU 间互联：900 GB/s NVLink (intra-node)
    - 跨节点互联：400 Gbps InfiniBand NIC per GPU
  - 软件环境：SGLang + NVSHMEM + NCCL + GPUDirect RDMA

- 开源Serving框架是什么。修改了什么。
  - **Serving 框架**：SGLang (https://github.com/sgl-project/sglang)。
  - 核心修改：
    1. **解耦架构改造**：将 SGLang 的 monolithic 部署拆分为 attention instances 和 MoE instances 两组，每个 instance 运行于一块 GPU。Attention 侧复用 SGLang 的 request batching/dispatching/KV-cache 管理。MoE 侧每个 instance 持有 expert 子集。
    2. **Adaptive Two-Phase Communication**：使用 NVSHMEM 的 one-sided putmem_signal/signal_wait 原语实现跨子集群数据交换。Intra-node collectives 使用 NCCL。将 layer index + token count 等元数据打包进 signal value 避免单独传输。共享 expert 放置在 attention 侧，与跨子集群通信重叠执行。
    3. **AEBS Scheduler**：GPU kernel 实现，每 MoE 层在每个 MoE instance 上独立执行（synchronization-free），通过冗余计算避免跨 GPU 协调。
    4. **Scaling Controller**：MoE controller + attention controller 周期性（15 min 间隔）收集 activation statistics，运行 Algorithm 2 搜索最优 (n_a, n_e)，增量调整实例数量。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未明确给出 JANUS 开源链接，基于 SGLang 实现。
  - **JANUS Serving 框架执行全过程（DeepSeek-V2 decode 阶段，1A6E 配置为例）**：

    ```
    ① 输入：用户请求到达 → Attention Controller 将请求分发到 n_a=1 个 attention instance (GPU A)。
    
    ② Attention Instance (GPU A)：
       - SGLang continuous batching 管理 in-flight decode batch B。
       - 执行 Attention 层 (MLA)：KV cache 常驻 GPU A HBM → Q/K/V 计算。
       - 执行 Shared Expert（overlap 阶段）：在等待 MoE 结果时计算共享 expert FFN。
    
    ③ Cross-Sub-Cluster Communication (每 MoE 层)：
       注意: Attention instance 在 GPU A, MoE instances 在 GPU E0–E5 (共 n_e=6)
       - Phase 1 (Intra-node 聚合): 若同一 attention node 有多个 attention instances，
         通过 NVLink NCCL 聚合中间激活 → 减少跨节点传输次数。
       - Phase 2 (Inter-node 传输): GPU A 通过 NVSHMEM putmem_signal (GPUDirect RDMA)
         将激活直接写入 MoE instance GPU 的接收 buffer → signal 通知完成。
         元数据 (layer index, token count) 打包在 signal value 中。
       - 根据配置选择 Case-1 (直接点到点) 或 Case-2 (一比一中继+多播)。
    
    ④ MoE Gating (MoE 侧 GPU E0–E5)：
       - 接收完整 activation → Router softmax(W_gate·h) → Top-K routing。
       - 每 GPU 独立运行 AEBS GPU kernel:
         → 收集当前 batch 所有 token 的激活逻辑 expert IDs
         → 单副本 expert 固定分配；多副本 expert 贪心选负载最低实例
         → 重写每个 token 路由为物理 replica IDs
         → Dispatch token 激活到持有对应 replica 的 GPU
       - AEBS 开销: <90μs (batch=4096, 16 MoE instances)
    
    ⑤ Expert FFN (MoE 侧 GPU E0–E5)：
       - 各 GPU 对收到的 tokens 执行本地 expert FFN (GEMM via cuBLAS)。
       - MoE 层延迟由 max(a_max) 决定——即激活 expert 数最多的 GPU 决定。
    
    ⑥ Combine (反向两阶段通信)：
       - Phase 1: MoE 侧 intra-node all-reduce 聚合中间结果。
       - Phase 2: NVSHMEM putmem_signal 将结果传回 attention GPU。
    
    ⑦ 输出：Attention GPU 完成 Shared Expert + MoE output 的 residual add → next token。
    
    ⑧ Scaling Loop (15 min 间隔)：
       - MoE Controller 收集各层 activation statistics。
       - 用 recent trace 构建 Monte Carlo â_max 查找表。
       - Algorithm 2: 枚举 (n_a, n_e) 搜索空间 → 求解 Eq. (2) 稳态 B* → 检查 TPOT SLO + memory feasibility → 选择 min(n_a+n_e)。
       - 增量调整实例数: 添加/移除 attention 和 MoE instances → 重新运行 expert placement (Algorithm 3, 最小化 co-activation 共现)。
    ```

    关键性能数据（DeepSeek-V2, H100）：
    | SLO | JANUS TPG (tok/s/GPU) | vs SGLang | vs MegaScale | vs xDeepServe |
    |-----|----------------------|-----------|-------------|---------------|
    | 200ms | 最高 | up to 4.7× | up to 2.2× | up to 3.3× |
    | 150ms | 最高 | up to 4.7× | up to 2.2× | up to 3.3× |
    | Dynamic trace (24h) | — | 39% GPU-h节省 | 16% GPU-h节省 | — |

## Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts

- 属于Serving调度的实现是什么？实验比较什么？
  - LLEP 提出一种新的 Expert Parallelism (EP) 负载均衡算法，核心实现是动态将超载 GPU 的多余 token 和 expert 权重路由到欠载 GPU，保证所有 GPU 在最短集体延迟内完成计算并满足内存约束。具体实现包括：
    1. **Least-Loaded Assignment (LLA) 算法（Alg. 2）**：按 expert 负载降序，对每个 expert 判断其原生 GPU 是否可容纳所有 token。若不能，则通过 LLAS（Alg. 3）子程序将多余 token 溢出（spill）到负载最轻的 GPU，直至全部 token 分配完毕。同时构建对应的权重传输计划（weight transfer plan）。
    2. **自适应阈值 λ 切换**：当全局 expert 负载的 max/mean 比值低于 λ 时，LLEP 回退到标准 EP，因为此时负载已足够均衡，LLA 的额外开销不经济。
    3. **Backward-pass 支持**：反向传播时，溢出 expert 权重的梯度会被传回其原生设备，与原生梯度累加。LLEP 保持 MoE 的精确数学计算（exact computation），不改变模型行为。
    4. **容量与效率约束**：α 因子控制每 GPU 最大 token 容量（m_α = α · Σl_i / P），m 为每个 GEMM 的最小 token 数以保持计算效率。低于 m 的溢出量被强制本地计算（避免低效的微小 GEMM）。
  - 实验比较：
    - **受控实验（Section 5.1）**：LLEP vs Standard EP，在 gpt-oss-120b、DeepSeek-V3、Kimi-K2 三种 MoE 配置下，模拟 30%-95% token 集中到 1/4/16 个 expert 的不均衡场景。指标：Forward pass speedup 和 peak memory per GPU。
    - **端到端实验（Section 5.2）**：LLEP vs Standard EP，使用真实预训练的 gpt-oss-20b 和 gpt-oss-120b，在 Megatron-Math 数据集上测量全模型吞吐量。训练实验用 gpt-oss-20b 全参数 SFT（Zero-3 + CPU offloading），指标为 AIME'25 accuracy vs wall-time。
    - **消融实验（Section 5.3）**：消融 batch size B、factor α、adaptive ratio λ、hidden size D/H、expert 数量 N 对 speedup 的影响。
    - 结果：MoE 层最高 6.1× speedup（gpt-oss-120b, 95%→1 expert），峰值内存降低 5×。端到端全模型 gpt-oss-120b 加速 1.88×，gpt-oss-20b 加速 2.2×。

- 硬件平台是什么，配置是什么。
  - **受控实验（Section 5.1）**：8× NVIDIA H200 GPU（单节点），batch size per GPU：32K tokens (gpt-oss) / 16K tokens (DeepSeek-V3, Kimi-K2)。
  - **端到端实验（Section 5.2）**：
    - gpt-oss-20b：1-8 GPU（扩展性测试），论文未明确 GPU 型号（使用与受控实验相同硬件，H200）。
    - gpt-oss-120b：多 GPU 扩展性测试。训练实验使用 Zero-3 + CPU offloading for gradients and optimizer states。
  - **消融实验（Section 5.3）**：NVIDIA H200 GPU。论文未明确说明 CPU、内存、互联（NVLink/NVSwitch）等具体配置，但使用 PyTorch NCCL 后端进行 GPU 间通信。

- 开源Serving框架是什么。修改了什么。
  - LLEP **不修改现有 Serving 框架**（如 vLLM、SGLang），而是在 PyTorch 分布式训练/推理基础上实现独立的 EP 负载均衡模块。
  - 基线框架：PyTorch 分布式（torch.distributed）+ NCCL 后端，用于标准 EP 的 All-to-All 和 P2P 通信。
  - 修改/新增内容：
    1. **LLA/LLAS 算法模块**：纯 Python 实现，接收全局 expert token 负载统计，输出 token 分配计划和权重传输计划。不修改 PyTorch autograd 或 NCCL primitive。
    2. **LLEP dispatch_combine 流程（Alg. 4）**：在标准 EP 的 dispatch-combine（Alg. 1）基础上新增：λ 阈值判断 → LLA 规划 → 构建含 foreign expert chunk 的 All-to-All → P2P 权重传输 → 对 native + foreign experts 执行 Grouped-GEMM → All-to-All reverse 结合。
    3. **权重传输**：通过 NCCL P2P（peer-to-peer）操作在 GPU 间传输 expert 权重矩阵，仅在 LLA 判定需要溢出时触发。
    4. **Backward 梯度回流**：溢出 expert 的反向梯度通过 P2P 传回原生 GPU 并与本地梯度累加，确保训练正确性。
    5. **可优化方向（论文提出但未实现）**：C++/Triton kernel 融合通信操作、避免 memory-intensive index select 的直接 All-to-All on unsorted tensors、compute/communication overlap、intra-node 优先的 spill 策略。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：代码开源在 https://github.com/SalesforceAIResearch/LeastLoadedEP
  - **LLEP 执行全过程（以 8×H200, gpt-oss-120b MoE layer, 128 experts / 4 active experts, EP=8, 每 GPU 16 experts 为例）**：

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 输入 & Router 阶段（所有 GPU 本地执行）                       │
    │    每 GPU 持有 B_p=32K tokens，hidden dim D, K=4                 │
    │    Router 计算: s_i = softmax_i(u^T W_r) → top-K gate weights   │
    │    → 全局收集 per-expert token 负载 l ∈ R^N (含 128 experts)    │
    │           ↓                                                      │
    │ 2. 自适应判断（Alg. 4, Line 214）                               │
    │    if max(l) / mean(l) < λ (=1.3):                              │
    │      执行标准 EP (Alg. 1)，跳过 LLA                             │
    │      → 正常 dispatch-combine → 输出                             │
    │    else:                                                         │
    │           ↓                                                      │
    │ 3. Least-Loaded Assignment (Alg. 2, CPU 侧 Python 执行)          │
    │    - 按 expert 负载降序排序 l → 最大负载 expert 先分配           │
    │    - m_α = α × Σl_i / P = 1 × total_load / 8 (每GPU容量上限)    │
    │    - 对每个 expert i:                                            │
    │      · Case 1: 原生 GPU 容量足够 → 全部分配给原生 GPU            │
    │      · Case 2: 原生 GPU 部分容纳 → 容纳部分，剩余溢出到最轻GPU   │
    │      · Case 3: 原生 GPU 已满 → 全部溢出到最轻GPU                 │
    │    - LLAS (Alg. 3): 按 g_a[g]+g_p[g] 排序其他GPU, 贪心分配      │
    │    - 输出: A (token 分配计划) + W (权重传输计划)                 │
    │           ↓                                                      │
    │ 4. LLEP Dispatch（Alg. 4, Lines 217-226）                       │
    │    ┌─ Token 重排 ───────────────────────────────────────────┐   │
    │    │  按路由索引 sort → chunk B_p 和 G_p 到 per-GPU 段       │   │
    │    │  每个 GPU 的 chunk 含: native expert tokens +            │   │
    │    │  foreign expert tokens (由 LLA 分配)                     │   │
    │    └───────────────────────────────────────────────────────┘   │
    │    ┌─ All-to-All Dispatch (NCCL) ───────────────────────────┐   │
    │    │  GPU p 发送: {B_i, G_i | i ∈ assigned experts}         │   │
    │    │  GPU p 接收: {B̂_i, Ĝ_i | i ∈ [pM,(p+1)M-1] ∪ S}       │   │
    │    │  其中 S = foreign experts assigned to GPU p              │   │
    │    └───────────────────────────────────────────────────────┘   │
    │    ┌─ P2P 权重传输 (NCCL P2P) ──────────────────────────────┐  │
    │    │  for each j ∈ S (foreign experts):                      │  │
    │    │    W_j: GPU_native(j) → GPU_p (via P2P Send/Recv)       │  │
    │    │    传输量: D × H per expert (如 8192×8192=64M floats)    │  │
    │    └───────────────────────────────────────────────────────┘   │
    │           ↓                                                      │
    │ 5. Grouped-GEMM 计算（Alg. 4, Line 229）                        │
    │    ┌─ GPU p 上的 GEMMs ─────────────────────────────────────┐   │
    │    │  for expert i ∈ native experts ∪ S:                     │  │
    │    │    Ĥ_i = Ĝ_i ⊙ B̂_i W_i  (SwiGLU: 3×GEMM per expert)    │  │
    │    │  使用 cuBLAS 独立 GEMM kernel (非 fused grouped-GEMM)    │  │
    │    │  cuBLAS 的硬件优化 GEMM 比 Triton grouped-GEMM 更快     │  │
    │    └───────────────────────────────────────────────────────┘   │
    │           ↓                                                      │
    │ 6. LLEP Combine（Alg. 4, Lines 231-237）                        │
    │    ┌─ All-to-All Reverse (NCCL) ────────────────────────────┐   │
    │    │  将专家输出 {Ĥ_i} 传回各 token 的原始设备               │   │
    │    └───────────────────────────────────────────────────────┘   │
    │    ┌─ Token 还原 ───────────────────────────────────────────┐   │
    │    │  concat({H_i}) → reverse_sort → reshape(B_p, K, H)     │   │
    │    │  H'_p = sum over K dim                                 │   │
    │    └───────────────────────────────────────────────────────┘   │
    │           ↓                                                      │
    │ 7. Backward Pass（若训练）                                       │
    │    - 梯度反向传播：All-to-All reverse (forward combine)          │
    │      → expert FFN backward → All-to-All reverse (forward dispatch)│
    │    - Foreign expert 梯度: P2P 传回原生 GPU 与原生梯度累加       │
    │    - LLA 算法不在计算图中，不影响 autograd                       │
    │           ↓                                                      │
    │ 8. 输出: MoE layer 输出 H'_p                                     │
    └─────────────────────────────────────────────────────────────────┘
    ```

    **关键性能数据**：
    | 场景 | 模型 | 不平衡度 | LLEP Speedup | Memory 节省 |
    |------|------|---------|-------------|------------|
    | 受控实验 | gpt-oss-120b | 95%→1 expert | 6.11× | 5× |
    | 受控实验 | DeepSeek-V3 | 95%→1 expert | ~4.5× | ~3× |
    | 受控实验 | Kimi-K2 | 95%→1 expert | ~4× | ~3.5× |
    | 端到端 | gpt-oss-120b | 自然分布 | 1.88× | — |
    | 端到端 | gpt-oss-20b | 自然分布 | 2.2× | — |
    | 训练 (SFT) | gpt-oss-20b | 自然分布 | 1.25× | — |

## LongCat-Flash Technical Report

- 属于Serving调度的实现是什么？实验比较什么？
  - LongCat-Flash 推理系统提出多项 Serving 调度创新：
    1. **Single Batch Overlap (SBO) 四阶段 Pipeline**：利用 ScMoE 架构在单个 batch 内实现 module-level computation-communication overlap。Stage 1: MLA_0 单独执行（作为后续阶段的输入）；Stage 2: All-to-All Dispatch 与 Dense FFN + Attn_0 (QKV Projection) 重叠；Stage 3: MoE GEMM 独立执行；Stage 4: Attn_1 (Core Attention + Output Projection) + Dense FFN 与 All-to-All Combine 重叠。SBO 区别于 DeepSeek-V3 的 TBO（需要两个 batch 做重叠），在单 batch 内即可隐藏通信。
    2. **PD-Disaggregated 架构**：Prefill 和 Decode 分离部署，层级别 KV cache 传输减少 TTFT。最小部署单元为 2 nodes × 16 H800-80GB GPUs。
    3. **MTP Speculative Decoding**：单一 dense layer 的 MTP 作为 draft model，接受率约 90%。采用 C2T (Classifier-based Tree Construction) 过滤低接受概率 token 减少 verification 延迟。MTP 选择 dense layer（1.41% params, 92.1% accept rate）而非 ScMoE layer 以优化 draft-to-target cost ratio。
    4. **Multi-Step Overlapped Scheduler**：通过 TVD fusing (Target forward + Verification + Draft forward 融合为单个 CUDA Graph) 减少 kernel launch overhead。进一步引入 multi-step overlapped scheduler，单次 schedule iteration 预启动多个 forward step 的 kernel，隐藏 CPU scheduling 和同步延迟。需动态预分配 KV cache slots，通过数学归纳法保证 KV cache 分配收敛在 [2n, 3n] 范围内（n 为预启动步数）。
    5. **Wide EP Deployment with DeepEP**：修改 DeepEP 和 EPLB 支持 zero-computation experts（zero-comp experts 输出无需通信即可获得），避免传输 identity 结果。EP 可扩展到上千 GPUs 以降低 MoE GEMM 延迟。
    6. **TP Deployment for Dense FFN**：ScMoE 中 Dense FFN intermediate dim 较大（12288），采用 TP2 或 TP4（非 TP8）减少通信开销。Dense FFN 的 intra-node NVLink 通信（all-gather/reduce-scatter）与 MoE 的 inter-node RDMA 通信（all-to-all）通过 GPUDirect RDMA 并发执行。
  - 实验比较：
    - LongCat-Flash vs DeepSeek-V3 在吞吐量和延迟上的对比（Table 6）：在各种配置下（bf16/fp8, 不同 context length），LongCat-Flash TGS（generation throughput per GPU）和 TPS/u（per-user generation speed）均显著高于 DeepSeek-V3。在 5000 avg context, bf16, 128 H800 GPUs 下达到 100.5 TPS/u。
    - 理论 TPOT（Table 7）：LongCat-Flash SBO 理论 TPOT 16ms, DeepSeek-V3 TBO 30ms, Qwen3-235B TBO 26.2ms。理论 cost: LongCat-Flash \$0.09/1M output tokens, DeepSeek-V3 \$0.17, Qwen3-235B \$0.15。
    - 实测 TPOT 约 26ms（batch size 96），达到理论值的 61.5%。
    - 成本：\$0.70 per million output tokens（基于 H800 \$2/hour）。

- 硬件平台是什么，配置是什么。
  - **推理部署集群**：NVIDIA H800-80GB GPUs，NVLink intra-node + RDMA inter-node（GPUDirect RDMA），200Gb/s per accelerator RDMA。典型部署单元：2 nodes × 16 GPUs（Prefill + Decode 各一 node），按需扩展到 128 GPUs。Wide EP deployment 支持上千 GPUs 扩展以降低解码延迟。
  - **通信库**：DeepEP（修改版支持 zero-computation experts），EPLB（修改版）。NVLink Sharp 硬件加速 broadcast (multimem.st) 和 in-switch reduction (multimem.ld_reduce)。

- 开源Serving框架是什么。修改了什么。
  - 论文未明确说明基于哪个开源 Serving 框架（如 vLLM、SGLang 等），以自研推理系统实现。
  - 核心修改/实现：
    1. **SBO Pipeline Scheduler**：四阶段 module-level overlap 调度器，实现单个 batch 内的 computation-communication overlap。非 DeepSeek-V3 的 TBO（需要两个 batch）。
    2. **PD-Disaggregation with Layer-wise KV Transmission**：层级别传输 KV cache（而非等待全部 KV cache 完成后再传输），显著降低高 QPS 下的 TTFT。
    3. **MTP Speculative Decoding Engine**：dense MTP head 作为 draft model + C2T rejection filter + TVD CUDA Graph fusion。
    4. **Multi-Step Overlapped Scheduler**：CPU 端调度器预分配多步 KV cache slots 并批量 launch kernel，隐藏 CPU-GPU 同步延迟。
    5. **DeepEP/EPLB 修改**：支持 zero-computation experts 的路由和负载均衡，zero-comp expert 输出跳过通信直接返回。
    6. **Custom Communication Kernels**：基于 NVLink Sharp PTX 的 all-gather/reduce-scatter kernel，直接使用 multimem 指令，比 NCCL 和 MSCCL++ 更快（4KB-96MB message size 范围内），仅需 4 thread blocks。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：模型权重和 GitHub 仓库开源（https://github.com/meituan-longcat）。推理系统层面（SBO scheduler, multi-step scheduler, custom kernels）论文未明确说明是否全部开源，但引用和使用了 DeepEP、FlashMLA、DeepGEMM 等开源组件并做了修改。
  - **推理全过程（SBO 单 token 解码）**：
    ```
    [请求输入]
    Prompt tokens → Router 计算 → Token-to-Expert Dispatch

    [SBO Pipeline 四阶段 - 以单个 token 为例]
    Stage 1: MLA_0 (单独执行)
      - Input 过 MLA_0 → 产生 q,k,v cache entry + MoE input
    Stage 2 (并行执行):
      - Dense FFN(chunk_a)
      - Attn_0: QKV Projection(chunk_a)
      - All-to-All Dispatch(chunk_b) → 通过 DeepEP RDMA 跨节点发送 token 到 expert 所在 GPU
    Stage 3: MoE GEMM(chunk_b)
      - SwapAB MoE GEMM kernel → 各 GPU 计算分配的 expert
    Stage 4 (并行执行):
      - Attn_1: Core Attention(chunk_a) + Output Projection(chunk_a)
      - Dense FFN(chunk_b)
      - All-to-All Combine(chunk_b) → 通过 DeepEP RDMA 回收 expert 计算结果

    [Speculative Decoding 增强]
    Target model forward → MTP Draft forward → C2T classifier filter →
    Target Verification (TVD fused CUDA Graph)

    [Multi-Step Overlapped Scheduler]
    CPU 预分配 4 步 KV cache slots → 批量 launch 4 个 SBO pipeline 的 CUDA kernels →
    GPU 连续执行不被 CPU 打断
    ```

    **作用**：SBO 将单 batch 内 All-to-All 通信（约 708us dispatch+combine 总计）大部分隐藏于 Dense FFN + Attention 计算（约 264us）中，non-overlapping communication 从 25.3% 降至 8.4%。MTP 以接受率 90% 将有效生成速度提升约 1.8x。Multi-step scheduler 消除 CPU launch 瓶颈。最终实现 100+ TPS/user，\$0.70/1M output tokens。

## LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

- 属于Serving调度的实现是什么？实验比较什么？
  - LYNX 在 vLLM 上实现了一个批处理级别的动态专家选择系统，核心 Serving 层面的实现包括：
    1. **Phase-aware Optimizer（相位感知优化器）**：集成在 vLLM 的 batch scheduler 内，判断当前迭代是否处于 memory-bound 的 decode 阶段。对于 co-located 部署（prefill/decode 同机但不同时混合），识别 pure-decode batches 为 memory-bound；对于 disaggregated 部署（prefill/decode 分离节点），直接标记 decode 节点为 memory-bound；对于 chunked prefill，标记仅含 decode tokens 的 batch 为 memory-bound。只有 memory-bound 迭代才触发 LYNX 的专家重映射，compute-bound 迭代（如 prefill）直接绕过。
    2. **Batch-Aware Expert Remapping Pipeline**：在每层 MoE router 输出后插入三个 fused kernel——confidence analyzer（对每 token 的 router logits 做 AffinityBinning 离散化）、adaptive expert scorer（batch 级别加权打分，选出最小关键专家集）、expert remapper（将低置信度 token 重映射到保留的专家集上）。最终减少每个 batch 激活的专家总数，降低从 HBM 加载专家权重的内存带宽压力。
    3. **Continuous Batching 兼容**：LYNX 完全在每次 forward pass 的 runtime 内执行，不依赖 workload 的先验知识，自适应 continuous batching 导致的 batch 组合每次迭代变化。
  - 实验比较：
    - Baseline：vLLM v0.10.1 默认推理（v1 scheduler，所有默认优化开启）
    - LYNX vs Baseline 在 TPOT（time-per-output-token）上的对比
    - 两类服务场景：co-located prefill/decode 和 disaggregated prefill/decode
    - 下游准确率：GSM8K, HumanEval, MBPP, MATH, ChartQA, MMMU, AIME, GPQA
    - 真实 trace：ShareGPT（对话）和 Mooncake（工具代理）
    - SLO-aware throughput：20ms/25ms/30ms P99 TPOT 约束下的系统吞吐量
    - 与 offloading（Fiddler）和量化（INT4 GPTQ/AWQ）的互补性

- 硬件平台是什么，配置是什么。
  - **主评测平台**：NVIDIA H200 GPU (141 GB HBM)，SXM NVLink 互联
  - **CPU**：2x AMD EPYC 9554 64-Core (128 cores total)，1.5 TB DRAM
  - **OS**：Ubuntu 22.04.4 LTS，NVIDIA driver 560.35.05，CUDA 12.6
  - **Offloading 实验**：单卡 NVIDIA A100 GPU (94 GB)，19 GB offload 到 CPU
  - **并行策略**：Mixtral-8x7B/Qwen2-57B/Llama-4-Scout 用 TP=2，Qwen3-30B 单卡，DeepSeek-Coder/Llama-Maverick/Qwen3-235B 用 TP=4
  - EP 实验：TP=2,EP=2 和 TP=4,EP=4

- 开源Serving框架是什么。修改了什么。
  - **框架**：vLLM v0.10.1 (v1 scheduler)
  - **LYNX 自身开源**：论文未提供 GitHub 链接或开源仓库。通过 CLI flag 启用。
  - 核心修改：
    1. **Batch Scheduler 集成 Phase-aware Optimizer**：在 vLLM scheduler 中增加 memory-bound 判断逻辑，对三种服务策略分别识别 decode-only iterations。
    2. **Router 输出拦截**：在每层 MoE router top-k 后插入 4 个 fused kernel（confidence analyzer → adaptive expert scorer → expert remapper），在 expert computation kernel 启动前完成专家集缩减和 token 重映射。
    3. **Expert Kernel Launch 参数调整**：以缩减后的 active expert set 作为 dispatch 参数启动专家计算。
    4. **CUDA Graph 兼容**：4 个 fused kernel 保持静态控制流，支持 vLLM CUDAGraph capture。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：论文未提供开源代码链接。实现基于 vLLM v0.10.1，通过 CLI flag 启用。
  - **使用例子与全过程**（基于论文描述还原）：
    1. **[Batch Scheduler]** 接收请求 → continuous batching → Phase-aware Optimizer 判断 memory-bound → 若 decode-only batch 则设置 ENABLE_LYNX flag
    2. **[Model Forward - per layer]** Attention 计算 → MoE Router 产生 logits/top-k → LYNX Confidence Analyzer (Kernel 1)：拦截 router probability，对每 token 做 log-ratio AffinityBinning → Adaptive Expert Scorer (Kernel 2-3)：batch 级指数加权打分，确定最小关键专家集 → Expert Remapper (Kernel 4)：低置信度 token 重映射到 active expert set，保留高置信度 token 的 top-ranked expert，compaction + renormalize → Expert Computation Kernel：以缩减的 expert set 从 HBM 加载权重执行计算
    3. **[硬件执行 - H200]** CUDA Graph 已捕获静态执行图 → 4 次 fused kernel launch (替代 700+ PyTorch ops) → Expert weights 从 HBM 加载量减少 → Decode latency 降低
    **作用**：在不修改模型、不依赖校准数据的情况下，每次 decode iteration 动态缩减 batch 级活跃专家数，直接减少 HBM 数据搬运量，缓解 MoE decode 的 memory bandwidth 瓶颈。median TPOT 降低 1.09-1.30x，SLO 约束下系统吞吐量提升最多 2.1x。
