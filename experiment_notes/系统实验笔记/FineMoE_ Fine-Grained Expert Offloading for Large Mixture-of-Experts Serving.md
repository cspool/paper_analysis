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
