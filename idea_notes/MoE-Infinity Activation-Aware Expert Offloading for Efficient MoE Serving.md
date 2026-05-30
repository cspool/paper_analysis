## MoE-Infinity Activation-Aware Expert Offloading for Efficient MoE Serving

- baseline方法是什么？
  Baseline 是现有 MoE 推理 offloading 系统中的专家缓存策略，核心分为三类：
  1. **基于依赖的预取**（DeepSpeed-Inference, HuggingFace TGI）：按计算图执行顺序预测下一层的 expert，将所有可能的 expert 都预取到 GPU，不考虑激活稀疏性，导致 PCIe 链路上大量无效数据传输，GPU 频繁因等 expert 而空闲（GPU idle time 513ms）。
  2. **基于计数的缓存**（BrainStorm, DeepUM）：用全局频率计数器追踪每个 expert 的历史使用次数，假设高频 expert 未来也会被使用。但在跨请求场景下 expert 使用趋于均匀分布，计数方法无法区分请求内的偏斜重用模式，BrainStorm 甚至比按需取 expert 的 vLLM 更差（934ms vs 485ms TPOT）。
  3. **LRU/LFU 局部性缓存**（vLLM, Llama.cpp/Ollama, Mixtral-Offloading）：按最近/最不频繁使用淘汰，仅考虑单个 expert 的访问局部性，不感知同一请求内 expert 间的协同激活关系（grouped activation），导致缓存命中率低。

  全栈执行例子（以 DeepSpeed-Inference 处理 "What is AI?" prompt，DeepSeek-V2-Lite 64×2.4B，A5000 GPU 为例）：
  - **算法层**：Router top-k gating 为每个 token 选择 top-6 experts → dispatch token 到对应 expert FFN
  - **系统框架层**：DeepSpeed-Inference 按计算图依赖顺序预取——处理 layer i 时预取 layer i+1 的所有 64 个 expert，不区分哪些会实际激活。每层需传输大量无用参数 → PCIe 4.0 32GB/s 下仅传输就需数百毫秒 → GPU 大量时间空闲等待
  - **编译框架层**：论文未明确说明。DeepSpeed 使用 PyTorch 原生执行。
  - **kernel调度层**：论文未明确说明。DeepSpeed 使用 cuBLAS GEMM 执行 expert FFN，数据传输用 cudaMemcpy。
  - **硬件架构层**：NVIDIA RTX A5000 (24GB)，CPU host memory ↔ GPU 通过 PCIe 4.0 ×16 (32GB/s)，GPU 计算单元在等待 DMA 传输期间空闲。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoE-Infinity 提出了 **Sparsity-Aware Expert Cache**，核心洞察是：batch_size=1 时，同一请求内 expert 激活具有高度稀疏性（<5% experts 被重复使用）和偏斜重用模式（skewed reuse），且相似请求共享相似的 expert 激活组（可被 K-means 聚类为 10-30 组）。但不同请求间的 expert 激活组转换不可预测（Markov 转移概率 <0.3）。因此，预测应通过**匹配请求级激活模式**而非学习跨请求转移规律来实现。

  具体设计：
  1. **EAMC（Expert Activation Matrix Collection）+ 余弦距离匹配**：追踪每个请求的 request-level EAM（L×E 矩阵），在 EAMC 中保存历史 rEAM。新请求的当前 iEAM 与 EAMC 中历史 rEAM 做余弦距离匹配，找到最相似的历史激活模式。**解决缺陷**：相比全局计数（不区分请求）和依赖预取（不感知稀疏性），EAMC 在请求级别捕获专家协同激活模式（S: sparsity 和 G: grouped activation 均满足）。
  2. **PredictEAM + Layer Proximity Decay**：对匹配到的历史 rEAM 进行聚合、行归一化，施加层邻近衰减（1-(i-l)/L），生成 pEAM。**解决缺陷**：考虑了重用预测（R: reuse 属性），并利用 MoE 逐层执行特性——越远的层预测置信度越低，避免过早预取不准确的 expert。
  3. **Probability-aware Cache Eviction**：淘汰 priority score 最小的 expert，score = n_token / ((pEAM + ε) × (1 - layer_idx/L))。综合考虑三个因素：expert 历史激活频率（n_token）、pEAM 中的预测概率、层位置（浅层优先保留，因浅层预取预测置信度低）。**解决缺陷**：LRU 不看未来、计数方法在均匀分布下失效，而 EAMC 匹配利用请求级偏斜模式准确识别哪些 expert 应被淘汰。
  4. **Prefetching 与计算重叠**：根据 pEAM 预取下一层高概率 expert，与当前层 GPU 计算重叠。每个 GPU 独立 I/O 线程使用 pinned memory + DMA 传输。**解决缺陷**：依赖预取全部 expert 导致 PCIe 拥堵阻塞按需取 expert（DeepSpeed），而 MoE-Infinity 只预取少量高概率 expert，保证按需取 expert 的 PCIe 带宽不被占用。

  全栈执行例子（MoE-Infinity 处理同一 "What is AI?" prompt，DeepSeek-V2-Lite 64×2.4B，A5000 GPU）：
  - **算法层**：Router top-k gating 同 baseline（选择 top-6 experts/token），不修改模型结构和路由算法
  - **系统框架层**：Prefill 阶段累积 rEAM → EAMC 余弦匹配（CPU 执行，21μs/query @1K EAMs）。Decode 每次迭代：Layer i Router dispatch → Cache lookup → Hit 直接用 / Miss FetchOnDemand；同时 PredictEAM → 仅预取 layer i+1 中 pEAM 概率 top-k expert（非全部 64 个）；Cache 满时按 probability-aware priority 淘汰。GPU idle time 从 513ms 降至 51ms（3.1–16.7× 加速）
  - **编译框架层**：论文未明确说明。基于 PyTorch 推理运行时，无 graph capture 或 custom compilation pass。
  - **kernel调度层**：每个 GPU 独立 I/O 线程用 pinned memory + DMA 做 CPU→GPU 传输，PCIe 4.0 单线程即可打满 32GB/s。集成 FlashAttention 优化 attention kernel。MoE FFN 用标准 GEMM 执行。
  - **硬件架构层**：同 baseline 硬件（A5000 + PCIe 4.0）。关键优化不在硬件层面，而在减少无效 PCIe 传输量——通过 EAMC 匹配将传输量从"全 expert 预取"降至"仅高概率 expert 预取"，PCIe 带宽用于真正需要的 expert 传输，GPU 闲置率从 baseline 的 51-80% 大幅降低。
