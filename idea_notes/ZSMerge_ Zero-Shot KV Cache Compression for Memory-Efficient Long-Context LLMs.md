## ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

- baseline方法是什么？
  Baseline 是现有 KV cache 管理方法，分为三类：
  (1) **稀疏驱逐（Sparse Eviction）**：H2O 基于累积 attention score 阈值动态驱逐低贡献 token + 保留最近 token；StreamingLLM 固定保留 attention sinks（初始 token）+ 滑动窗口；SnapKV 在 prefilling 阶段做一次性 KV cache 剪枝。共同缺陷：**永久丢弃 token 导致不可逆信息损失和 attention distribution drift**——被驱逐 token 的 KV 信息完全丢失，后续解码步无法恢复，错误在长序列中累积传播（Figure 4 中纯驱逐在 5% cache 下相对误差远高于残差合并）。
  (2) **Token 合并（Token Merging）**：LESS 使用辅助网络学习压缩 token，通过 recurrent merging 合并相似 token 并做 attention rectification。缺陷：**需要额外网络 + 训练数据（C4）微调**，跨模型架构和新任务泛化差（Falcon-7B 上 5% cache 下 ROUGE-1 从 27.06 暴跌至 7.75），训练开销大。
  (3) **上下文无关优化**：GQA/MQA（减少 KV head）、KV cache 量化（KV cache 低精度存储）、低秩近似。缺陷：模型架构修改或精度损失。

  全栈执行例子（LLaMA2-7B + H2O on NVIDIA A800-80GB）：
  **算法pipeline**：LLaMA2-7B 32 层 × 32 heads × 128 dim，在解码步 T 时 KV cache 为 K_T, V_T ∈ R^{T×d_head}。H2O 维护每个 token t 的累积 attention score Σa_t，在每个 decoding step 保留 top-k "heavy hitter" tokens + 最近 w 个 tokens，其余永久驱逐。被驱逐 token 的 key-value 信息完全删除，后续所有 query 无法访问。当 T 增长到 54K 时，即使只保留 5% tokens（~2700），剩余 tokens 的 attention distribution 也显著偏移——heavy hitters 筛选依赖历史 attention pattern，而该 pattern 本身因之前驱逐而被扭曲（compounding error）。
  **系统框架**：HuggingFace Transformers 加载 LLaMA2-7B，使用 PyTorch SDPA。H2O 在每个 attention layer 的前向传播中插入 KV cache 管理逻辑。
  **编译框架**：论文未明确说明。
  **kernel调度**：论文未明确说明。使用标准 PyTorch scaled_dot_product_attention。
  **硬件架构**：NVIDIA A800-80GB，VRAM 25GB 用于模型参数加载，KV cache 随 token 线性增长 ~1MB/token。FullKV 在 54K tokens 时 KV cache 消耗 54GB，总 VRAM ~79GB > 80GB 导致 OOM。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ZSMerge 提出"稀疏+残差"混合压缩，在不引入参数/训练的前提下解决驱逐和合并方法的缺陷：
  
  (1) **残差合并解决驱逐的信息损失**（核心创新）：Baseline 驱逐方法永久删除低贡献 token → ZSMerge 动态将被驱逐 token 合并入 Br 个残差 slot。每次驱逐 token (k_t, v_t)，通过 k_r·k_t 找最相似 residual slot，用增量均值聚合更新 slot（Eq. 7）。这等价于将多个原始 token 压缩编码为一个 slot，而非丢弃——解除了"驱逐即信息永久丢失"的根本约束。Figure 4 验证：残差合并比纯驱逐减少 37-89% 的 attention 输出误差。
  
  (2) **补偿注意力解决合并后表示偏差**（Theorem 1）：Token 合并产生 mismatch——k_r 是多个 token 的均值，与对应的 v 分布不匹配。ZSMerge 在 softmax logit 中加 α·log w_r 补偿项（Eq. 8），其中 w_r 为 slot r 合并的 token 数。log w_r 将"此 slot 代表多个 token"的信息注入 attention 评分。Theorem 1 证明 â_i ≥ a_i（∀ 未压缩 token i），即未压缩 token 的 attention 占比在压缩后不降低，防止压缩 token 因 log w 补偿而"过度放大"。
  
  (3) **三分区 budget 优于已有方法的二分法**：H2O 仅重排 heavy hitters + 最近 token，LESS 用合并网络。ZSMerge 的三分区（proximity + context + residual）将"保持局部上下文"、"保留全局关键信息"、"压缩冗余"三者解耦分配预算——这是纯驱逐或纯合并无法同时做到的。
  
  (4) **零样本无参数设计解决泛化问题**：LESS 的辅助合并网络需在 C4 上训练，泛化到 Falcon 架构时性能崩溃。ZSMerge 仅依赖 token 间 key 相似度（dot product）和累积 attention score（Eq. 5），无任何学习参数，可直接应用于 MHA/MQA/GQA 任意架构。

  全栈执行例子（ZSMerge on LLaMA2-7B, NVIDIA A800-80GB）：
  **算法pipeline**：LLaMA2-7B 相同 32 层，每层每个 head 独立运行 ZSMerge。解码步 T：(a) 更新 s_t = 0.98·s_t + a_t（每个 token 累积 attention 衰减和）；(b) 分配 B=B_p+B_c+B_r，B_p 保留最近 token（proximity），B_c 选 top-B_c 按 s 排序（context），剩余 token 按 Eq. 6-7 合并入 B_r 个 residual slot；(c) K_B = [K_p∥K_c∥K_r] 拼接后，用 Eq. 8 计算补偿注意力（每个 slot k_r 带 log w_r 偏置）；(d) softmax + V_B 加权输出。复杂度 O(T + B·d)，线性于 T。54K tokens 下仅需 18K cache budget（~3:1 压缩），VRAM 恒定 43GB（82% 减少），解码吞吐维持 9 tokens/sec（3× FullKV）。
  **系统框架**：HuggingFace Transformers，仅全局替换 `scaled_dot_product_attention` 函数。支持 LLaMA/Falcon/Mistral 系列。`change_mode` 方法支持运行中切换压缩模式。prefill 阶段通过 `window_size` 参数限制 s 初始化范围（类似 SnapKV）。
  **编译框架**：论文未明确说明。
  **kernel调度**：论文未明确说明。使用 PyTorch 标准 SDPA 或 Flash Attention v2（通过 KVCache-Factory）。无自定义 CUDA kernel。
  **硬件架构**：NVIDIA A800-80GB。实验使用单 GPU 推理。
  **芯片设计**：论文未明确说明。
