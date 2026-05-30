## CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios

- baseline方法是什么？
  Baseline 是标准的 LLM 长上下文推理流程：每个 token 的 Key 和 Value 以完整维度 hout（通常等于 hidden_size/num_heads）存储在 KV Cache 中，KV Cache 大小随序列长度 n 线性增长（2 × n_layers × n × hout × dtype_size）。在 200K token 场景下（LLaMA-2-7B），KV Cache 约占用 100GB。现有训练无关压缩方法（如 StreamingLLM 的 token pruning + 保留 attention sink；H2O 的 Heavy-Hitter Oracle token pruning）面临压缩率上限，高压缩率时因丢弃关键 token 导致检索任务（如 LongEval）性能崩溃。ASVD（训练无关 channel shrinking via SVD）在高压缩率（80%）时导致模型丧失语言建模能力，输出不可解析的乱码 token。训练依赖方法（如 MLA/DeepSeek-V2）虽压缩率高但需从零重训整个模型，无法适配已有预训练模型。

  全栈执行例子（Baseline / 标准 LLaMA-2-7B 长上下文推理，200K tokens）：
  - 算法pipeline：每个 Transformer 层的 W^K, W^V 将输入 X 映射为 K ∈ R^{n×hout}, V ∈ R^{n×hout}，全部存入 KV Cache，不做任何压缩。Attention: softmax(QK^T/√d) × V。
  - 系统框架：标准 HuggingFace Transformers 推理流程，KV Cache 为 autoregressive 解码的 cache 机制，每步 decode 将新 KV append 到 cache。
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明
  - 硬件架构：记忆瓶颈——200K tokens 时 KV Cache ~100GB，远超 A100-80G 或 RTX 4090 24GB 显存

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文 CSKV 提出 training-efficient 的通道收缩 KV Cache 压缩方法，通过低秩分解 + 双分支缓存 + SVD-based 逐层微调，实现 80% 压缩且保持长上下文能力。

  **对应解决 Baseline 缺陷的设计**：

  1. **低秩分解解决通道维度冗余**：观察到 KV Cache 的奇异值呈长尾分布——移除最小的 50% 奇异值仅导致 MMLU 平均精度损失 <1%（0.458→0.449）。将 W^K ∈ R^{hin×hout} 分解为 A^K ∈ R^{hin×hcomp} 和 B^K ∈ R^{hcomp×hout}，仅存储压缩特征 hcomp 维（而非完整 hout 维），内存从 O(n×hout) 降至 O(n×hcomp)。

  2. **双分支 KV Cache 解决信息损失**：近期 token（窗口大小 m=32）保留完整精度，确保局部上下文预测质量不受影响；历史 token 从压缩特征通过 B^K 重建用于 attention。这避免了 token pruning 方法（如 StreamingLLM、H2O）完全丢弃 token 导致的检索信息丢失问题。

  3. **ASVD 初始化 + 逐层 MSE 重建微调解决训练代价问题**：ASVD（Activation-aware SVD）使用标定数据（256 样本）计算缩放矩阵 S，使低秩分解关注激活值大的维度。仅优化逐层重建损失（MSE(K, K_hat)+MSE(V, V_hat)）而非端到端语言建模损失，训练仅需 90 分钟/单 A100（vs 从零重训练的数天/数月）。随机初始化在此设置下完全无法收敛（Loss ~1e9），证明 SVD-based 初始化的必要性。

  4. **量化正交兼容**：通道压缩（channel shrinking）与量化（quantization）是正交维度，可与 KIVI 4-bit QAT 无缝结合，达到 95% 总压缩率（80% channel + 4-bit = 95% total），保持 >90% 长上下文能力。

  全栈执行例子（CSKV）：
  - 算法pipeline：(1) Prefilling：X → K_full = XW^K（attention 计算用），K_C = XA^K（存入 Compressed Cache），K_local = K_full[-m:, :]（保留 m 个完整 token）；(2) Decoding：新 token → 更新两个 cache → 从 Compressed Cache 用 B^K 重建历史 token 的 K_hat → concat([K_hat, K_local]) 用于 attention → 从 Full Cache 移除最旧 token 保持窗口 m；(3) 训练：ASVD 初始化 A^K/B^K/A^V/B^V → 逐层 MSE(K, XA^KB^K) + MSE(V, XA^VB^V) → AdamW 微调
  - 系统框架：即插即用集成到 HuggingFace 推理流程——仅修改 attention 层的 Key/Value 投影和 cache 管理，不改 LLM backbone 结构
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明
  - 硬件架构：训练在单 A100-80G 上 90 分钟完成；推理 KV Cache 从 ~100GB 降至 ~20GB（80% 压缩），可适配 RTX 4090 24GB

  **消融发现的关键 insights**：
  - 随机初始化低秩矩阵导致训练完全失败（Avg.Acc=0.00），ASVD 初始化为关键使能技术
  - 窗口大小 m 与性能正相关，但 m>32 后收益递减（m=32 Avg.Acc=0.92, m=4096 Avg.Acc=0.96），m=32 已足够保留局部信息
  - Key cache 比 Value cache 对压缩更不敏感——在固定 budget 下应给 Key 分配更高压缩率（K 87.5% + V 12.5%: Avg.Acc=0.97 vs K 12.5% + V 87.5%: Avg.Acc=0.80, 均在 50% 总压缩率下）
  - 与量化直接 PTQ 结合导致性能崩溃，需 QAT 才能保持性能（80% 通道压缩 + 4-bit QAT → Avg.Acc=0.90 vs PTQ → 0.00）
