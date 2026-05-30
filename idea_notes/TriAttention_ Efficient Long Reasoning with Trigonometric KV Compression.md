## TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

- baseline方法是什么？
  Baseline 方法分为两类：(A) post-RoPE attention-based 方法（SnapKV, H2O, R-KV, LazyEviction）——使用最近 token 的 post-RoPE attention scores 估算 key 重要性，但 query 经 RoPE 旋转后朝向随位置变化，只有最近约 25 个 query 保持"当前"朝向，观察窗口极小，导致重要 key 未被检测到就被 evict；(B) post-RoPE norm-based 方法（VATP）——只使用 vector 范数而忽略方向信息，因为 post-RoPE 空间中方向与位置旋转纠缠，难以利用方向信号。两类方法的核心缺陷相同：都基于 post-RoPE 空间操作，受位置旋转限制。
  
  全栈执行例子（Qwen3-8B 使用 R-KV 进行 KV 压缩推理 on A100 80GB）：
  **算法pipeline**：R-KV 每 128 tokens 触发一次剪枝——收集最近 N 个 query 对所有 key 的 attention scores（post-RoPE QK^T），沿 query 维度聚合评分，结合 redundancy detection（hash similarity between adjacent tokens）标记冗余 token，保留 top-B 个非冗余 token。问题：(a) 最近 N 个 query 中大部分位置因 RoPE 旋转而方向过时，仅约 25 个有效——这对 retrieval head 特别致命，相关 token 可能沉寂数千步后才被需要；(b) 注意力分数在 25-query 窗口内缺乏统计稳健性，噪声主导选择。AIME25 上 R-KV 准确率仅 17.5%（Full Attention 40.8%）。
  **系统框架**：HuggingFace Transformers + FlashAttention-2，模型权重加载 Qwen3-8B，每次 decode step 计算 full attention（O(T)），每 128 步标记一次剪枝。
  **编译框架**：论文未明确说明。
  **kernel调度**：FlashAttention-2 fused kernel，tiled QK^T + online softmax，在 A100 80GB 上 batch decode。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TriAttention 回到 pre-RoPE 空间，利用发现的 Q/K 浓度现象（pre-RoPE Q/K 围绕固定中心高浓度聚集，R ≈ 0.98，跨位置稳定）来解决 post-RoPE 的不稳定性：
  
  1. **绕过位置旋转限制**：pre-RoPE 向量不受 RoPE 旋转影响，Q 中心跨所有位置稳定——不再受限于仅 25 个有效 query 的观察窗口。校准数据中一次计算的 Q/K 中心可泛化到任意推理长度。
  
  2. **三角函数级数评分替代注意力观测**：当 Q/K 聚集时，attention logit 退化为仅依赖 Q-K 距离 Δ 的三角函数级数 Σ_f ‖q̄_f‖·‖k̄_f‖·cos(ω_f·Δ+φ̄_f)。用 Q 中心替代未来 query，直接用该级数对 key 打分 (S_trig)——无需观察实际 attention scores，避免了小窗口噪声。
  
  3. **范数信号补充分离方向与规模**：pre-RoPE 空间的方向和范数是分离的（浓度度量 R 捕获方向聚集度，范数独立变化），因此 S_norm = Σ_f (1-R_f)·E[‖q_f‖]·‖k_f‖ 在浓度低的 head 中平滑补充范数信息——而 post-RoPE 中方向被旋转污染，难以利用。
  
  4. **自适应加权**：R_f 直接作为加权因子——R_f 高（浓度强，方向预测可靠）时 (1-R_f) 小，S_trig 主导；R_f 低时 Snorm 贡献更大。无需超参调节。
  
  AIME25 上 TriAttention 准确率 32.9%（R-KV 17.5%），几乎翻倍。
  
  全栈执行例子（TriAttention on Qwen3-8B，推理 on A100 80GB）：
  **算法pipeline**：(1) 离线校准：使用少量校准数据（50K tokens 即可，编码/聊天/HTML 均稳定）计算各 head 各频段的 E[q_f], E[k_f], E[‖q_f‖], R_f。(2) 推理阶段每 128 tokens：遍历 cache 中每个 key k，对其每个未来距离 δ∈{1,2,4,...,2^16} 计算 S_trig (k, Δ+δ) = Σ_f ‖E[q_f]‖·‖k_f‖·cos(ω_f(Δ+δ)+(arg(E[q_f])-arg(k_f)))，加上 S_norm = Σ_f (1-R_f)·E[‖q_f‖]·‖k_f‖，平均所有 δ 得最终评分。(3) GQA 场景：per-head z-score normalize 后 max 聚合。(4) 保留 top-B，裁剪 KV cache。关键：S_trig 不依赖任何实际 attention 计算——只用到离线预计算的 E[q_f] 和 cache 中已有的 k_f。
  **系统框架**：vLLM plugin（triattention/vllm/runtime/integration_monkeypatch.py），自动发现激活——通过 monkeypatch scheduler 和 worker 注入剪枝逻辑。也支持 SGLang 集成和 MLX (Apple Silicon) 部署。论文中使用 HuggingFace Transformers + FlashAttention-2 进行评估。
  **kernel调度**：FlashAttention-2 标准 fused attention kernel，TriAttention 不修改 kernel 层——剪枝操作在 attention 计算前进行，仅减少输入的 KV 数量，不改变 attention 本身的计算图。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。
