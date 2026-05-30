## Cost-Optimal Grouped-Query Attention for Long-Context LLMs

- baseline方法是什么？
  Baseline 是当前广泛采用的 Llama-3 GQA 配置方式：(1) 强制 nh × dh = d（如 d=1536, dh=64 → nh=32），head 数量由 hidden size 唯一确定，不可独立调整；(2) 固定 nkv=8（Llama-3 全系列统一），不随上下文长度或目标 loss 变化；(3) 模型大小 N 和 GQA 配置独立决定，不考虑推理上下文长度 T 对 time-variant cost（attention FLOPs + KV cache memory）的影响。

  缺陷：(a) nh × dh = d 是原 Transformer 论文的随意选择（Vaswani et al., 2017），无理论基础，导致 attention FLOPs（4TL dh nh）不可调——当 T 很大时 attention 占主导但无法减少；(b) nkv=8 在长上下文场景（T=128K）下导致 KV cache 内存（2TL dh nkv）巨大——128K 时 ~90% 推理内存被 KV cache 占用、仅 ~10% 用于模型参数（Figure 8）；(c) 现有 scaling law（Hoffmann et al., Kaplan et al.）仅考虑训练 FLOPs 不考虑推理成本和上下文长度。实验表明 Llama-3 GQA 在 T=128K 时 "highly suboptimal"——用 cost-optimal 配置可减少 >50% memory 和 FLOPs 且 loss 相等。

  全栈执行例子（Baseline / Llama-3 GQA, T=128K, 1.2B model）：
  - 算法pipeline：nh=32, nkv=8, dh=64 → 32 query heads 共享 8 KV heads（每组 4 query heads 共享 1 KV）。每层 attention FLOPs = 4TL dh nh = 4 × 128K × 36 × 64 × 32 ≈ 37.7G FLOPs（仅 attention softmax 部分）；KV cache = 2TL dh nkv = 2 × 128K × 36 × 64 × 8 = 4.7B floats ≈ 9.4GB（BF16）。模型参数 N=1.2B → 参数内存 ~2.4GB。总内存 ~11.8GB，KV cache 占 ~80%。
  - 系统框架：标准 HuggingFace Transformers 或 vLLM 推理；KV cache 为 autoregressive decode 的 key-value 缓存，每步追加新 token 的 KV。
  - 编译框架：论文未明确说明。
  - kernel调度：使用标准 FlashAttention-2 加速 attention。
  - 硬件架构：NVIDIA A800 GPU（80GB）；长上下文下 memory bandwidth bound——KV cache 读取是 bottleneck。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出两个关键改变 + 三步搜索过程来寻找 cost-optimal GQA 配置：

  **Change 1: 解耦 nh 与 d** → 解决 Baseline 缺陷(a)。解除 nh × dh = d 约束，使 nh 成为独立超参数自由控制 time-variant FLOPs。这允许在长上下文时使用更少的 query head（如 nh=8 替代 nh=32）大幅减少 attention FLOPs，同时通过增加模型大小 N 来补偿 loss（增加 time-invariant FLOPs 但远小于 attention 节省）。

  **Change 2: 联合优化 N 与 GQA 配置** → 解决 Baseline 缺陷(b)(c)。将推理成本分解为 time-invariant（N：模型参数 FLOPs 2N + 内存 N）和 time-variant（T: attention FLOPs 4TL dh nh + KV cache 2TL dh nkv）。通过同时调整 N, nh, nkv 优化推理资源分配——长上下文下 time-variant cost 主导，应减少 head 数、增大模型 size 以更高效利用硬件。

  **三步搜索过程** → 系统性解决 "给定 target loss L* 和 context length T，什么 GQA 配置最省推理成本" 这一问题：
  - Step 1 (Candidate Selection): 定义 H_cand = {nh=1,2,4,...,32} × {nkv=1,2,4,...,32, nkv≤nh} = 21 个候选
  - Step 2 (Scaling Curve Fitting): 对每个 H 训练 3M→1.2B 模型，拟合 L(N;H) = (a/N)^b + E（R²>0.999）
  - Step 3 (Cost Minimization): 对每个 H 求 N*(H) 满足 L*，计算硬件感知成本 Z = 0.9·M^0.5 + 0.1·C^(1/3)，选 Z 最小者

  关键理论洞察：上下文长度 T 对 loss 的影响与 N 和 H 相独立（Section 5.7 verified），因此可以用短上下文（T=8K）的 scaling curve 外推至长上下文，大幅节省算力。

  全栈执行例子（Cost-Optimal GQA, T=128K, L*=2.615）：
  - 算法pipeline：搜索得 H*=(nh=8, nkv=1), N*=1.8B。8 query heads 全部共享 1 KV head（退化为 MQA）。每层 attention FLOPs = 4TL dh nh = 4 × 128K × 36 × 64 × 8 ≈ 9.4G（节省 75% vs baseline 37.7G）；KV cache = 2TL dh nkv = 2 × 128K × 36 × 64 × 1 = 589M floats ≈ 1.18GB（节省 87.5% vs baseline 9.4GB）。模型参数 N=1.8B → ~3.6GB。总内存 ~4.8GB，节省 ~60% vs baseline。尽管模型大了 50%，但因 KV cache 从 8 KV heads 降为 1，总推理资源大幅减少。Downstream accuracy 几乎不变（common-sense 45.5% vs 45.7%，NIAH 略优）。
  - 系统框架：同 baseline——标准推理 pipeline，仅改变模型配置（L,d,nh,nkv）；与 FlashAttention-2 和现有推理框架完全兼容。
  - 编译框架：论文未明确说明。
  - kernel调度：标准 FlashAttention-2——因 head 数减少，单个 attention 计算的 tiling 效率反而可能提升（更大 tile、更少 kernel launch overhead）。
  - 硬件架构：NVIDIA A800 GPU；减少 KV cache → 更少 HBM 读写 → 从 memory bandwidth bound 转向 compute bound → 更高效利用 GPU 算力。

  对应解决的完整映射：
  - Baseline 缺陷(a) nh×dh=d 限制 → Change 1 解除 → nh 从 32→8（128K 时），attention FLOPs 降 75%
  - Baseline 缺陷(b) nkv=8 固定 → Change 2 + Step 3 → nkv 从 8→1（128K 时），KV cache 内存降 87.5%
  - Baseline 缺陷(c) scaling law 不考虑推理 → 引入 M_infer(T) + C_infer(T) + Z 统一成本函数，Step 2 scaling law 拟合 → 精确量化 N vs. H 的 tradeoff

  核心发现：
  - 长上下文下应使用 **更少的 head + 更大的模型**（更多 time-invariant 资源），因 time-variant cost 主导
  - 常用 Llama-3 GQA (d/dh, 8) 仅对特定 (L*, T) 组合最优，大多数情况下 suboptimal
  - nh 比 nkv 对 loss 更重要（相同参数增量下 nh 增加带来更大的 loss 降低），两者均呈 diminishing returns
  - loss 与 nh 呈 power-plus-constant 关系：L(nh) = a·nh^b + c，与 model size 和 context length 独立
  - 对齐 training FLOPs 时，用更少 head 可获更多训练数据，优势更大（88%/83% memory/FLOPs 节省）
  - 成本函数默认权重 λ=0.9（偏重 memory）可调整以适配不同部署约束
