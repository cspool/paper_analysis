## Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Nexus 提出一种增强型 MoE 架构，核心创新是**基于域嵌入的自适应路由器**：
    1. **Router 设计**：用 2 层 MLP（SwiGLU 激活）作为投影层 P_r，将预计算的域嵌入 d_i ∈ R^m（由 Cohere Embed v3 对每域名数据集编码后取平均得到）投影为专家嵌入 e_i ∈ R^h。路由概率通过 s_i = softmax(x · e_i) 计算，即输入 token 与各专家嵌入的点积相似度。这与超网络（hypernetwork）密切相关——投影层在运行时为给定输入生成路由参数。
    2. **Upcycling 阶段**：分别在不同域（ArXiv, Books, C4, StackExchange, Wikipedia）上独立训练 dense expert 模型，之后将各 expert 的 FFN 层沿新维度拼接为 MoE 层 FFN_{moe} = FFN_s + [FFN_e1, ..., FFN_en]。Seed 模型的原始 FFN 作为共享专家（始终激活），非 FFN 参数（attention 等）通过简单权重平均合并：φ_{moe} = Σ φ_i / n。
    3. **扩展阶段**：新域到来时，计算新域嵌入 d_new，通过已训练的投影层得到 e_new = P_r(d_new)，新 expert FFN 直接拼接到已有 expert 数组后，非 FFN 参数用加权平均 φ_f = (1-λ)·φ_moe + λ·φ_new（λ=1/(n+1)），然后用 1B token 做轻量微调。
  - 实验比较：
    1. **初始 Upcycling 性能**：对比 Dense Merging（BTM 风格等权平均）和 upcycled MoE with Linear Router（标准线性路由器的 upcycled MoE），在 470M 和 2.8B 两个 seed model 规模上评估 Knowledge / Science / Reasoning / MMLU 四类共 15 个下游任务。
    2. **扩展新 Expert（Code）**：在 2.8B seed model 的 upcycled MoE 上新增 Code expert（Starcoder 数据训练），比较 200M / 500M / 1B finetuning tokens 下 Nexus vs MoE (Linear Router) 的 Code 性能和通用任务性能。
    3. **Ablations**：load balancing loss factor 变化（0.05 vs 0.0005）、训练数据采样策略（按域大小比例 vs 均匀采样）、域嵌入投影前后的 cosine similarity 可视化。
    4. **Expert 专业化度量**：按域计算各 expert 的平均路由概率（routing frequency），验证 domain specialization 是否在 upcycling 后保持。

- 硬件平台是什么，配置是什么。
  - 训练平台：论文未明确说明具体 GPU 型号/数量
  - 精度：论文未明确说明（推测为 BF16 或 FP32）
  - 优化器：AdamW（论文提到使用 AdamW — 参考 Nemotron-4 的 recipe，但未单独列出 Nexus 使用的优化器；Section 4.1 提到 cosine decay schedule 但未指定优化器名）
  - 学习率：linear warmup 10% steps → max lr 1e-3 → cosine decay → 3e-4（dense expert 训练阶段）; cos decay to 3e-5（upcycling 最后 1B tokens）
  - 分布式框架：论文未明确说明

- 模型是什么。数据集和bench分别是什么。
  - 模型：Decoder-only autoregressive Transformer，470M 和 2.8B 参数两种规模的 seed model，使用 parallel attention layers、SwiGLU activation、no biases、BPE tokenizer（vocab 256k）。
    - 470M MoE：1 shared expert + 6 routed experts → total 1.3B params, 605M active（top-2 routing）
    - 2.8B MoE：1 shared expert + 4 routed experts → total 9.1B params, 4.3B active（top-2 routing）
  - 训练数据集：SlimPajama（627B token English corpus），包含 ArXiv, Books, C4, StackExchange, Wikipedia 子集，排除了 Github/StackExchange 用于后续 Code domain ablation
  - 扩展数据集：StarCoder code documents（Code expert 训练）
  - Dense expert 训练 token 量：470M scale 用 25B tokens/expert，2.8B scale 用 40B tokens/expert
  - MoE 训练 token 量：25B (470M) / 40B (2.8B)，最后 1B tokens 上做 upweight 原始预训练数据
  - Seed model 在 full SlimPajama 750B tokens 上训练
  - Benchmark：
    - Knowledge: OpenBookQA, Natural Questions, TriviaQA, QUAC (0-shot), SQuAD (4-shot)
    - Science: ARC-Easy, ARC-Challenge, SciQ (0-shot)
    - Reasoning: CommonSenseQA, SIQA, PIQA, WinoGrande, HellaSwag (0-shot)
    - General: MMLU (5-shot)
    - Code: MBPP, LBPP, HumanEval-Pack (Cpp, JS, Java, Go, Python, Rust) (0-shot)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：**未开源**。截至搜索日期（2026/05），无公开 GitHub 仓库或官方代码实现。论文在 Papers with Code 上显示 "No code implementations yet"。
  - 算法 pipeline 核心执行流程（基于论文 Section 3 及 Figure 2 伪代码）：

  **Phase 1: Dense Expert 训练**
  1. 从 seed model（已在 750B tokens 上预训练）初始化
  2. 对每个域（ArXiv, Books, C4, StackExchange, Wikipedia）独立训练 dense expert：用各自域的数据做 continue training（25B/40B tokens）
  3. 对每个域的数据用 Cohere Embed v3 编码所有样本，取平均得到域嵌入 d_i ∈ R^m

  **Phase 2: Upcycling（合并为 MoE）**
  1. MoE 层构造：
     - FFN_{moe} = concat([FFN_e1, ..., FFN_en]) along new dimension
     - Shared expert = FFN_seed（始终激活）
     - 非 FFN 参数（attention, norms, embedding）: φ_{moe} = mean(φ_1, ..., φ_n)
  2. Router 训练（每个 Transformer block 一个独立 router）：
     ```
     # 域到专家嵌入投影 (2-layer MLP, SwiGLU)
     # d_i: domain embedding [m], W1: [2h x m], W2: [h x h]
     expert_embeddings[i] = W2 @ SwiGLU(W1 @ d_i)  # [h]
     # 按 token 路由
     router_probs = softmax(inputs @ expert_embeddings)  # [batch, seq, n_experts]
     # Top-1 选路由专家 (+ shared expert 始终激活 = top-2)
     index, gate = topk(router_probs, k=1)
     # 输出
     out = shared_expert_ffn(inputs) + gate * routed_expert_ffns[index](inputs)
     ```
  3. 继续训练：用所有域 + 原始预训练数据的 mix 训练 25B/40B tokens，最后 1B tokens 上做 upweight 原始预训练数据 + cos decay lr to 3e-5

  **Phase 3: Extension（添加新 expert）**
  1. 用 StarCoder 数据训练新的 dense Code expert（8B tokens）
  2. 计算 Code domain embedding d_code，通过投影层得到 e_code = P_r(d_code)
  3. 追加 FFN_code 到 MoE 层，加权平均非 FFN 参数
  4. 轻量微调（up to 1B tokens）：data mix = 50% 旧域+预训练数据 + 50% Code 数据

  **张量流动**（以 upcycling 阶段一个 Transformer block 的 forward 为例）：
  - Input: x ∈ R^{s×h}（s 序列长度, h 隐藏维度）
  - Router 计算：预存 domain_embeddings ∈ R^{m×n} → 投影层 P_r (2-layer SwiGLU MLP) → expert_embeddings ∈ R^{h×n} → router_probs = softmax(x @ expert_embeddings) ∈ R^{s×n} → Top-1 gate → selected expert index ∈ Z^s
  - Shared expert: always → y_shared = FFN_seed(x) ∈ R^{s×h}
  - Routed expert: 按 index gather → y_routed = FFN_{index[i]}(x[i]) ∈ R^{s×h}
  - Output: y = y_shared + gate * y_routed ∈ R^{s×h}

  - 分布式 Muon (Algorithm 1) 张量流动：全梯度 G(fp32) → DP reduce-scatter 分片 → 本地动量更新(fp32) → DP gather 恢复全梯度矩阵(bf16) → Newton-Schulz N=5 迭代 → 取本地参数分片 → apply_update(p, u) with weight decay → DP all-gather 同步(fp32)。非矩阵参数（RMSNorm、LM head、embedding）仍用 AdamW。
  - Lemma 1：对 shape [A,B] 满秩矩阵，Muon 理论更新 RMS = √(1/max(A,B))。因此 √(max(A,B)) 缩放抵消此效应。
