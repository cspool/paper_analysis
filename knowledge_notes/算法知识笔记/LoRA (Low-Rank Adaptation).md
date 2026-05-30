## LoRA (Low-Rank Adaptation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA (Low-Rank Adaptation) 由 Hu et al. (2021) 提出，是一种参数高效微调（PEFT）方法。核心思想：对于预训练权重矩阵 W ∈ R^{d1×d2}，不直接微调 W，而是在旁路添加一对低秩分解矩阵 W^A ∈ R^{d1×r} 和 W^B ∈ R^{r×d2}（r << min(d1,d2)），前向计算变为 x' = xW + x·W^A·W^B + b。训练时冻结 W，仅更新 W^A（Kaiming 初始化）和 W^B（零初始化），确保训练起始 ΔW = 0。推理时可将 ΔW = W^A·W^B 与 W 合并（merge），无额外推理开销。典型 r 值：8/16/32/64。在 LLaMA-2 7B 的 Transformer 层中，有 7 个线性模块（Q/K/V/O/G/U/D），每个均可附加 LoRA。LoRA 可调参数通常仅为全模型参数的 <1%。

从算法pipeline角度拆解术语：
LoRA 在 LLaMA-2 Transformer 层中的前向计算：
```
输入: x ∈ R^{batch × seq × d1}
冻结权重: W_m ∈ R^{d1 × d2}  (m ∈ {Q,K,V,O,G,U,D})
LoRA矩阵: W_m^A ∈ R^{d1×r}, W_m^B ∈ R^{r×d2} (r=32)

# 标准前向 + LoRA 修正
output = x @ W_m + x @ W_m^A @ W_m^B + b_m

# 训练: W_m 冻结, W_m^A, W_m^B 可训练
# 参数量: r×(d1+d2), 对 LLaMA-2 7B d=4096: 32×8192=262K/module
# 7模块×32层 ≈ 80M (~1% of 7B)
```
推理场景：(a) 合并模式：W' = W + W^A·W^B，零额外开销；(b) 非合并模式（multi-tenant）：每次 forward 额外计算 7 个 LoRA 模块。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- HuggingFace PEFT: github.com/huggingface/peft, 通过 LoraConfig(r=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]) 配置。
- 变体：AdaLoRA（自适应 rank）、DoRA（magnitude+direction 分解）、QLoRA（4-bit 量化）、MOELoRA（sub-rank MoE experts）、MiLoRA（per-module experts + prompt-aware routing）。
- Multi-tenant: 多个任务各自有独立 LoRA weights，共享 frozen backbone。MiLoRA 在此场景下通过 prompt-aware routing 减少生成延迟。MOLE 进一步提出：多个预训练 LoRA 可通过逐层 gating 组合为统一模型，不同层对不同 LoRA 赋不同权重（Hierarchical Weight Control），保持各 LoRA 的个体特征。

涉及论文标题：
- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts
- MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning
- Mixture of LoRA Experts
- MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts

---
