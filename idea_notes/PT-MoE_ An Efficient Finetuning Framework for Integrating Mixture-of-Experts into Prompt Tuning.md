## PT-MoE: An Efficient Finetuning Framework for Integrating Mixture-of-Experts into Prompt Tuning

- baseline方法是什么？
  - Standard Prompt Tuning (PT): 将可训练的 soft prompt 向量 P∈R^{T×H}（T=prompt length, H=hidden dim）直接 prepend 到输入 embedding 序列前，冻结 base model 所有参数，仅优化 P。Soft prompt 从 task-relevant 文本的 word embeddings 初始化。训练 loss 为语言模型的标准 NLL loss，只计算非 prompt 位置的 token。
  - 全栈执行例子（Baseline: PT, LLaMA-3.2-1B-Instruct, 1 node 4×A100 80GB, MRQA QA 任务, 81K trainable params）：
    (a) 模型推理算法层——输入文本经 tokenizer → base model embedding 层 → soft prompt P (40×2048) prepend 到 embedding 前 → frozen LLaMA decoder 逐层 self-attention + FFN 前向 → 输出 token logits → 仅计算非 prompt 位置的 CE loss → 反向传播仅更新 P（81K/1.2B=0.007% 参数）。PT 无 MoE、无分解、无 routing，所有输入共享同一 soft prompt。
    (b) 系统框架层——HuggingFace Transformers Trainer + DeepSpeed ZeRO-3。AdamW optimizer, lr=2e-5, constant_with_warmup schedule, warmup 500 steps, per_device_batch_size=32, gradient_accumulation=2。Inference 用 greedy decoding (do_sample=False, num_beams=1, temperature=1.0)。
    (c) 编译框架层——论文未明确说明。
    (d) Kernel 调度层——论文未明确说明。标准 PyTorch embedding concat + transformer forward，无自定义 kernel。
    (e) 硬件架构层——4×A100 80GB，基于 PyTorch 2.3.1+cu118 的标准 GPU 计算，无自定义硬件修改。
  - Baseline 的核心缺陷：(1) PT 使用单一的共享 soft prompt，缺乏对不同输入语义的适应能力——同一 prompt 处理所有输入，无法根据输入内容动态调整；(2) 参数量固定，每个 soft prompt 占据 T×H 参数，多个 task 需要多个完整的 prompt，线性增长；(3) 在数学推理任务上表现弱（PT 46.16% accuracy vs LoRA 56.47%），在 QA 任务上表现较好但并非最优。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - PT-MoE 通过两大核心设计解决 PT 缺陷：
    1. **Matrix Decomposition（矩阵分解）解决参数效率与共享问题**：将每个 expert soft prompt P_i∈R^{T×H} 分解为 P_i = A_i B，其中 A_i∈R^{T×R} 为 prompt 专属矩阵，B∈R^{R×H} 为所有 experts 共享矩阵。总参数量从 O(NTH) 降至 O(NTR+RH)。这实现了：(a) 参数跨 expert 共享——B 矩阵捕获公共知识，各 expert 的 A_i 仅编码差异化知识；(b) 低秩约束作为正则化，防止过拟合，尤其在 out-of-domain 数据上提升泛化。
    2. **Dynamic MoE Router 解决输入自适应问题**：Router 根据输入 embedding 均值 μ∈R^H 动态计算 routing weights w = softmax((Wμ+b) ⊙ (1+ε))，使用 top-k 硬选择（straight-through estimator）和 probationary routing（输出乘 router confidence），实现输入条件化的 expert 选择与加权聚合 P = Σᵢ w_i A_i B。

  - 对比 baseline 的全栈执行例子（PT-MoE, LLaMA-3.2-1B-Instruct, 1 node 4×A100 80GB, MRQA QA 任务, 80K trainable params）：
    (a) 模型推理算法层——输入 text → embedding E (b×s×h) → mean pooling μ (b×h) → linear router Wμ+b (b×n=2) → noise injection + softmax + top-k hard selection → routing weights w (b×2) → 对 i=1,2: weighted A_i ∈ R^{k×d} (k=40, d=36) → summed P_raw = Σ w_i A_i (b×k×d) → project to model dim P = P_raw × B (B∈R^{36×2048}) → concat(P, E) → frozen LLaMA decode → NLL loss only on non-prompt positions → backprop to router + {A_i} + B（80K params）。初始化：task-relevant texts → word embeddings → SVD → UΣV^T → A_i init from U_{:R} Σ_R^{1/2}, B init from Σ_R^{1/2} V_R^T。
    (b) 系统框架层——与 PT 相同：HuggingFace Transformers + DeepSpeed ZeRO-3, AdamW lr=2e-5, warmup 500 steps, batch_size=32, gradient_accumulation=2。增加 router 的 noise 调度——训练时加乘性高斯噪声 σ² 鼓励探索，推理时去掉噪声保证确定性。
    (c) 编译框架层——论文未明确说明。
    (d) Kernel 调度层——论文未明确说明。router linear + weighted sum + matrix multiply (P_raw × B) 均为标准 PyTorch 操作，无自定义 kernel。
    (e) 硬件架构层——与 PT 相同：4×A100 80GB，无自定义硬件。

  - 效果对比：
    - QA (MRQA F1 avg): PT 56.77% → PT-MoE 58.26%（+1.49 pts），SMoP 56.25%（仅用 MoE 反而下降），DPT 55.77%（仅用分解也下降），证明分解与 MoE 的互补性——单独使用均不如 PT，组合后超越 PT。
    - Math (Accuracy avg): PT 46.16% → PT-MoE 56.91%（+10.75 pts），超越 LoRA 56.47%。在 Division 子集上 PT-MoE 79.16% > LoRA 52.08%，体现跨任务一致性。
    - 参数效率: PT-MoE 80K < LoRA 106K < HydraLoRA 278K，同时性能超越。PT-MoE 用 25% fewer params than LoRA，达到更优性能。
    - 核心设计洞察：矩阵分解提供参数共享与正则化（B 矩阵跨 expert 共享低秩基），MoE routing 提供输入条件化动态适应（不同输入激活不同 A_i 组合）。两者协同：没有分解，MoE 的参数随 expert 数线性增长失去效率优势；没有 MoE，分解的共享 B 缺乏对不同输入模式的差异化能力。
