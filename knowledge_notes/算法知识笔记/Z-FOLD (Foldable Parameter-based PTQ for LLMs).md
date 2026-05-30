## Z-FOLD (Foldable Parameter-based PTQ for LLMs)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Z-FOLD 是 Samsung Research 在 EMNLP 2023 提出的针对 LLM 的 learning-free PTQ 方案。核心设计：利用 Transformer 中可折叠（foldable）的额外参数——即可以数学合并入其他层而不增加推理开销的参数（如 LayerNorm 的 affine weight）——来更精细地量化权重。Z-FOLD 通过优化 scale 和 zero-point（量化参数）以及 foldable parameters 来最小化基于 Hessian 的重构误差 `tr(ΔW·H·ΔW^T)`，其中 H=2E[XX^T]。Z-FOLD 是 learning-free 方法（无需梯度反向传播优化 weight-rounding policy），仅需前向统计量计算和闭式解优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Z-FOLD 的核心算法步骤：
```
# Z-FOLD: 基于 Layer-wise Hessian 的 learning-free 量化
H = 2 * mean(X @ X.T)                   # [d, d] Hessian 近似

for each layer:
    # Step 1: 量化参数初始化（使用传统的 min-max 或 MSE 方法）
    s_init, z_init = init_quant_params(W)
    
    # Step 2: 利用 foldable parameters 优化量化参数
    # foldable parameters: LayerNorm weight γ, bias β
    # 优化目标: min_{s, z, γ, β} tr(ΔW(s,z,γ,β) · H · ΔW(s,z,γ,β)^T)
    s_opt, z_opt, γ_opt, β_opt = optimize_quant_and_foldable(W, H, X)
    
    # Step 3: 应用优化后的参数进行量化
    W_int = quantize(W, s_opt, z_opt)    # 使用最近舍入 (RTN)
    
    # Step 4: 将 foldable parameters 合并到相邻层
    # γ_opt 合并到前一层输出或当前层权重，推理时零额外开销
```
在 aespa 中，Z-FOLD 被用于量化参数计算阶段（Algorithm 1 line 4），但 Hessian 被替换为 attention-aware 版本：对 W_V 使用 `H_V = 2E[XA^TAX^T]`，显著提升 INT2 精度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Z-FOLD 是 learning-free 方法，不需要 GPU 训练或梯度下降，因此处理速度较快。在 aespa 的实验中，Z-FOLD 是唯一同时展示合理精度和处理时间的 layer-wise 方法（OPTQ 虽快但 INT2 精度崩溃，BRECQ 精度好但不可扩展）。Z-FOLD 的局限性：(1) 依赖最近舍入而非优化 weight-rounding policy；(2) 不考虑 attention 内部跨层依赖（使用标准 H=2E[XX^T]）。aespa 通过在 Z-FOLD 之后增加 AdaRound weight-rounding 优化并替换为 attention-aware Hessian，补齐了这些短板。

涉及论文标题：
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers
