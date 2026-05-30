## Layer-wise Non-uniform Quantization (LNQ / 逐层非均匀量化算法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LNQ (Layer-wise Non-uniform Quantization) 是 GuidedQuant 论文提出的一种新的 weight-only non-uniform scalar PTQ 算法。它为每个 output channel 维护独立的 codebook，通过 alternating minimization 交替优化 codebook 向量 c 和 assignment 矩阵 P：(1) Codebook 更新：给定 assignment P，将问题退化为标准最小二乘，使用闭式解 `c = (PᵀHP)⁻¹PᵀHw`，其中 H=XᵀX 是 layer-wise Hessian；(2) Assignment 更新：给定 codebook c，使用 cyclic coordinate descent (CD) 优化，每次更新一个输入维度 i 的权重值 `Ŵ_i = Round(W_i - H_{i,others}/H_{ii} · (Ŵ_others - W_others))`。LNQ 保证目标函数单调递减并收敛（Proposition 4.1）。与 GPTVQ 1D（使用梯度下降优化 codebook + GPTQ 优化 assignment）相比，LNQ 的闭式 codebook 解和 CD assignment 优化都是更优的选择。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LNQ 算法伪代码（Algorithm 2）：
```
# 输入：H ∈ R^{d_in×d_in}, W ∈ R^{d_in×d_out}, 初始 assignment P
# 超参数：T（交替迭代数）, K（CD 循环轮数）
# 典型值：T=2, K=4 (7B/13B); T=1, K=4 (70B)

H = LLᵀ                                      # Cholesky 分解

for j = 1 to d_out:                          # 每个 output channel 独立并行
    for t = 1 to T:                           # 交替优化
        # Codebook 闭式解（least squares）
        c = (PᵀHP)⁻¹PᵀHw_j                   # O(d_in² m)
        ŵ_j = P @ c                           # 量化权重
        
        # Assignment 优化（cyclic CD）
        for k = 1 to K:                        # K 轮 CD 循环
            for i = 1 to d_in:                 # 遍历每个输入维度
                # 坐标下降闭式解（Eq. 11）
                Ŵ_{i,j} = Round(W_{i,j} - H_{i,[d_in]\i}/H_{i,i} · (Ŵ_{[d_in]\i, j} - W_{[d_in]\i, j}))
        
        # 更新 P：P_{i,q}=1 当 Ŵ_{i,j}=c_q，否则 0
```

GPU 加速技巧：(1) Precomputation trick：预计算 future coordinates 的 Hessian 贡献 B=StrictUpper(H̃)(Ŵ-W)，减少每次 CD 更新的 FLOPs；(2) Lazy batch-updates：将坐标按 batch_size b 分块处理，每块内依次更新、仅 block 内全局修正，减少 memory-bound 全局操作。两技巧组合实现 4× 加速（Llama-2-7B 4-bit: 3.9h → 0.9h on 1×RTX 6000 Ada）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LNQ 适用于 weight-only non-uniform scalar 量化（2-4 bits）。与 SqueezeLLM（k-means + diagonal Fisher）相比，LNQ 使用 layer-wise output error 目标且用闭式解+CD 替代 Lloyd's 算法。与 GPTVQ 1D 相比，LNQ 用闭式 codebook 解替代梯度下降、用 CD 替代 GPTQ，在所有 settings 下均优于 GPTVQ 1D。可进一步集成 GuidedQuant 的 end loss guided 目标（LNQ + GQuant），将原始 Hessian H=XᵀX 替换为 guided Hessian H̄_k。代码随 GuidedQuant 开源（github.com/snu-mllab/GuidedQuant）。

涉及论文标题：
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

---
