## Cyclic Coordinate Descent for Quantization Assignment（循环坐标下降量化分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cyclic Coordinate Descent (CD) 是 LNQ 中用于优化 quantization assignment 的迭代算法。给定固定 codebook c 和 Hessian H，CD 在每一轮中按固定顺序（i=1→d_in）依次更新每个权重值：对第 i 个输入维度，选择使目标函数 `(ŵ-w)ᵀH(ŵ-w)` 最小化的 codebook 值。核心闭式解（Behdin et al., 2023 Lemma 1）：`Ŵ_i = Round(W_i - H_{i,others}/H_{ii} · (Ŵ_others - W_others))`，其中 Round 将值映射到最近的 codebook entry。CD 是 descent method：在初始化 feasible solution 的前提下保证目标函数单调递减。LNQ 中 CD 以当前 assignment+codebook 对应的量化权重为初值，保证每次 CD 调用不增加目标值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
带预计算优化的 CD 算法（Algorithm 4）：
```
# 输入：H, W, c, Ŵ（当前量化权重）, Q（rounded indices）
H̃ = diag(H)⁻¹H                              # 归一化 Hessian
U = StrictUpper(H̃)                            # 严格上三角

for k = 1 to K:                               # K 轮 CD 循环
    B = U @ (Ŵ - W)                           # 预计算 future 坐标贡献
    
    for s in 1, b+1, 2b+1, ..., d_in-b+1:    # lazy batch (b=128)
        for i = s to s+b-1:                   # batch 内 sequential
            Ŵ[i,:] = Round(W[i,:] - B[i,:])   # 坐标下降更新
            Q[i,:] = RoundIdx(W[i,:] - B[i,:])
            B[i+1:s+b, :] += U[i+1:s+b, i] @ (Ŵ[i,:] - W[i,:])  # 局部修正
        
        # batch 完成后全局修正
        B[s+b:, :] += U[s+b:, s:s+b] @ (Ŵ[s:s+b, :] - W[s:s+b, :])

# 从 Q 提取 P：P_{i,q}^{(j)} = 1 if q=Q_{i,j} else 0
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CD 在 LLM 量化中的使用：(1) QuantEase (Behdin et al., 2023) 首次将 cyclic CD 用于 uniform 量化；(2) CDQuant (Nair & Suggala, 2024) 证明 greedy CD 优于 GPTQ，cyclic CD 性能接近但计算成本更低；(3) QuIP (Chee et al., 2024) 使用 CD 作为 post-GPTQ/LDLQ 精炼步骤；(4) LNQ (GuidedQuant, 2025) 将 CD 嵌入 alternating minimization 框架用于 non-uniform 量化，在 GuidedQuant 的 ablation study（Table 14）中验证 CD 在所有 settings 下匹配或优于 GPTQ。GPU 上的优化实现使用 precomputation + lazy batch-updates 可达 4× 加速。

涉及论文标题：
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

---
