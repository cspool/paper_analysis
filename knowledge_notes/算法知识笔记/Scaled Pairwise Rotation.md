## Scaled Pairwise Rotation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Scaled Pairwise Rotation（缩放成对旋转）是 ParoQuant 提出的完整预量化权重变换算子 T，由 channel-wise scaling + K 个串联 independent Givens rotations 组成：T(W) = (∏_{t=1}^{K} R(P_t, Θ_t)) · diag(α) · W。其中 diag(α) 是逐通道缩放（α ∈ R^n），R(P_t, Θ_t) 是第 t 个 independent rotation。三者分工：(1) Scaling 拉平全局通道间幅值差异，直接压制"整通道都是离群值"的情况；(2) Rotations 在 token 级别对齐通道对内的值，使数据点聚集到 x=y 线附近（图 1 Right），收窄组内动态范围；(3) 串联多个 rotations 弥补单个 independent rotation 参数有限的问题。推理时，对激活 X 应用逆变换 T^{-1}(X)=X·diag(1/α)·R_1^{-1}·...·R_K^{-1}，在 fused CUDA kernel 中完成。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ParoQuant 两阶段优化流程：
```
# Stage 1: 优化旋转角度 θ 和缩放因子 α
for epoch in 1..10:
    for each layer l:
        Y_fp = l(X)  # 原始层输出
        l' = copy(l)
        for each linear in l':
            for each 128-channel group:
                W_s = diag(α) · W              # Scaling
                W_t = W_s
                for t in 1..K:                  # K=8 rotations
                    for (i,j) in P_t:
                        c,s = cos(θ_t[i,j]), sin(θ_t[i,j])
                        W_t[i,:], W_t[j,:] = c*W_t[i,:]-s*W_t[j,:], s*W_t[i,:]+c*W_t[j,:]
                W_q = RTN_quantize(W_t, 4bit, g=128)  # INT4 均匀量化
        Y_q = l'(X')  # X' 为已量化前层输出
        loss = SmoothL1(Y_q, Y_fp)
        AdamW_update(θ, α, lr=0.05, cosine_decay)

# Stage 2: 微调权重和量化参数
for epoch in 1..10:
    # 固定 θ, α, 微调 W, s_q, z_q
    loss = SmoothL1(Y_q, Y_fp)
    AdamW_update(W, s_q, z_q, lr=1e-5/1e-6)
```
旋转角度 θ 初始化为 0（恒等变换），α 初始化为 1。校准集：2048 样本 × 2048 tokens，WikiText2/C4/RedPajama 均匀混合 + 64 样本 Pile 验证集。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与同类方法的区别：(1) vs AWQ (仅 scaling)——ParoQuant 多了旋转步骤，~10% 推理开销换 2.4% 推理精度提升；(2) vs QTIP (Hadamard 旋转)——Hadamard 是固定的/随机的 O(n log n) 变换，ParoQuant 是可学习的 O(Kn) 变换，参数更少、可并行化、~25% 更快，精度匹敌；(3) vs SpinQuant (可学习全旋转)——SpinQuant 旋转矩阵需合并到前序层权重，仅适用于少数层（output projection），ParoQuant 的 sparse Givens 旋转可直接在线计算，适用所有线性层。

涉及论文标题：
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

---
