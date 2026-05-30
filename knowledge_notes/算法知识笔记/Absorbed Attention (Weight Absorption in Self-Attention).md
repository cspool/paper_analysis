## Absorbed Attention (Weight Absorption in Self-Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Absorbed Attention 是利用矩阵乘法结合律，将 KV cache 的 up-projection 矩阵 W_K/W_V 分别吸收进 query projection W_Q 和 output projection W_O，避免推理时显式计算完整 K、V 矩阵的技术。由 MLA 首次引入，MTLA 继承并适配。

核心变换（MTLA 版）：
```
标准: K = Ĉ @ W_K, V = Ĉ @ W_V  → output = softmax(Q@K^T/√d) @ V @ W_O
吸收: W_Q_absorbed = W_Q @ W_K^T, W_V_absorbed = W_V @ W_O
     output = softmax(X @ W_Q_absorbed @ Ĉ^T/√d) @ Ĉ @ W_V_absorbed
```

好处：(1) 避免 up-project Ĉ(r维)→K(n_h·d_h维) 再 down-project 的冗余；(2) Ĉ 直接参与 attention，计算量减少（r vs n_h·d_h）；(3) 内存带宽节省。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**对比：r=256, n_h·d_h=512, d=512, t=T/2**：

```
# 吸收前 K^T 计算量: t·r·(n_h·d_h) = t·256·512 = 131072t FLOPs
# 吸收后 K^T 计算量: N·d·r = N·512·256 = 131072N FLOPs
# 当 N=t (decode) 时两者相同，但免去存储中间 K 的内存和带宽
# 训练时 N=T > t, 吸收后计算量稍增但内存显著减少

# 吸收前 V@W_O 计算量: t·(n_h·d_h)·d + t·r·(n_h·d_h) = t·512·512 + t·256·512
# 吸收后 V@W_O 计算量: t·r·d = t·256·512
# 节省了显式 V 生成的 t·512·512 FLOPs（一半计算量）
```

术语一般如何实现？如何使用？

Weight absorption 前提：位置编码（RoPE）不能直接施加在 latent vector 上（旋转矩阵与 up-projection 矩阵不满足交换律）。因此必须使用 decoupled RoPE——位置部分单独计算 Q_RK_R^T 后加法合并到 content attention score。吸收后权重在 training 后预计算，直接 baked into FlashAttention-2 自定义 CUDA kernel 参数。

涉及论文标题：
- Multi-head_Temporal_Latent_Attention
- TransMLA: Multi-Head Latent Attention Is All You Need

TransMLA 通过 Absorb 操作实现 GQA→MLA 转换后的推理加速。转换后，W^{UK}（NoPE 部分，移除 RoPE 后）被吸收进 query projection：q̂_{t,i} = [(W_i^{UK})^T q_{t,i}^C; q_{t,i}^R]，所有 head 共享一个 latent KV head k̂_t = [c_t^{KV}; k_t^R]，仅需缓存 r_kv 维而非 2gd 维的 KV cache。TransMLA 的 RoRoPE 技术确保转换后的 W^{UK} 不含 RoPE（位置信息集中在第一 head 独立处理），满足 Absorb 操作的前提条件。
