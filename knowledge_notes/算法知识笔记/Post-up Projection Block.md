## Post-up Projection Block

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-up Projection Block 是 xLSTM 7B 引入的 block 架构设计，将 mLSTM cell 直接放在 embedding 维度（d_model）运行，并在 mLSTM 层后接独立的 position-wise SwiGLU MLP。与 Mamba 和早期 xLSTM 的 *pre-up projection block*（先 up-project 到更高维度 -> mLSTM/mamba -> down-project 回 embedding 维度，无独立 FFN）形成对比。Post-up 设计的动机：(1) mLSTM 操作的计算量和 GPU 内存消耗随维度线性增长，因此在较低维度（embedding dim）运行可大幅减少开销；(2) 添加独立 SwiGLU MLP 增加了高度优化的线性层（矩阵乘法）FLOPs 占比（Tensor Core 利用率更高）；(3) 丢弃 channel-wise convolution 和 learnable skip connection 等小 kernel 调用，避免 GPU 利用率下降。xLSTM 7B 的 32 个 block 均采用此设计，获得 3.5× 训练速度提升（1.4B 参数规模），且不影响下游任务性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Post-up Projection Block (xLSTM 7B)
输入: x ∈ R^{T×d}  (d = 4096)

// Block 内两层：
// Layer 1: mLSTM (sequence mixing, 在 embedding dim 运行)
x_norm = RMSNorm(x)
x_mix = mLSTM(x_norm)       // multi-head mLSTM cells, 每 head 维度 d_hv=d/N_head
z = x + x_mix                // 残差连接

// Layer 2: Gated MLP (channel mixing)
z_norm = RMSNorm(z)
z_mlp = SwiGLU(z_norm)      // SwiGLU: x ⊙ σ(W_gate @ x) 经 W_up 投影
y = z + z_mlp                // 残差连接

// 对比 Pre-up Projection Block:
// x → UpProj(x) ∈ R^{factor×d} → Conv → mLSTM/SSM → DownProj → output
// (无独立 MLP 层)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源实现位于 https://github.com/NX-AI/xlstm
- SwiGLU MLP 使用 projection factor 2.66（常见于 Transformer），2 个 linear 层（gate + up）后接 SiLU 激活和 element-wise 乘，再过 1 个 linear down-project
- Norm 层使用 RMSNorm（pre-norm 设置）
- 32 个 block 堆叠，总参数 6.87B
- 该设计同时提高了推理吞吐：xLSTM 7B 在 H100 上推理比 Falcon-Mamba 和 Codestral-Mamba 快约 50%

涉及论文标题：
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

---
