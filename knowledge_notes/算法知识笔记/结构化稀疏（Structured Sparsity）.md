## 结构化稀疏（Structured Sparsity）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 结构化稀疏是把稀疏模式约束到规则、可预测结构（块/带状/分块对角/蝴蝶/窗口/2:4 等）上的稀疏化方法：相比非结构化稀疏（逐元素任意位置置零），它牺牲一定压缩率换取"可预测性"——稀疏模式由固定块形状、固定变换层级或确定性混合路径决定，而不是任意不规则非零分布。LLM/Transformer 语境下它同时提供算法压缩与架构规整性：暴露可复用的数据流、有界依赖与可特化的执行调度（MLX 论文 II-A 对结构化稀疏的定义）。典型形式包括块对角矩阵分解（butterfly factorization）、2:4 半结构化稀疏（NVIDIA Sparse Tensor Core 支持，每 4 个连续元素恰好 2 个非零）、block-wise N:M、分块对角因果掩码、以及 FFT/滑动窗口等固定混合模式。MLX 论文的核心观察是：结构化算子的数据流图具有"前向分层、有界局部性"的公共执行形态（closed-set locality），可折叠到紧凑空间阵列执行。
- 与半结构化/非结构化稀疏的关系（本地知识库旁证）：知识库已有 N:M 半结构化稀疏（N_M Semi-structured Sparsity）、2:4 半结构化稀疏（2_4 Semi-structured Sparsity）与 BBC（Bitmap-Bitmap-CSR）等条目——2:4 要求固定 50% 稀疏率且对齐 tensor core 4×4×4 粒度；Mustafar 等 KV cache 剪枝工作指出非结构化稀疏可到 70% 稀疏度但需要专用 kernel/硬件处理不规则索引，而结构化稀疏牺牲比例换硬件友好性。MLX 的蝴蝶/FFT 结构化稀疏则走向另一极：稀疏模式完全确定（无需索引），代价是分解本身引入近似误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MLX 混合化 Transformer block 中的结构化稀疏 pipeline（伪代码，s=压缩率、B=蝴蝶块大小）：
```
# 沿序列维 N：结构化 = 固定长度 L 的 chunk 内 FFT + 低频截断（FFT-CMP）
for c in range(N//L):
    F = FFT_L(Q[cL:(c+1)L, :])     # 每 chunk 一个 L 点 FFT（固定混合模式）
    F_trunc = F[:sL, :]            # 截断高半频，保留 sL 个低频系数（确定性）
    Qs[c] = IFFT_{sL}(F_trunc)     # 缩短 token 序列 → N 变 sN
# 沿隐藏维 D：结构化 = B×B 块内蝴蝶分解（hierarchical BSMM）
#   W → (D/B)×(D/B) 个 B×B tile，每 tile W_b = ∏_{k=1}^{log2 B} B_B^(k)（块对角蝴蝶因子）
#   Y = X @ W  ≈ 逐 tile 的蝴蝶稀疏矩阵乘，复杂度 O((D²/B)·log B)
```
对比非结构化稀疏（如 Wanda 逐元素剪枝）需索引数组/位图与 gather 访存；2:4 半结构化对齐 tensor core 但固定 50% 稀疏率；MLX 的结构化稀疏把稀疏模式变成"分层交换 + 截断"的确定算子，非零位置编译期已知——这使数据流可在片上静态路由（蝴蝶 stride ±2/±4/±8 映射到 skip-hop 网格），这是非结构化稀疏做不到的。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用分层：(1) 算法层——分解/截断得到结构化矩阵（蝴蝶因子 B_B^(k)、FFT 截断比 s、块大小 B），精度-效率由 (s, B) 双旋钮调节（s=0.75/0.5、B=16/32/64，B=32 最优）；LLM 上配合 LoRA 微调压缩层恢复精度（Llama2-7B/InternLM2-7B 超 60% 层应用后 QKV+Attention 计算削减 57%-72%、精度降 <1.45%）。(2) kernel/硬件层——GPU 上蝴蝶/FFT kernel 落 CUDA core（TensorCore 支持 2:4 类规则稀疏但不支持蝴蝶，导致执行单元不匹配、速度增益远小于 FLOP 削减）；MLX 空间阵列上用 CDC + tagged block 把稀疏依赖折叠成跨层流水，roofline 利用率 52%-84%。(3) 使用例子（ViT 从头训练验证）："bd.*" 块分解替代稠密投影削减 45%-55% FLOP 仅轻微精度损失，2D-FFT token mixing（FNet）同 FLOP 削减但 2-3% 精度损失，FFT-CMP（s=0.5）65% FLOP 削减仅 1.6% 精度下降。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
