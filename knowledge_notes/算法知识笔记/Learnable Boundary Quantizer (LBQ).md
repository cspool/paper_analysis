## Learnable Boundary Quantizer (LBQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learnable Boundary Quantizer (LBQ) 是 PassionSR 提出的可训练量化器，用于 one-step diffusion 图像超分模型的 PTQ。与传统 PTQ 方法使用固定 min/max 或 grid search 确定 clip bounds 不同，LBQ 将量化边界 B_l（下界）和 B_u（上界）声明为可学习参数，通过梯度下降在校准集上直接优化最优量化区间。其核心公式为：(1) X_c = Clip(X, B_l, B_u)，将输入裁剪到可训练边界内；(2) α = (B_u - B_l) / (2^N - 1)，计算量化 scale；(3) X_I = round((X_c - B_l) / α)，映射到离散整数；(4) X_q = α · X_I + B_l，fake-quantized 输出。与 LSQ（Learned Step Size Quantization）的差异在于：LSQ 仅训练 step size s，而 LBQ 同时训练上下两个边界 B_l 和 B_u，等价于同时训练 scale 和 zero point，并允许非对称量化间隔。LBQ 通过直通估计器（STE）反向传播梯度：当 X ∈ [B_l, B_u] 时 ∂L/∂X = 1，否则为 0。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PassionSR 中 LBQ 应用于 UNet 和 VAE 的所有可量化层（Linear、Conv、MatMul），与 LET 等效变换协同使用：
```
# LBQ 前向传播（fake quantization）
def lbq_forward(X, B_l, B_u, N_bits):
    X_c = torch.clamp(X, B_l, B_u)
    alpha = (B_u - B_l) / (2**N_bits - 1)
    X_int = torch.round((X_c - B_l) / alpha)
    X_q = alpha * X_int + B_l
    return X_q

# LBQ + LET 在 Linear 层中的联合使用
# 可训练参数: B_l, B_u (LBQ, per-tensor), s, δ (LET, per-channel)
X_tilde = (X - δ) / s               # LET 等效变换
W_tilde = s * W                      # LET 等效变换
B_tilde = B + δ @ W                  # LET 等效变换
Y_q = lbq_forward(X_tilde, B_l_a, B_u_a, 8) @ lbq_forward(W_tilde, B_l_w, B_u_w, 8) + B_tilde
```
仅 B_l 和 B_u 可训练（每个量化器 2 个参数），权重 W 保持冻结。在 DQC Stage 1 中 LBQ 冻结、仅训练 LET；在 Stage 2 中 LBQ 和 LET 联合训练。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LBQ 的实现基于 PyTorch 的 fake quantization 机制：(1) 为每个待量化层创建一对可学习参数 B_l, B_u；(2) 前向传播中执行 fake quant（含 clamp + round + rescale）；(3) 反向传播中通过 STE 近似梯度；(4) 校准完成后，将 B_l, B_u 固化为 INT8/INT6 推理的 scale 和 zero-point。PassionSR 代码仓库（https://github.com/libozhu03/PassionSR）在 `ptq_quantize_single.py` 中实现了 LBQ 与 LET 的联合训练流程。LBQ 也可独立使用（消融实验中 LBQ-only 的 W6A6 PSNR=23.15），但与 LET 联合使用时性能显著提升（PSNR 25.40）。

涉及论文标题：
- PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution

---
