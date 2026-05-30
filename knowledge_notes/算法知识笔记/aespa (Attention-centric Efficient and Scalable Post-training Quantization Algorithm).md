## aespa (Attention-centric Efficient and Scalable Post-training Quantization Algorithm)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
*aespa* 是 Samsung Research 在 NeurIPS 2024 提出的针对超大规模 Transformer 模型的 PTQ 算法。核心策略是"逐层量化 + attention-wise 重构"：每层单独量化以保持效率，但损失函数以 attention 输出重构为目标以引入跨层依赖。aespa 分两步执行：(1) 使用 Z-FOLD 结合提出的 attention-aware Hessian 计算量化参数（scale 和 zero-point）；(2) 使用 AdaRound 结合提出的精炼损失函数优化 weight-rounding policy。关键创新在于精炼量化目标：对 W_V 用 `H_V = 2E[XA^TAX^T]` 替代传统 `H = 2E[XX^T]`，将 Q 和 K 的信息通过 attention map A 耦合进 V 的 Hessian；对 W_Q/W_K 的损失函数分别引入 `E[K^TK]` 和 `E[Q^TQ]` 注入跨投影依赖。通过预计算这些统计量，每轮迭代仅需 O(d_h d^2) FLOPs，远低于传统 block-wise 方法的 O(B d_h L·max{d,L})。代码开源：https://github.com/SamsungLabs/aespa（CC BY-NC 4.0）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-7B Attention Block 的 INT2 量化为例，aespa 完整 pipeline：
```
# 输入: W_Q, W_K, W_V, W_O, W_FFN, 校准数据 X (128 seqs × 2048 tokens)

# === Phase 1: 全精度前向预计算关键统计量 ===
Q, K, V = W_Q(X), W_K(X), W_V(X)        # 全精度前向一次
A = softmax(Q @ K.T / sqrt(d))          # [B, H, L, L] attention map
A_out = A @ V                           # attention output

H_xx = mean(X @ X.T)                    # [d, d] 用于所有层的标准 Hessian
H_v  = mean(X @ A.T @ A @ X.T)            # [d, d] 用于 W_V 的 attention-aware Hessian
E_ktk = mean(K.transpose(-2,-1) @ K)     # [d_h, d_h] 用于 W_Q 损失
E_qtq = mean(Q.transpose(-2,-1) @ Q)     # [d_h, d_h] 用于 W_K 损失

# === Phase 2: 量化参数计算 (Z-FOLD) ===
for layer in [W_Q, W_K, W_V, W_O, W_FFN]:
    H = H_v if layer == W_V else H_xx    # 选对应 Hessian
    s = argmin_s tr(ΔW(s) @ H @ ΔW(s).T)  # Z-FOLD: 优化 step size

# === Phase 3: Weight-rounding 优化 (AdaRound with proposed losses) ===
for iter in range(2000):
    for W_V:
        ΔW_V = W_hat_V - W_V
        loss = sum((ΔW_V @ H_v) * ΔW_V)           # Equation (17), 一次矩阵乘
    for W_Q:
        ΔW_Q = W_hat_Q - W_Q
        loss = tr(E_ktk @ ΔW_Q @ H_xx @ ΔW_Q.T)   # Equation (21), 两次矩阵乘
    for W_K:
        ΔW_K = W_hat_K - W_K
        loss = tr(E_qtq @ ΔW_K @ H_xx @ ΔW_K.T)   # Equation (22)
    for W_O, W_FFN:
        loss = tr(ΔW @ H_xx @ ΔW.T)               # 标准 layer-wise 损失
    loss += λ * rounding_regularization            # AdaRound rounding loss
    loss.backward(); update(W_int)

# 复杂度: 每轮 O(d_h d^2), 与校准数据量无关
# OPT-125M: C_aespa=0.24 GFLOPS vs C_exist(B=4)=6.7 GFLOPS (28× gap)
```
与 baseline BRECQ 的关键区别：BRECQ 每轮需完整 `SA(Q_hat, K_hat, V_hat)` forward，而 aespa 通过预计算将 attention 操作"折叠"进统计量矩阵，后续迭代完全避开 attention computation。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
aespa 官方实现（https://github.com/SamsungLabs/aespa）提供 quantize.py（主量化入口）、quantizer.py（核心量化器）、aespa.py（精炼损失实现）、quant_utils.py（工具函数）及定制模型文件（modeling_llama_custom.py, modeling_bloom_custom.py）。支持 OPT、BLOOM、LLaMA、LLaMA2 模型族。关键超参：校准集 128 segments × 2048 tokens from C4；weight-rounding 优化 2000 iterations、lr=0.015、rounding loss weight λ=1.5；量化前使用 OPTQ 初始化 W_int 以加速收敛。仅做 weight-only 量化（激活保持 FP16）。处理时间：OPT-125M INT2 约 5 分钟（GPU）；OPT-6.7B INT2 约 10.2 小时。在资源受限场景下可跳过 weight-rounding 优化，仅用 Z-FOLD + attention-aware Hessian 计算量化参数，OPT-1.3B 仅需 0.35 小时，仍优于 Z-FOLD 用标准 Hessian 的性能。

涉及论文标题：
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers
