## Static Token-Wise Quantization (STWQ / 静态逐Token量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
STWQ（Static Token-Wise Quantization）是 PTQ4ARVG 提出的一种离线分配 per-token 量化参数的方法，专门利用 ARVG 模型的两大独有特性：(1) 固定 token 序列长度（ARVG 生成固定数量的图像 token，不像 LLM 生成可变长度文本）；(2) 跨样本位置不变分布（不同类别和条件的样本在同一 token 位置的激活分布保持一致）。基于这两点，STWQ 在离线校准时为每个 token 位置静态设定量化参数（scale δ 和 zero point z），推理时直接使用而无需在线校准。具体包括两部分：(a) 对 AdaLN 模块输入沿 token 序列逐位置分配量化参数；(b) 对线性层输入将首 token（sink token，含条件信息，分布显著不同于其他 token）与其他 normal token 分开量化。使用 percentile 校准而非 min-max 以保证精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 VAR-d16 模型的 STWQ 离线校准流程为例（W6A6）：

```
输入: 校准数据 X_cal (128 张 ImageNet), ARVG decoder
输出: 静态 per-token 量化参数

# === AdaLN 模块的 STWQ ===
for each block l in decoder:
    X_adaln = run_forward_get_adaln_input(X_cal)  # shape: [N_samples, T, C]

    # 沿 token 维度逐位置校准
    for t = 1 to T (固定的 token 序列长度):
        # 收集所有样本在位置 t 的激活
        X_t = X_adaln[:, t, :]                     # shape: [N_samples, C]

        # Percentile 校准 (而非 min-max)
        low = percentile(X_t, p_low)               # 如 p_low = 0.1%
        high = percentile(X_t, p_high)             # 如 p_high = 99.9%
        δ_adaln[t] = (high - low) / (2^b - 1)
        z_adaln[t] = round(-low / δ_adaln[t])
    # 存储: {δ_adaln[t], z_adaln[t]}, t=1..T (用于推理)

# === 线性层的 STWQ (sink token + normal token) ===
for each linear layer (qkv, fc1, fc2, etc.):
    X_lin = run_forward_get_linear_input(X_cal)    # shape: [N_samples, T, C_in]

    # Sink token (首 token, t=0)
    X_sink = X_lin[:, 0, :]                        # shape: [N_samples, C_in]
    δ_sink = (percentile(X_sink, 99.9) - percentile(X_sink, 0.1)) / (2^b - 1)
    z_sink = round(-percentile(X_sink, 0.1) / δ_sink)

    # Normal tokens (其余 token, t=1..T-1)
    X_normal = X_lin[:, 1:, :]                     # shape: [N_samples, T-1, C_in]
    δ_normal = (percentile(X_normal, 99.9) - percentile(X_normal, 0.1)) / (2^b - 1)
    z_normal = round(-percentile(X_normal, 0.1) / δ_normal)

# 推理时直接使用预设参数 (无在线校准)
for each inference step t:
    if t == 0:
        δ, z = δ_sink, z_sink
    else:
        δ, z = δ_normal, z_normal  # (或 AdaLN 层的 δ_adaln[t])
    X_int = clamp(round(X / δ) + z, 0, 2^b - 1)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
STWQ 与 Dynamic Token-Wise Quantization（DTWQ，如 LLM.int8）的本质区别：DTWQ 在每次推理时动态计算 min-max 量化参数，引入额外开销（LLM.int8 在 GPT-3-13B 上造成 0.5× speedup loss），且 min-max 校准精度低（VAR 上 DTWQ 导致 FID 降 15.3）；STWQ 利用 ARVG 的固定 token 长度和位置不变分布，将量化参数完全离线设定，推理时零额外开销。PTQ4ARVG 在 VAR-d16 W6A6 上的实验表明：STWQ 将 SmoothQuant baseline 的 FID 从 18.54 降至 10.41（+SQ+STWQ），且 speedup 保持 2.92×（vs DTWQ 的 2.46×）。STWQ 兼容标准 CUDA kernel 部署。开源：https://github.com/BienLuky/PTQ4ARVG。

涉及论文标题：
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models
