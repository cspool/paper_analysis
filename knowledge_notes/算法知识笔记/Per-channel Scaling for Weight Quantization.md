## Per-channel Scaling for Weight Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-channel Scaling 是 AWQ 提出的保护显著权重的等效变换技术。核心思想：在权重量化前，对显著通道的权重乘以一个大于 1 的缩放因子 s，同时对对应激活通道除以 s（等效变换保持 MatMul 输出不变）。数学上：`Q(w·s) · (x/s)`，量化误差比例从 1 变为 `Δ'/Δ · 1/s`。由于：(1) 单元素缩放通常不改变 group 的 max → `Δ' ≈ Δ`；(2) `s > 1` → 显著权重的相对误差降低。Scale s 通过参数化搜索确定：`s = s_X^α`，其中 s_X 为 per-channel 平均激活幅度，α ∈ [0,1] 单一超参。最优 α 通过 20 步网格搜索最小化 `||Q(W·diag(s))(diag(s)^{-1}·X) - WX||` 确定。最终 diag(s)^{-1} 融合进前一层（如 LayerNorm weight 或前一层 Linear weight），不增加推理开销。与 KV Cache 量化中的 per-channel scaling（在线实时计算，BitDecoding）关键区别：AWQ 的 weight per-channel scaling 是离线预计算的，不引入运行时延迟。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 OPT-6.7B 某 Linear 层 INT3-g128 量化为例：
```python
# 输入: W [C_out, C_in] FP16, X [C_in, L] FP16 (校准集 cached activations)

# Step 1: 计算 per-channel 激活幅度
s_X = X.abs().mean(dim=1)  # [C_in], 每通道平均激活幅度

# Step 2: 网格搜索 α
best_alpha, best_loss = 0.0, float('inf')
for alpha in linspace(0, 1, 20):  # grid_size=20
    s = s_X ** alpha  # [C_in]
    
    # 等效变换的前向模拟
    W_scaled = W * s.unsqueeze(0)          # [C_out, C_in]
    W_q = groupwise_int_quantize(W_scaled, bits=3, group_size=128)
    # 反量化: W_deq = dequantize(W_q)  [C_out, C_in]
    # 激活反向缩放
    X_inv_scaled = X / s.unsqueeze(1)      # [C_in, L]
    Y_q = W_deq @ X_inv_scaled             # [C_out, L]
    Y_fp = W @ X
    
    loss = (Y_q - Y_fp).pow(2).mean()
    if loss < best_loss:
        best_alpha = alpha

# Step 3: 最终量化
s_final = s_X ** best_alpha
W_final_q = groupwise_int_quantize(W * s_final.unsqueeze(0), bits=3, group_size=128)
# diag(s)^{-1} 融合进前一层
```

Table 2 数据（OPT-6.7B, s_X 的 top 1% channels 缩放 s=2）：
- 未缩放 (RTN): Wiki-2 PPL = 23.54
- s=1.25: PPL = 12.87
- s=1.5: PPL = 12.48
- s=2 (最优): PPL = 11.92
- s=4 (过保护): PPL = 12.36 (non-salient 误差增大)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Per-channel scaling 在 AWQ 中被离线计算并存储在模型中。推理时的实现：(1) 激活侧——前一层输出的 hidden state 逐元素除以 per-channel scale（可融合进 LayerNorm 的 γ 和 β 参数，或融合进前一层 Linear 的 weight/bias）；(2) 权重侧——量化权重存储时已包含 scaling 信息（W_scaled = W · s 后再量化），因此无需额外推理逻辑。在 AutoAWQ 代码库中，per-channel scale 在量化脚本中计算并与模型权重一起保存为 safetensors。vLLM 和 TensorRT-LLM 推理时自动识别 AWQ 格式的模型，加载 per-channel scale 并融合进前一层。这种离线计算 + 推理融合的方式使 AWQ 的 per-channel scaling 在精度提升的同时零运行时开销。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

**FlatQuant 中的 Per-channel Scaling**：FlatQuant 在预量化仿射变换前显式引入可学习的逐通道缩放向量 diag(c) ∈ R^n，作为三个核心组件之一（LT + PS + LCT）。在 LLaMA Transformer 架构中，diag(c) 作用于激活 X 进入线性层之前：Y = (X diag(c)^{-1}) · (diag(c) W^T)。缩放因子 c_j 与仿射变换矩阵 P₁、P₂ 和裁剪阈值 α 联合通过梯度下降优化（Eq.4, MSE loss）。diag(c) 可以逐对融合到前层 LayerNorm 或线性层中消除推理开销。在 down-projection 层中，diag(c) 作用在仿射变换后的激活 X̃ 上，然后融合到 up-projection 的权重 W_u 中。消融实验（Table 16）表明 PS 叠加在 LT 上可将 WikiText-2 PPL 从 8.50 进一步降至 7.95。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- FlatQuant: Flatness Matters for LLM Quantization
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

**ParoQuant 中的 Channel-wise Scaling**：ParoQuant 将 channel-wise scaling diag(α) 与 independent Givens rotations 联合使用构成 scaled pairwise rotation：T(W) = (∏R_t)·diag(α)·W。与 AWQ（grid search 搜索 α）不同，ParoQuant 的 α 通过 AdamW 梯度下降优化（lr=0.05，初始化为 1），且与旋转角度 θ 联合优化——scaling 负责全局幅值均衡，rotations 负责 token 级跨通道值对齐。同时 ParoQuant 的 scaling 逆变换 diag(1/α) 无法简单合并到前序算子（因为后接 rotations），而是在 fused CUDA kernel 中与 rotations 一起在线计算，总开销约 10%。

---
