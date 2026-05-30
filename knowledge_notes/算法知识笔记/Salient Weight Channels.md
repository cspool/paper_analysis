## Salient Weight Channels

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Salient Weight Channels（显著权重通道）是 AWQ 论文发现并命名的概念：LLM 权重中约 0.1%-1% 的通道对模型性能至关重要。关键发现：显著性的判断依据是**激活分布**而非权重分布——通过 activation magnitude（per-channel 平均激活幅度）筛选出的 top 0.1%-1% 通道，保持其为 FP16 可将 INT3-g128 量化的 OPT-6.7B WikiText-2 PPL 从 43.2 降至 13.0（接近 FP16 baseline 10.86）。而基于权重 magnitude 或随机选择的同等比例通道则几乎无效（PPL 仍 > 23）。直观解释：激活幅度大的通道处理更重要的特征（对应输出维度贡献大），因此这些通道对应的权重对模型精度至关重要。这一发现为 AWQ 的 per-channel scaling 策略提供了理论依据——不需要真正的混合精度（部分 FP16 + 部分 INT），而是通过等效缩放降低显著通道的量化误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
显著权重的识别与保护流程（以 OPT-6.7B INT3-g128 为例）：
```python
# Step 1: 用校准集收集激活统计
X = collect_activations(model, calibration_data)  # [C_in, total_tokens]
s_X = X.abs().mean(dim=1)  # per-channel 平均激活幅度

# Step 2: 识别显著通道 (按激活幅度排序，取 top k%)
k = 0.01  # 1% = top 1% channels
threshold = torch.quantile(s_X, 1 - k)
salient_mask = s_X >= threshold  # [C_in], True for salient channels

# Step 3 (直接方法 - 混合精度): 
# 保持显著通道为 FP16，其余为 INT3
W_salient = W[:, salient_mask]   # FP16
W_non_salient = W[:, ~salient_mask]  # INT3 量化

# Step 4 (AWQ 方法 - per-channel scaling):
# 避免混合精度，用等效缩放保护显著通道
s = torch.ones(C_in)
s[salient_mask] = s_X[salient_mask] ** alpha  # 放大显著通道
W_q = groupwise_quantize(W * s.unsqueeze(0))
# diag(s)^{-1} 融合进前一层
```

Table 1 关键数据（OPT-6.7B, INT3-g128, WikiText-2 PPL↓）：
| 保护方法 | 0.1% FP16 | 1% FP16 | 3% FP16 |
|----------|-----------|---------|---------|
| 基于激活分布 | 11.58 | 11.39 | 11.36 |
| 基于权重分布 | 23.41 | 22.37 | 22.45 |
| 随机选择 | 23.54 | 24.23 | 24.22 |
| 全量 RTN (无保护) | 23.54 |
| FP16 (上界) | 10.86 |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
显著通道的识别仅需一次校准集前向传播 + per-channel 平均幅度统计，无需反向传播或重建，因此极快且数据高效。在 AWQ 实现中，每个需要量化的 Linear 层（Q/K/V/O/FC1/FC2 等）独立计算显著通道并搜索最优 scale。实际部署中，显著通道的 scale 和量化权重一起序列化保存，推理框架（vLLM/TensorRT-LLM）加载后自动执行等效变换。显著性概念启发了后续工作：SmoothQuant 使用 per-channel scaling 平滑激活 outlier；GPTQ + AWQ 组合中，先做 per-channel scaling 保护显著通道再做二阶误差补偿，取得更好 INT2 结果。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- PB-LLM Partially Binarized Large Language Models
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

PB-LLM 从二值化角度补充了显著权重检测的另一个维度：**检测粒度和准则的选择**。PB-LLM 比较了 (1) Magnitude vs Hessian 两种检测准则，发现在 PTQ 场景下 Hessian 准则（v_i = w_i²/[H⁻¹]_{ii}²，源自 SparseGPT）略优于 Magnitude，但差异在 QAT 中不明显；(2) Element-wise vs Column-wise 两种粒度，发现 LLM 中显著权重呈均匀随机散射分布（无列聚集模式，见图 3），因此 column-wise 选择会损害二值化性能，必须使用 element-wise 检测。这与 AWQ 的 per-channel activation-based 检测形成互补——AWQ 关注激活 outlier 通道（per-channel），PB-LLM 关注权重大幅值单点（element-wise）。

SLiM 将显著权重概念进一步推广到联合权重-激活显著性：saliency = |diag(x_mean) × W|，其中 x_mean 为校准集逐输入通道的平均激活幅度。与 AWQ 仅用激活幅度不同，SLiM 的显著性同时考虑权重和激活的幅值，作为量化误差补偿（SLiM-LoRA）和激活感知量化（SLiM-Quant^O）的统一重要性度量。在 SLiM-LoRA 中，该显著性函数满足可逆性和可加性，使低秩适配器能通过 SVD 直接从误差显著性矩阵中闭式推导。

---
