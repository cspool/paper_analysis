## Per-State-Group Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-state-group quantization 是 Quamba2 提出的针对 SSM 选择性参数 $B_t$ 和 $C_t$ 的量化方法。在 Mamba2 中，$B_t$ 和 $C_t$ 被组织为多个 state group（默认每组 128 channel，共 8 组），组内共享参数（类比 grouped-query attention）。Quamba2 发现 **state persistence**：各 state group 中激活的 group（数值较大的 group）在时间步和输入样本间保持一致（例如 group 6 在 B 中总是高激活，group 7 在 B 和 C 中总是几乎无变化）。基于此，对每个 state group 使用独立的 scaling factor（而非对整个 B/C 的 per-tensor 量化），大幅提升了小数值 group 的量化精度。在 W4A8 Quamba2-8B 消融中，per-state-group 从 55.1% 提升到 60.7% LAMBADA accuracy（+5.6%），是缩小与 FP16 差距的关键技术。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Offline calibration
for each SSM block:
    for each state group g in B:
        s_B[g] = max(|B_t[g]|) / 127.0              # 每 state group 独立 scale
    for each state group g in C:
        s_C[g] = max(|C_t[g]|) / 127.0

# Online inference
B_quant[g] = clamp(round(B_t[g] / s_B[g]), -127, 127)
C_quant[g] = clamp(round(C_t[g] / s_C[g]), -127, 127)
# 注意：cached SSM states 使用相同的 head/channel group indices
# 因此直接复用 x_t 的 sort+cluster index，无需额外 online reorder
h_t = A_t * h_{t-1} + B_quant * x_quant
y_ssd = C_quant * h_t
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该技术特别适合 Mamba2 的 multi-input SSM 架构（有显式 state group），对 Mamba1 也可用类似逻辑（按 head 分组）。开销极小：仅需存储 G 个额外的 FP16 scaling factors（G=state group 数，通常 ≤8）。论文发现 cached SSM states 自然地遵循与 SSM 输入 x 相同的 head/channel 分组（因为 states 是在 channel-wise SSD scan 中从 x 派生的），因此不需要额外为 states 做 online reordering，直接复用 x 的 scale 分组。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models
