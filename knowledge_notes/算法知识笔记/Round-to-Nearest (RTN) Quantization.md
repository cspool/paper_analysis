## Round-to-Nearest (RTN) Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Round-to-Nearest（RTN）量化是最朴素的权重量化方法，也称为 Naive Rounding 或 Nearest Rounding。其过程为：对每个权重值 w，计算 w/scale 得到浮点索引，然后用 round() 函数将其映射到最近的整数索引（即最近的量化级别），反量化时乘以 scale 恢复。RTN 无需校准数据，不进行任何优化，计算开销极小。在 GPTQ 和 AWQ 等高级 PTQ 方法出现之前，RTN 是 LLM 量化的默认 baseline。AFPQ 论文在 RTN 基础上验证了非对称 FP 量化的有效性——表 1、表 2 的 RTN 实验对比了 INT4/NF4-sym/NF4-asym 等格式在不同 group-size 下的 WikiText-2 ppl 和 MMLU 精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AFPQ 论文中 RTN 量化（对称 FP 版本）的计算过程（Algorithm 1: FPSYMQuant/FPSYMDequant）：
```
def FPSYMQuant(weight_tensor, group_size, range):
    for each group of group_size weights:
        w_max = max(group)
        w_min = min(group)
        scale = max(w_max, abs(w_min)) / (range / 2)
        w_4bit = round(weight / scale)
    return w_4bit, scale

def FPSYMDequant(w_4bit, scale):
    w_deq = scale * w_4bit
    return w_deq
```
AFPQ 的非对称版本（FPASYMQuant/FPASYMDequant）在此基础上为正值和负值使用不同的 scale：
```
scale_pos = w_max / (range / 2)
scale_neg = -w_min / (range / 2)
w_deq = scale_pos * w_4bit_pos + scale_neg * w_4bit_neg
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RTN 量化通常在 PyTorch 中自定义实现。简单流程：(1) 分组 weight tensor（按 group_size 或 per-tensor）；(2) 计算每组的 scale 参数（可选 zero_point）；(3) `torch.round(weight / scale)` 得到量化索引；(4) 存储量化索引和 scale。在 bitsandbytes 库中，RTN 量化被用作 baseline，NF4 格式的量化即默认使用 RTN。在 HuggingFace transformers 中可通过 `load_in_4bit=True` 结合 `BitsAndBytesConfig` 使用。KIVI 中使用 group-wise RTN 量化 KV cache，对 key cache 沿 channel 维度、value cache 沿 token 维度分别分组计算 scale/zero-point 后 round-to-nearest。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
