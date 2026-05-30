## Asymmetric Floating Point Quantization (AFPQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AFPQ（Asymmetric Floating Point Quantization，非对称浮点量化）是 Zhang et al. 提出的针对 LLM 权重非对称分布的浮点量化方法。核心创新：为每组 weight group 内的正值和负值分别设置独立的 scale 参数（scale_pos 和 scale_neg），替代传统对称 FP 量化中所有值共享一个 scale 的做法。数学上，传统对称 FP 量化使用公式 `scale = max(w_max, |w_min|) / (range/2)` 和 `w_q = round(w / scale)`，当 weight group 的分布不对称时，绝对值小的一侧会有大量 FP 候选值落在权重范围之外，造成表达能力浪费。AFPQ 使用 `scale_pos = w_max / (range/2)` 和 `scale_neg = -w_min / (range/2)`，分别量化正负值，使得 FP 候选值精确覆盖权重的实际范围。每组存储两个 scale（与 INT-asym 存储 scale+zero_point 的两个参数存储量相同），无额外存储开销。AFPQ 还保留了 FP 格式在零附近密集分布的优势——相比 INT-asym 的"scale+zero_point"方法（会移动 zero point），AFPQ 的双 scale 方案不改变零的位置。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AFPQ 的量化与反量化伪代码（来自论文 Algorithm 1: FPASYMQuant/FPASYMDequant）：
```
def FPASYMQuant(weight_tensor, group_size, range):
    for each group of group_size weights:
        w_max = max(group)
        w_min = min(group)
        scale_pos = w_max / (range / 2)
        scale_neg = -w_min / (range / 2)
        # 正值和负值分别量化
        w_4bit_pos = round(weight[weight > 0] / scale_pos)
        w_4bit_neg = round(weight[weight < 0] / scale_neg)
    return w_4bit, scale_pos, scale_neg

def FPASYMDequant(w_4bit, scale_pos, scale_neg):
    w_deq = scale_pos * w_4bit_pos + scale_neg * w_4bit_neg
    return w_deq
```
与 INT-asym 的对比（INT 使用 scale + zero_point）：
```
# INT-asym Quant
scale = (w_max - w_min) / (2^bit - 1)
zero_point = round(-w_min / scale)
w_int = round(weight / scale) + zero_point

# INT-asym Dequant
w_deq = scale * (w_int - zero_point)
```
AFPQ 的 scale_pos/scale_neg 与 INT-asym 的 scale/zero_point 都是每 group 两个参数，存储量相同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AFPQ 作为即插即用的量化格式，可集成到现有的 W-only PTQ 方法中：(1) RTN 量化——直接使用 FPASYMQuant/FPASYMDequant 替代对称量化；(2) GPTQ 集成——在 OBS 框架的逐列量化步骤中，将 INT quant 替换为 NF-asym quant；(3) AWQ 集成——在 saliency-based scaling 后，将 INT quant 替换为 NF-asym quant。AFPQ 代码开源：https://github.com/zhangsichengsjtu/AFPQ。实验中使用的格式：FP4 E2M1、FP3 E2M0、NF4、NF3。AFPQ 的推理系统基于 FasterTransformer，自定义 NF-asym dequantization kernel（packed byte 解包 → LUT NF→FP16 映射 → 分正负 scale 反量化 → FP16 GEMM）。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

---
