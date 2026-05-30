## NormalFloat (NF4/NF3)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NormalFloat（NF）是 Dettmers et al. (2021) 设计的专为神经网络权重量化优化的数据类型。其核心思想：假设预训练神经网络权重近似服从正态分布 N(0, σ)，将量化级别按照正态分布的分位数等距划分，使得每个量化 bin 内的概率质量（权重数量）近似相等（信息论最优）。NF 格式的值在零附近密集（对应大部分权重集中区域），远离零则稀疏（对应长尾大权重）。NF4 有 16 个候选值，NF3 有 8 个候选值。具体值（来自 bitsandbytes）：NF4 = [-1, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1]；NF3 = [-1, -0.535, -0.247, 0, 0.183, 0.382, 0.623, 1]。AFPQ 论文中的 NF4-asym 和 NF3-asym 方法在标准 NF 格式基础上增加了非对称 scaling，显著提升了量化精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
NF 格式的设计源于信息论：最优量化器应当使每个量化 bin 覆盖相等的概率质量（Max-Lloyd 量化器）。对于标准正态分布 N(0,1)，其 CDF 的反函数（分位数函数）直接给出最优量化级别的值：
```
def create_nf_levels(num_levels):
    # 对标准正态分布按等概率间隔取分位数
    probs = np.linspace(0, 1, num_levels + 1)
    # 取每个 bin 的中点（概率平均值处）的分位数
    mid_probs = (probs[:-1] + probs[1:]) / 2
    levels = norm.ppf(mid_probs)  # 正态分布分位数函数
    levels = levels / max(abs(levels))  # 归一化到 [-1, 1]
    return levels
```
AFPQ 的非对称版本在此之上，将正负值分别乘以不同 scale。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NF4 格式由 bitsandbytes 库（https://github.com/TimDettmers/bitsandbytes）首先实现并推广。使用方式：`from transformers import BitsAndBytesConfig; config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`。在 HuggingFace 生态中，NF4 是 `load_in_4bit=True` 的默认量化格式。在 AFPQ 论文中，NF3 也被用来进行 3-bit 量化实验，在 GPTQ/AWQ 中均使用 NF3-asym 替代 INT3。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention
- Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4)

---
