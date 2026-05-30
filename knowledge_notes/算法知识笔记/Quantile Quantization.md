## Quantile Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantile Quantization（分位数量化）是一种信息论最优的量化策略，确保每个量化 bin 包含相等数量的来自输入张量的值，即量化后的值均匀分布。其核心思想：如果权重近似服从正态分布 N(0,σ)，则将正态分布的 CDF 反函数（分位数函数 Q_X）作用于等间距的概率值，得到的量化级别 q_i = (Q_X(i/(2^k+1)) + Q_X((i+1)/(2^k+1))) / 2 能使每个 bin 具有相等的概率质量。相比均匀量化（bin 等距），分位数量化在零附近密度更高（捕获大部分权重），在尾部间距更大，信息利用率更高。QLoRA 的 NormalFloat (NF) 数据类型就是分位数量化的具体实现。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# k-bit Quantile Quantization 的级别生成
from scipy.stats import norm

def quantile_quantization_levels(k):
    num_levels = 2^k
    # 等概率间隔（0到1之间num_levels+1个边界）
    probs = np.linspace(0, 1, num_levels + 1)
    # 每个 bin 的中点分位数
    mid_probs = (probs[:-1] + probs[1:]) / 2
    # 正态分布分位数函数（CDF反函数）
    levels = norm.ppf(mid_probs)
    # 归一化到 [-1, 1]
    levels = levels / max(abs(levels))
    return levels

# 4-bit: q = [-1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091,
#              0.0, 0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0]
# 特点：零点附近最密集（8 values in [-0.1, 0.1]），远离零点逐渐稀疏
```
在 IR-QLoRA 论文中，NF2/NF3/NF4 均使用分位数量化生成量化级别，NF2 使用对称设置（避免信息过度偏离）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
bitsandbytes 库实现了基于分位数量化的 NF4 格式。使用 HuggingFace 的 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` 即可加载分位数量化模型。在 PyTorch 中，可自定义分位数量化器：对每块权重，使用 absmax scaling 将值归一化到 [-1, 1] → 匹配到最近的分位数级别 → 存储级别索引。分位数量化的理论最优性仅对正态分布严格成立；当实际权重分布偏态时，需考虑非对称扩展（如 AFPQ 的非对称 NF）。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention

---
