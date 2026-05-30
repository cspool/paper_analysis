## Fourier Transform-based Adapters (FT-based Adapters / 傅里叶变换基适配器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fourier Transform-based Adapters（FT-based Adapters）是一类 PEFT 适配器，将权重更新 ΔW 表示为稀疏系数矩阵 F 在频域（变换域）中的表示：F = H' ΔW H ⟹ ΔW = H'^{-1} F H^{-1}，其中 H 和 H' 是预定义的固定正交变换核。F 为稀疏矩阵（仅 p 个可训练非零元素），稀疏位置由参数选择策略决定。与 LoRA（ΔW = BA，rank ≤ r）相比，FT-based adapters 的关键优势是"全秩表示能力"：只要 F 的每行每列平均有 ≥2 个非零元，F 就以高概率满秩（rank = min(d_in, d_out)），远超 LoRA 的秩瓶颈。已知的 FT-based adapter 变体包括：(1) FourierFT (DFT kernel)，(2) LoCA/DCA (DCT kernel)，(3) SSH/DHA (DHT kernel)，(4) QWHA/WHA (WHT kernel)。QWHA 论文中的 WHA（WHT-based Adapter）采用单变换设计 ΔW = F H^{-1}（而非双变换 H'^{-1} F H^{-1}），因为在量化场景中输出通道间独立，双变换不提升表示力，反而增加计算开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FT-based Adapter 的核心机制——从稀疏系数到权重更新的展开：

```
# 定义
F = Scatter(c, E)  ∈ R^{d_out × d_in}
# c ∈ R^p: 可训练系数值向量
# E ∈ R^{p × 2}: 非零元素索引列表（固定或可训练）
# Scatter: F[E[l,0], E[l,1]] = c[l], 其余为0

# WHA (QWHA - 单变换):
ΔW = F @ H^{-1}
# H: WHT 矩阵, d_in × d_in, 仅 ±1/√(d_in)
# 计算复杂度: O(p·d_in) sparse-dense + O(d_in log d_in) fast WHT

# DCA/DHA (LoCA/SSH - 双变换):
ΔW = H'^{-1} @ F @ H^{-1}
# H' 和 H: DCT/DHT kernel (包含正弦/余弦计算)
# 计算复杂度: O(p·(d_in+d_out)) + O(d_in log d_in) + O(d_out log d_out)
# 双变换开销显著，训练时间约 WHA 的 3-10x（batch=4: 6.0h vs 26.1h/30.1h）

# 秩分析 (对于随机选择的 F):
# F 的每行非零元平均数: k = p/d_in > r
# F 的每列非零元平均数: l = p/d_out > r
# 当 k,l ≥ 2 时, rank(F) → min(d_in, d_out) 以高概率
# LoRA 对比: rank(BA) ≤ r << min(d_in, d_out)
```

参数选择策略对比（FT-based adapters 必须选定 E——哪些系数位置参与微调）：
- Random：纯随机选择位置，初始化 c=0（FourierFT 原始方案）
- SSH：50% 幅值最大位置 + 50% 随机（假设预训练和微调权重的频谱模式相似）
- LoCA：随机初始化位置 + 微调过程中通过重参数化更新 E（训练开销约 ×2-3）
- AdaAlloc (QWHA)：通道级自适应分配 + 通道内幅值选择 + Refinement（量化感知）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FT-based adapters 的实现要点：(1) 变换矩阵 H 预计算并跨层缓存，不同维度用不同大小的 H；(2) 前向传播中 F @ H^{-1} 的计算可通过 (i) 先计算 F @ H^{-1} 展开为稠密 ΔW，或 (ii) 推理时先对激活做 H^{-1} X 再与稀疏 F 相乘（更快）；(3) 反向传播仅更新 c（F 的非零值），E 通常固定（LoCA 例外，E 通过重参数化可训练）；(4) WHT 相比 DCT/DHT 的计算优势：仅 ±1 元素 → 仅用加减法 → 训练时间与 LoRA 相当，而 DCT/DHT 需复数/三角函数计算。适用场景：需要高表示力（超越 LoRA rank 限制）且可接受少量额外推理开销的场景。在 QA-PEFT 中，FT-based adapter 必须配合量化感知初始化（非随机/零初始化）才能发挥优势，否则效果不如 LoRA-based QA-PEFT。

涉及论文标题：
- QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning
