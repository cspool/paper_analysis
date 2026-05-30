## Signed Absolute Block Maximum Normalization（带符号绝对块最大值归一化 / BOF4-S 归一化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Signed Absolute Block Maximum Normalization 是 BOF4 论文提出的归一化改进。传统 absmax normalization 除以 `max_i |w_{b,i}|`（恒正），归一化后权重在 [-1,1]，两端各有 1/(2I) 离散概率质量。Signed 版本改用 `w_b^max = w_{b, argmax_i|w_{b,i}|}`（带符号值），归一化后仅在 x=1 有 1/I 离散概率质量。这使得量化码本只需固定一个端点（x̂(16)=1）而非两个（-1 和 1），释放一个 reconstruction level 降低整体量化误差。signed normalization 与 double quantization 组合需额外 1 bit 编码量化常数符号。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Standard absmax (NF4, AF4, BOF4):
w_max = max(abs(block))                # always positive
x = block / w_max                       # masses at both -1 (1/2I) and +1 (1/2I)
# Codebook fixes BOTH x̂(1)=-1, x̂(16)=1 → wastes one level

# Signed absmax (BOF4-S):
j_star = argmax(abs(block))
w_max = block[j_star]                  # keeps the SIGN
x = block / w_max                       # mass at +1 only (1/I)
# Codebook fixes only x̂(16)=1 → one extra free level
```
假设 block 中最大绝对值权重为 -0.5。Standard：除以 0.5，该权重映射到 x=-1；Signed：除以 -0.5，该权重映射到 x=1。均精确保存最大权重（BF16 精度），但 signed 版本归一化后分布更紧凑。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 https://github.com/ifnspaml/bof4 中实现。关键：(1) 归一化使用 signed max；(2) Lloyd 训练码本时固定 2 个 level（0 和 1）而非 3 个（-1, 0, 1）；(3) 最外层 centroid 更新中概率质量从 1/(2I) 变为 1/I（论文 Eq. 40 vs 39）。BOF4-S 在所有 block size 和误差指标下均优于 BOF4（non-signed）。

涉及论文标题：
- Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4)

---
