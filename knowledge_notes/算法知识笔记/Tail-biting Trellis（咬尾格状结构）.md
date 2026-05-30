## Tail-biting Trellis（咬尾格状结构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tail-biting trellis 是 trellis 编码的一种变体，要求 trellis walk 的起始和结束状态共享 L-kV 个比特位，形成"环状"约束。在 QTIP 中，tail-biting 解决了一个工程问题：直接量化长度 T 的序列到 (L,k,V) trellis 产生 kT + (L-kV) 比特（起始状态需额外 L-kV bits）。当硬件字长为 w（如 32 bits）且 w|kT 时，每序列需读取 ⌈(L-kV)/w⌉w 个浪费比特。精确求解 tail-biting 问题需 O(2^{2L}) 动态规划，对 L≥12 不可行。QTIP 提出 Algorithm 4 近似求解：旋转序列 T/2 步 → Viterbi 量化 → 提取重叠 → 以该重叠作为 tail-biting 约束重新量化原始序列，仅需 2 次 Viterbi 调用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QTIP Algorithm 4（Tail-biting 近似）：
```
输入: S ∈ R^T, (L,k,V) trellis G.
S' ← Rotate S right by ⌊T/2⌋      # 旋转序列
Ŝ' ← Viterbi(S', G)               # 第一遍 Viterbi
O ← L-kV bit overlap of Ŝ'_{⌊T/2⌋} and Ŝ'_{⌊T/2⌋+1}  # 提取重叠
Ŝ ← Viterbi(S, G) with start/end overlap = O  # 以 O 约束的第二遍 Viterbi
输出: tail-biting Ŝ.
```
实验表明该近似对 i.i.d. 高斯源几乎无损：量化 4K 个 T=256 序列时，2-bit (k=2) 的 Algorithm 4 MSE 0.0733 等于最优解 0.0733（Table 2）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Tail-biting 使总编码比特数 = kT（起始状态隐含在结束状态中），当 kT 整除 w 时无浪费比特。在 QTIP 中，T_x=T_y=16, k=2 → kT = 2×16×16 = 512 bits = 16 个 32-bit word，完美对齐。Tail-biting 的近似在 i.i.d. 数据上极准但可能对非 i.i.d. 数据降质；QTIP 依赖 RHT 确保权重的 i.i.d. 特性。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing
