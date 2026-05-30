## LogART Arithmetic Element (AE, 对数量化算术单元)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LogART AE 是 LOGART 提出的针对 DLog 量化权重与激活乘法的无乘法器算术单元（Arithmetic Element）。核心架构（Figure 4(e)）：输入为量化 weight code Q_W (4-bit)、激活 X (8-bit)、控制信号 n₁ 和 chk_even；Decoder 是组合逻辑，判断每个元素使用 base-2 还是 base-√2，生成 Approx 模块使能信号和 Shift 模块移位位数；Approx 模块执行 SDE 展开的 shift-add（处理 base-√2 的 √2 因子）；Shift 模块处理 base-2 部分的移位；Adder tree 累加所有 partial sum。与 BRECQ AE（需 8×8 INT 乘法器，95.8 µm² / 6.28 µW）和 AdaLog AE（LUT+multiplier+shifter, 76.2 µm² / 5.56 µW）相比，LogART AE 面积仅 53.2 µm²，功耗 3.45 µW（28nm UMC, 250 MHz, 0.9V）。

从硬件架构角度拆解术语：
```
LogART AE 数据路径 (4-bit weight × 8-bit activation):
┌──────────────────────────────────────────────┐
│ Input: Q_W[3:0], X[7:0], n1, chk_even       │
├──────────────────────────────────────────────┤
│ Decoder (combinational logic):               │
│   region = Q_W < n1 ? "base-sqrt2" : "base-2"│
│   base-2: shift_amt = Q_W - n1               │
│   base-sqrt2: idx = Q_W // 2                 │
│     shift_amt = idx - n1/2 + floor(m-n1)/2  │
│     enable_approx = (Q_W mod 2 == 1)         │
├──────────────────────────────────────────────┤
│ Approx Module (when enable_approx=1):        │
│   approx_X = X + (X >> 1)  (K=2 SDE)        │
│        or   X + (X >> 1) + (X >> 5) (K=3)   │
├──────────────────────────────────────────────┤
│ Shift Module:                                │
│   base-2: out = X << shift_amt  (or >>)      │
│   base-sqrt2: out = approx_X << shift_amt    │
├──────────────────────────────────────────────┤
│ Adder Tree: Σ partial products → Y           │
└──────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？
AE 设计使用 Synopsys Design Compiler 在 28nm UMC 标准单元库上综合（250MHz, 0.9V）。面积/功耗报告由 DC 生成。LogART AE 的关键节约：完全消除乘法器阵列（通常占 MAC 单元 >70% 面积），用 barrel shifter 和少量加法器替代。Decoder 为简单组合逻辑（比较器+减法器），Approx 模块为固定 K 的移位加。推理时，量化后模型直接使用 LogART AE 做矩阵乘法，无需解量化到 FP16 再乘。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION
