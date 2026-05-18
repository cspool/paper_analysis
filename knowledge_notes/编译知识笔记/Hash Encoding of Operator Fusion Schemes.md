## Hash Encoding of Operator Fusion Schemes

术语是什么？通过联网搜索让回答具体和精准。

Hash Encoding of Operator Fusion Schemes是STOF提出的将抽象operator fusion pattern量化为searchable binary numerical expression的方法。灵感来源于数字电路的高低电压电平表示——将DL computational graph中每个operator映射为一个number（binary 0或1），相同number的相邻operator属于同一fused segment，不同number表示fusion boundary。编码过程：首先通过neural hashing（convolutional subgraph analysis, F_hash∘F_conv(G)）从computational graph G中发现频繁出现的经典子图结构，然后用predefined rules从识别到的子图结构中提取潜在高性能fusion scheme作为initial scheme，最后用hash encoding将scheme转为binary array（可进一步压缩为hexadecimal格式）。此编码的关键性质：1) bidirectional mapping——同一binary array既表示fusion pattern又可通过numerical decoding映射到compilation template; 2) 任意scheme可表示——binary array长度=operator数量，每个位置可独立设值，search space覆盖所有可能fusion组合; 3) search-friendly——binary/hex格式可直接输入search engine做expansion和parameter tuning。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

STOF中hash encoding的工作流程（以BERT forward propagation为例）：

```
// ===== Step 1: Neural Hashing for Subgraph Discovery =====
// Input: PyTorch FX computational graph G
//   Operators: GEMM(#0) Add(#1) GEMM(#2) Scale(#3) Mask(#4) 
//              Softmax(#5) GEMM(#6) GEMM(#7) Add(#8) Layernorm(#9)
//              GEMM(#10) GEMM(#11) GEMM(#12) Add(#13) Layernorm(#14)

// H(G) = F_hash(F_conv(G))
// F_conv: convolutional feature extractor 
//   → 提取local graph structural features 
//     (operator type, input/output edges, dependency chains)
// F_hash: hash mapper
//   → 将extracted features压缩/discretize为unique hash fingerprint
//   → 频率分析: 识别频繁出现的经典子图结构
//     e.g., [GEMM→Scale→Mask→Softmax→GEMM] 频繁出现 → MHA pattern
//           [GEMM→Add→Layernorm] 频繁出现 → CI+MI fusion pattern

// ===== Step 2: Predefined Rules for Initial Scheme =====
// Based on motivation analysis (Section 3):
// - 小batch/short seq: GEMM chain优先fuse为1 segment
// - CI operator最多2个/segment (GPU resource constraint)
// - MI-only chains可自由合并
//
// Initial scheme (binary array):
// [#0 #1 #2 #3 #4 #5 #6] [#7 #8 #9] [#10 #11 #12] [#13 #14]
//   0  1  0  0  0  0  0    1  1  1    0   0   0     1   1
//   ↑                   ↑  ↑         ↑  ↑           ↑
//   MHA (fused kernel)    CI+MI       CI+CI        MI+MI
//   segment 0             segment 1   segment 2    segment 3

// ===== Step 3: Hash Encoding Properties =====
// - Numbers无operator semantics含义——仅用于标记segment boundary
// - 同一segment内所有operator share same number
// - Adjacent segments自动有不同numbers（boundary定义）
// - Hexadecimal compression: binary "00111000..." → hex "0x38A7..."
//   高压缩率，便于caching和comparison
```

与已有fusion compilation approach的对比：
- AStitch/Welder: rule-driven fusion categories (MI-only, CI+MI) → 固定边界
- Chimera/MCFuser: loop-based construction, CI chain最多到一定长度 → 搜索空间受限
- STOF: hash encoding → arbitrary fusion scheme可表示 → 搜索空间完整覆盖

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

STOF以Python实现（PyTorch FX graph manipulation）。neural hashing的F_conv使用轻量GNN（类似GraphSAGE的message passing）提取每个operator节点的local neighborhood特征；F_hash用locality-sensitive hashing (LSH)将continuous feature vector离散化为hash code。predefined rules基于Section 3的motivation实验发现（不同tensor dimensions下fusion效果的experimental规律）。整个hash encoding→numerical decoding pipeline在每次inference tuning时执行一次（per input scale），产生的fusion scheme在相同input scale下可复用（通过cache机制）。与torch.compile的集成：STOF通过操纵fx.GraphModule对象实现graph capture和node replacement，整体兼容torch.compile接口。

涉及论文标题：
- Accelerating Sparse Transformer Inference on GPU
