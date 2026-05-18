## BankPE / BufferPE Dual Architecture for PQ（面向PQ的BankPE/BufferPE双计算单元架构）

术语是什么？通过联网搜索让回答具体和精准。
BankPE和BufferPE是AQPIM在HBM-PIM架构中设计的两种Processing Element，根据PQ和attention操作的计算特征和物理位置约束进行分工。BankPE位于DRAM die的每个bank旁——紧邻DRAM arrays，享有极高的internal bandwidth (TB/s级)但面临严格面积约束（额外area直接损害memory density）。BufferPE位于HBM buffer die——带宽较低但面积约束宽松（buffer die用更先进逻辑工艺），适合容纳面积大或需要跨bank数据聚合的计算单元。两者均为AttAcc! PE设计的复用和优化。核心分工原则：non-data-intensive操作（单个bank内独立完成，无需跨bank通信）→ BankPE；data-intensive或需要跨bank aggregation的操作→ BufferPE。

从硬件架构角度拆解术语：
```
AQPIM Operation-Placement Mapping (Table I):
┌──────────────────────┬──────────┬─────────────────────┐
│ Operation            │ Location │ Required Units       │
├──────────────────────┼──────────┼─────────────────────┤
│ Distance Calc (DC)   │ BankPE   │ ADD, MUL, SUM       │
│ Cluster Assign (CA)  │ BufferPE │ MIN (argmin)        │
│ Centroid Calc (CC)   │ Both     │ MUL,SUM(Bank)+DIV(Buf)│
│ Attention K (ATNK)   │ BankPE   │ MUL, SUM            │
│ Softmax (SFM)        │ BufferPE │ ADD,SUM,MAX,DIV,EXP │
│ Attention V (ATNV)   │ BankPE   │ MUL, SUM            │
└──────────────────────┴──────────┴─────────────────────┘

Codebook Generation Dataflow (blue arrows, Fig.7):
  GPU→BankPE(DC)→BufferPE(CA, MIN)→BankPE(CC numerator)
                                          ↕
                                  BufferPE(CC reciprocal)

PQ Attention Dataflow (red arrows, Fig.7):
  GPU→BankPE(ATNK: query×codebook)→BufferPE(SFM: softmax)
      ↑                                  ↓
      └──── BankPE(ATNV: attn weights×value codebook) ←──┘
```

术语一般如何实现？如何使用？
BankPE: 复用AttAcc! BankPE设计，保持FP16 ADD/MUL/SUM units，不添加任何quantization-specific ALU (如INT MAC)。BufferPE: 复用AttAcc! BufferPE的accumulators和softmax units (含DIV/EXP calculators)，额外承担MIN unit for cluster assignment和reciprocal computation for weighted k-means denominator。CC操作的分布式实现——BankPE计算weighted sum (numerator)，BufferPE计算sum of weights倒数的reciprocal→传回BankPE做单次multiplication完成division。Area: BankPE unchanged, intra-row indirection +0.43% per HBM; BufferPE unchanged (already has required units from AttAcc! design)。

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

