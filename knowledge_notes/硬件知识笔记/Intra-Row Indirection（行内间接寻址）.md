## Intra-Row Indirection（行内间接寻址）

术语是什么？通过联网搜索让回答具体和精准。
Intra-Row Indirection是AQPIM提出的硬件机制，用于在HBM-PIM bank内高效执行随机（indirect）memory lookup操作。核心思路：将lookup indices存储在GRF (General-Purpose Register File)中→通过MUX将indices路由到column decoder（而非标准row+column地址解码）→column decoder直接从已activated的row buffer中选择对应的inner product values→values stream到BufferPE或GRF。该机制的关键特性：(1) 单次row activation完成所有lookup——因为page-aware windowed clustering确保所有可能被引用的centroids/values在同一DRAM row内；(2) 仅需MUX+少量routing wires，无额外storage——复用现有1KB row buffer和column decoder；(3) 面积仅0.0565mm² per HBM (0.43% of BankPE area)，通过standard PDK [9] synthesis验证。

从硬件架构角度拆解术语：
```
Normal DRAM Read:                     Intra-Row Indirection:
  Address → Row Decoder →                Indices (GRF) →
  Activate Row → Row Buffer →            MUX → Column Decoder →
  Col Decoder → Data Out                 Route selected values from
  (One value per activation)             row buffer → datapath out
                                         (Multiple values per activation)

AQPIM Intra-Row Indirection Datapath (Fig.8):
┌──────────── Bank ────────────┐
│  Row Buffer (1KB, 512 FP16)  │ ← Pre-activated row (inner product values)
│       ↓                      │
│  Column Decoder ← MUX ← GRF │ ← Indices [3,7,2,...] control column select
│       ↓                      │
│  Output stream → BufferPE    │ → Sequential stream of selected FP16 values
│       or → GRF               │
└──────────────────────────────┘

BufferPE接收stream后执行softmax→结果返回到banks
```

术语一般如何实现？如何使用？
Implementation: (1) Column decoder input latched for pipelining → tCCDL (delay between RDs to same bank group) unaffected；(2) 所有DRAM traffic和contentions由standard DRAM controllers管理；(3) 新PIM command PIM_RET (row buffer retrieval)触发intra-row indirection操作。Use case: PQ-based attention的inner product matrix lookup——BankPE计算IPM (1×K, K=512 values) → 存储在同一row buffer → indices来自key/value codebook assignments → intra-row indirection依次select IPM values → BufferPE sum + softmax。对比naive approach (每次lookup一次row activation, 512× overhead)，intra-row indirection仅需1次activation/512 lookups。

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

