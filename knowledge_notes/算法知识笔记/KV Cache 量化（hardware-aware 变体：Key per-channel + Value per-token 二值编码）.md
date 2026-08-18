## KV Cache 量化（hardware-aware 变体：Key per-channel + Value per-token 二值编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- KV Cache 量化是把 LLM 推理中存储的 Key/Value 张量从 FP16/BF16 压缩到低比特表示（INT4/INT2 等）以降低内存占用与访存流量的技术。与权重量化的区别：(1) KV 是流式结构——新 token 的 K/V 实时追加，无法用需要离线全局统计的方法；(2) 数值分布随序列动态变化；(3) 量化误差跨层 residual 累积。主流的分布发现（KIVI [46]/KVQuant [29] 同期独立结论，Omni-LUT Fig.3 验证）：Key cache 有强 per-channel 特征——某些通道持续是大幅值 outlier，适合 per-channel 量化；Value cache 分布高度 token 相关、无稳定 per-channel 结构，适合 per-token 量化。Omni-LUT 的 hardware-aware 变体：Key 用离线校准 per-channel BCQ + AS-Bit 位分配，Value 用在线 per-token BC-UQ（TSE 在线求 min/max），量化结果直接是 LUT 可消费的 binary-coding bit-planes，不依赖 outlier 高精度隔离与 dequant。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 流水（一次 decode 步骤）：新 token → QKV 投影得 q,k,v → Key 走 Key Path：查离线校准的 per-channel zp_{k,c}/α_{i,c}（含 AS-Bit 位分配），BEA 贪心编码为 bit-planes；Value 走 Value Path：TSE 求 x_min/x_max → zp_v=(x_max+x_min)/2、δ_v=(x_max−x_min)/(2^b−1) → α_i=δ_v×power-of-2 basis → BEA 编码 → 量化 KV 追加到 cache（KV4=4 bit-plane、KV3=3 bit-plane）→ attention 的 QK^T 与 Attn×V 在 LUT datapath 上按 bit-plane 查表执行（bit-slicing：计算量∝bit-plane 数）。效果：KV3 相对 KV4 的 AA-GEMM 计算少 25%、KV 流量更小；3/4-bit KV vs FPE/FIGLUT 的 16-bit、Tender 的 8-bit，DRAM 能量优势显著（8192 input tokens 下总能量比 FPE/Tender/FIGLUT 低 50%/32%/38%）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现方式：算法侧 PyTorch + HuggingFace transformers，在每层 attention 内模拟 KV 量化；硬件侧 BQU（TSE+BEA，宽 128 匹配 head dim，不增 cycle）。同类公开实现：KIVI（github.com/jy-yuan/KIVI，per-channel Key + per-token Value + 全精度滑动窗口）、KVQuant（per-channel+per-token 非均匀 + outlier 隔离）、QuaRot（head-wise Hadamard 旋转消除 outlier 后简单 per-head asymmetric INT4）。Omni-LUT 对比结果（Table II/III）：KV4-BCQ 平均 PPL 增 0.17、KV3 增 0.75，与 SOTA 相当；AS-Bit 使 Key 有效位宽 4.25 bit，低于 KIVI/KVQuant/Oaken 的 4.8-5.0 bit。论文未开源（联网 2026-08 未找到仓库）。

涉及论文标题：
- Omni-LUT: Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization
