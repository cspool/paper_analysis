## AW-GEMM 与 AA-GEMM（activation-weight GEMM 与 activation-activation GEMM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 按 GEMM 两个操作数来源分类的 LLM 推理 GEMM 类型：(1) AW-GEMM（activation-weight GEMM）——一个操作数是模型权重（离线已知）、另一个是运行时激活，出现在 linear 层（QKV 投影、attention output 投影、FFN）；权重量化后成为 FP-INT 混合精度 GEMM（mpGEMM）。现有 LUT-based GEMM 加速器（FIGLUT [52]、LUT Tensor Core [50]）主要针对 AW-GEMM。(2) AA-GEMM（activation-activation GEMM）——两个操作数都是运行时激活，出现在 attention 的 QK^T 与 Attn×V（操作数是缓存的 Key/Value）。复杂度随上下文长度不同：prefill 中 AW-GEMM 线性层 O(T)、AA-GEMM 因 token 对 QK^T 为 O(T²)；decode 中每个新 token 的 AW-GEMM 不随上下文增长，而 attention 要读全量 KV cache 做 QK^T 与 Attn×V，AA-GEMM 计算与 KV 流量都随上下文增长。因此长上下文下 AA-GEMM 是计算与能量主导项，高效 LUT 执行必须覆盖 AA-GEMM。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 一次长上下文 decode 的 GEMM 序列：token embedding → AW-GEMM（QKV 投影，W4A16）→ AA-GEMM（QK^T：Query 与缓存 Key 的点积，W=0 权重参与）→ softmax → AA-GEMM（Attn×V：attention score 与缓存 Value 加权）→ AW-GEMM（output 投影）→ FFN 的 AW-GEMM。量化形式：AW-GEMM 中权重可离线量化（低 bit 权重×FP16 激活）；AA-GEMM 中 Key/Value 是运行时激活，必须 KV cache 量化（Omni-LUT：Key per-channel BCQ + Value per-token BC-UQ，都转成 binary-coding bit-planes）。执行例子（Omni-LUT LUT datapath）：QK^T 的 K 以 4 个 bit-plane 存储，PE 每 cycle 按 4 激活组查表、32 个量化 K 值并行 RAC，查表次数∝bit-plane 数（KV3 比 KV4 少 25% 计算）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现方式：商业加速器（GPU/TPU）不原生支持 FP-INT mpGEMM，通常先 dequant 权重到高精度再做 GEMM（低效）；LUT-based 加速器把"高精度激活×低比特权重"的部分积预计算进 LUT，查表+累加替代乘法。FIGLUT 只对 AW-GEMM 用 LUT、AA-GEMM 回退 FP systolic 后端（面积 0.39→1.03mm²）；Omni-LUT 通过 scale-aware LGU（row-wise 缩放内嵌查表）+ BQU（KV 在线量化）让 AA-GEMM 也留在 LUT datapath，等峰值吞吐下能效比 FIGLUT 高 1.25×-1.91×。对比指标（Table VI，OPT-6.7B 8192/512）：有效 GEMM TOPS Omni-LUT-KV4 1.78 vs FPE 0.76/Tender 0.75/FIGLUT 0.96。

涉及论文标题：
- Omni-LUT: Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization
