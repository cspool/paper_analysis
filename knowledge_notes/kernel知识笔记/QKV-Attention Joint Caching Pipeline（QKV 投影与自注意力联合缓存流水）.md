## QKV-Attention Joint Caching Pipeline（QKV 投影与自注意力联合缓存流水）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QKV-Attention Joint Caching Pipeline 是 DIAMoND 提出的跨模块（in-NAND ↔ near-DRAM）算子流水重叠设计：把 QKV 投影与 self-attention 的执行顺序联合优化，利用"Q 投影结果既与新生成的 K 向量、又与历史缓存 K 矩阵相乘"这一依赖结构，把 W_Q 放在 DRAM、W_K/W_V 留在 NAND，使 attention 主体计算与 K/V 投影并行。动机：naive 流程把 Q、K、V 投影全放 in-NAND，则 near-DRAM 等待投影结果时空闲；而 Q/K/V 权重矩阵较小，全部在 in-NAND 算本身也低效（in-NAND 优势在巨大专家矩阵）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
跨模块时间线（每层 self-attention，token t）：
```
near-DRAM:  Q = W_Q @ h_t                 ┐
            S_partial = Softmax(Q K_hist^T)├─ K_hist 已缓存于 DRAM
            (等待最终 K)                   ┘
in-NAND:    K = W_K @ h_t; V = W_V @ h_t  ← 与上面并行
汇合:       attention = S_partial × V     ← 仅 K 到达后补齐尾部
```
伪代码逻辑：naive = 顺序执行 [K,V 投影(NAND) → Q 投影(DRAM) → Softmax → O 投影]；pipeline = [Q 投影与 K_hist 部分 attention 先行(DRAM) ‖ K,V 投影(NAND) → Softmax 收尾 → O 投影]，关键路径只依赖最终 K。效果：self-attention 阶段延迟至多降低 13.5%（DIAMoND 实测）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖两件事：权重布局上 W_Q 驻 near-DRAM、W_K/W_V 驻 in-NAND（部署期决定，见 Mask-Based In-NAND Computing 条目）；运行时由 AES/mask 硬件按序发出 in-NAND 的 K/V read cycle 与 near-DRAM 的 attention 计算，二者经 2.5D 封装的独立 SSD 通道无冲突并行。使用方式：任何"小矩阵投影 + 大缓存 attention"的异构存储计算系统都可复用该重叠思想（把依赖后置的算子提前、与不相关投影重叠）。相关概念：kernel pipeline overlap（共享内存双缓冲、bubble-free pipelining 见本库其他条目），区别是本流水跨两个物理模块（存储计算与逻辑计算），而非同核内多级流水。

涉及论文标题：
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture
