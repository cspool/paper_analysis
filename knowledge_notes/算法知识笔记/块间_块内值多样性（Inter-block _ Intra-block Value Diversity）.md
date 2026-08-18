## 块间/块内值多样性（Inter-block / Intra-block Value Diversity）

术语解释
块间值多样性指不同 block（权重/激活矩阵的共享指数分组）之间值分布特征（所需指数范围）不同，单一固定位配置无法兼顾；块内值多样性指同一 block 内元素的值分布差异随 block 增大而放大，迫使更多元素共享同一指数与位配置而丢失细粒度表示。二者是 MXFP 在"极低比特 + 大 block"趋势下精度退化的根源。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：MXFP 用 block 共享指数缓解整张量范围失配，但极低比特下共享指数归一化后 block 间仍残留差异（inter-block diversity）——实验（Llama3-8B layer9 query projection 激活热力图）显示归一化后下层 block（如 Block 25）仍需要更大指数值（配 E2M1 保范围），上层 block（如 Block 679）多数元素只需小指数（可省指数位给尾数用 E1M2 提分辨率）；同时为摊薄 8-bit 指数元数据（Llama3-405B 从 block 32 的 11.7GB 降到 block 256 的 1.46GB）而增大 block size，会让更多元素共享同一指数/配置，放大 block 内值分布差异（intra-block diversity）。两者都导致固定配置（MXFP4=E2M1）表示失配、perplexity 随 block 增大单调上升。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化 pipeline 中的体现：oracle 格式（逐 block 允许任意配置、选 MSE 最小者）在 Llama3-8B 上测得的配置分布（Fig.5）表明激活与权重对指数位数的偏好不同且随位宽变化，单一固定配置（如 MXFP8 的 E5M2/E4M3）与实测偏好错位导致更高 MSE；oracle 在 4-bit 下 perplexity 达 Llama3 8.3/OPT 15.1，而 MXFP4 严重退化（Llama3 30.98/OPT 88.81，WikiText-2）。Oracle-SB（oracle+sub-blocking）在 block 64/256 时 perplexity 8.5/9.0 vs oracle 15.2/44.2，证明 intra-block diversity 由 sub-blocking 解决。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/应对：MXFFP 用"1-bit 配置字段（块级，解 inter-block diversity）+ sub-block 结构（解 intra-block diversity）+ 基于相对指数统计的运行时配置选择（Algorithm 1）"同时处理两类多样性；权重离线按双配置量化选 MSE 小者（=oracle），激活运行时用计数规则近似。使用场景：4-bit 低比特推理与 256 大 block 部署（元数据降 4× 仍保精度），也扩展到 ViT 等非 LLM 负载（ViT-base/large 4-bit Top-1 从 MXFP 的 76.46%/79.54% 恢复到 MXFFP 的 79.36%/81.36%）。

涉及论文标题：
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration
