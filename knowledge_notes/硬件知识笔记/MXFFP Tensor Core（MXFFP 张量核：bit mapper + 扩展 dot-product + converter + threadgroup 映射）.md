## MXFFP Tensor Core（MXFFP 张量核：bit mapper + 扩展 dot-product + converter + threadgroup 映射）

术语解释
MXFFP 论文提出的 Tensor Core 扩展设计：在传统 GPU Tensor Core 低位宽（如 FP4 E2M1）数据通路上做最小改动，用每 block 单 configuration bit 解释执行多 exponent-mantissa 配置（E1Mx/E2Mx）的 MXFFP 格式——bit mapper 按配置位重排操作数、dot-product 核加宽为内部 E2M2、converter 动态重生成 MXFFP block。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：MXFFP 格式支持 E1Mx/E2Mx 双配置，但 GPU Tensor Core 原生只对单一固定格式（如 FP4 E2M1）提供硬件原语（块共享指数 + 格式特化低比特转换）；→ 若为每种配置各建一套算术单元会破坏硬件利用率，因此引入"统一格式"执行：bit mapper（多路选择器阵列）在算术核前按 configuration bit 把 4-bit 操作数重排到统一位布局（E1M2 只转发最高指数位、其余用高 mantissa 位填充；E2M1 直送 2 指数位 + 1 mantissa 位），算术核只需支持一个内部格式 E2M2（E2M1 核 significand 路径 +1 尾数位、normalization 加宽、指数偏置与范围不变）；计算结果按 block 共享指数缩放后累加进高精度（FP16/BF16）accumulator；MXFFP converter（Max→Subtractor→Counter→Config Selector→Normalization&Round）把输出重转回 MXFFP。每 threadgroup 配 configuration 寄存器（2B，block set 一次加载、跨 4 step 保留）与 shared exponent 寄存器（2B，每 step 更新），使 TG 能按 block 逐 step 取配置。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
执行流程（Tensor Core MMA，16×16 输出 tile 拆 8 个 4×8 子矩阵）：每个子矩阵由 threadgroup（TG，4 个 dot-product 单元）经 4 step 迭代累加；MXFFP block 取入 matrix buffer（每 block set 9B 元数据 = 1B 配置位 + 8B 共享指数）；dot-product 单元每周期收 4×1（A）与 1×4（B）fragment 算 4 元素点积，操作数先经 bit mapper 按配置位重排进 E2M2 算术核；乘累加后按 block 指数缩放累加进高精度 accumulator；block set 完成后 converter 重生成 MXFFP block（大 block 时各 TG 先同步局部 E_b^MAX 得全局 E_b^MAX，E1Mx sub-block 的共享指数减 1）。面积/功耗：dot-product 单元面积 +26.4%、converter +4.3%、bit mapper 可忽略，threadgroup 合计 +22.26%（GPU 192 Tensor Core 规模下仅占 750mm² die 的 0.038%）；功耗 +21.4%，配置级 power gating 后有效 ~12%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：论文用 RTL 实现 baseline MXFP Tensor Core 与 MXFFP 增强设计，Synopsys Design Compiler + FreePDK 45nm 综合评估面积/功耗（NVIDIA Tensor Core 微架构不公开，用代表性 4-bit 低位宽数据通路建模，论文自述高估全 Tensor Core 开销）；性能用 Accel-Sim（RTX 5090 派生配置）+ CUTLASS GEMM kernel trace 评估（额外建模 shared exponent/configuration bit 元数据访存）；能量用 AccelWattch。使用：GEMM 延迟评估显示 MXFFP 与 MXFP 几乎相同（配置位无延迟开销），4-bit 1024³ GEMM 相对 BF16 达 2.7×、prefill 端到端 2.08×，使 4-bit 加速在保持精度（perplexity 降低 2-5×）下可被实际利用。

涉及论文标题：
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration
