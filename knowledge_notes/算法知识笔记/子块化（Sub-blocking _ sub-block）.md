## 子块化（Sub-blocking / sub-block）

术语解释
把大 block（如 256 元素）内部划分为更小的 sub-block（如 32 元素），每个 sub-block 拥有独立的 1-bit 位配置字段（8 个 sub-block 配置位合成 1B configuration set 保持字节寻址），但整个大 block 共享单一 8-bit 指数——以"配置粒度细、缩放粒度粗"的解耦结构同时获得低元数据与高精度。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：大 block 降低共享指数元数据开销但放大块内值多样性、损害精度（oracle 在 block 256 时 perplexity 44.2 vs block 32 的 15.2）；→ 把 block 拆成 32 元素 sub-block，每个 sub-block 独立选 E1Mx/E2Mx 配置，就能在 block 内继续捕捉值分布差异（intra-block optimization）；→ 但多个 sub-block 若各自存指数会放大元数据，故所有 sub-block 共享一个按 E2Mx 假设计算的指数，E1Mx 的 sub-block 使用时减 1 补偿偏置差——"配置位 per sub-block（1 bit）+ 指数 per block（8 bit）"解耦，元数据仅 1 bit/sub-block。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化 pipeline 中的流程（block 256 = 8 sub-block × 32）：
```
对每个大 block:  E_b^shared = 以 E2Mx 假设计算的统一指数
对每个 sub-block s (32 元素):
  相对指数 E_i^r = E_i - E_b^MAX(s)
  count_E1/count_E2 统计 → cfg_s ∈ {E1M2, E2M1}
  实际缩放: cfg_s=E1Mx 时用 E_b^shared - 1, E2Mx 时用 E_b^shared
```
结果：Llama3-8B 4-bit 下 MXFFP block 256（sub-block 32）perplexity 24.3，仍低于 MXFP block 32 的水平（≈30.98），且 sub-block 4 时平均 perplexity 退化仅 0.98、比 MXFP8/MXFP6 内存需求低 47.8%/29.1%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：内存布局上 configuration set（8 个 sub-block 配置位 → 1B）与共享指数并排存放；硬件上每个 threadgroup 处理一个 sub-block（映射到 4×8 子矩阵、4 step 执行），转换阶段先跨 threadgroup 同步全局 E_b^MAX（Max 单元），再逐 sub-block 用 Config Selector 选配置，Normalization & Round 按统一指数输出。使用：需要大 block（256）摊薄指数元数据、同时保持 4-bit 精度的推理部署。

涉及论文标题：
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration
