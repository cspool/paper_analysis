## 运行时量化转换与 oracle 配置选择（Runtime Conversion / Oracle Format，Algorithm 1）

术语解释
把 FP16/BF16 张量转换为低比特格式的过程。MXFFP 中权重为静态数据走离线 oracle 转换（两种配置都量化、选 MSE 小者，零运行时开销）；激活是运行时数据，用"相对指数统计计数"的轻量规则（Algorithm 1）近似 oracle 的每 block 最优配置选择，把逐 block 配置决策降到几次计数 + 1 次比较的硬件代价。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：MXFFP 支持多配置，但激活在推理运行时才出现、逐配置试量化会引入显著转换开销；→ 观察相对指数 E_i^r=E_i−E_b^MAX（≤0）直接决定哪种配置表示更精细（表 I：E1M2 偏好 E_i^r=0、E2M1 偏好 E_i^r∈{−2,−3}）；→ 因此只需统计 block 内 E_i^r=0 的个数 count_E1 与 E_i^r∈{−2,−3} 的个数 count_E2，用二次加权规则 count_E1²>count_E2 决策（大值元素主导数值保真度）；→ 选完配置再算共享指数 E_b^shared=E_b^MAX−E_element^MAX，逐元素量化输出。该规则使激活配置选择与 oracle 高度吻合（Fig.20a），最终输出 MSE 几乎等于 oracle（Fig.20b），端到端零额外延迟。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 1（MXFFP4 运行时转换，输入 block x={x_1..x_N}）：
```
E_i ← exponent(x_i);  E_b^MAX ← max_i E_i
E_i^r ← E_i − E_b^MAX
count_E1 ← |{E_i^r = 0}|;  count_E2 ← |{E_i^r ∈ {−2,−3}}|
if count_E1² > count_E2:  cfg ← E1M2, E_element^MAX ← 1
else:                     cfg ← E2M1, E_element^MAX ← 2
E_b^shared ← E_b^MAX − E_element^MAX
x̂_i ← quant(x_i / 2^{E_b^shared})
```
大 block 时 sub-block 各自决策配置，但统一指数按 E2Mx 计算、E1Mx sub-block 使用时减 1，复用同一硬件路径。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件上对应 MXFFP converter 流水线——Max 单元取 E_b^MAX、Subtractor 算相对指数、Counter 统计 E_i^r 分布、Config Selector 按二次加权规则决策、Normalization & Round 输出量化 block（Max 与 Normalization&Round 在 baseline MXFP 已存在，只需小幅扩展）；软件上权重离线走 oracle（双配置量化选 MSE）。使用：LLM 推理中所有 MMA 相关张量转 MX 格式的转换路径，覆盖 4/6/8-bit，也推广到 ViT 等非 LLM 负载。

涉及论文标题：
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration
