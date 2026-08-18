## Range Identifier（RID，范围标识符）与 RangeMap 范围映射

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RID 是 RangeGuard（ISCA 2026，SKKU IRIS Lab，https://iris.skku.edu/publication/c97_isca_2026/）提出的元数据中心纠错框架的核心概念：把每个数值的数值域划分为少量预定义"范围"（range），每个值映射到其所在范围的索引（如 4-bit RID 表达 16 个范围、8-bit RID 表达 256 个范围）。RID 不存储原始数据——写路径只把 RID 的 ECC 冗余存下来、原始数据照常存；读路径从取回数据重新生成候选 RID。逻辑链：DNN/LLM 的实际崩溃来自极少数 exponent 位翻转制造的天文数字 outlier（BF16 e[7] 0→1 翻转 ×2^128≈3×10^38），而不是 mantissa 低阶位的微小扰动 → 与其保护"原始比特"（恢复一个 32-bit 值至少需 64 bit 冗余），不如保护"值落在哪个范围"这个语义元数据（恢复 4-bit RID 只需 8 bit 冗余）→ 把稀缺的 16-bit parity 预算花在"改变范围"（inter-range）的错误上，范围内（intra-range）扰动直接放行。RID 概念上类似量化 level，但关键区别是：范围只作为"恢复目标"，无错值仍以全精度存储和使用，只有被纠错的损坏值才"吸附"到范围代表值——这是"有界近似纠错"（bounded approximate correction）的核心。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RangeMap 构造（简单映射，论文 §V-B）的算法流程（BF16，K 个范围）：
```
输入：值分布（零均值高斯，标准差 σ）、范围数 K、exponent 域 E={0..255}
1. 求 exponent 概率质量：P(e=k) = 2[Φ(2^(k+1-127)/σ) − Φ(2^(k-127)/σ)]   // Φ=标准正态 CDF
2. 定义 scale 函数：f(e) = 2^(e-127)
3. 把 E 划分为 K 个连续区间 {[l_k, r_k]}，每区间赋代表 exponent ê_k
4. 最小化 MAE：L = Σ_k Σ_{e=l_k..r_k} P(e)·|f(e) − f(ê_k)|     // 全局最优表
```
例子（σ=4 的 4-entry 表，Table II）：区间 [0,127]→代表值 0.5、{128}→2、{129}→4、[130,255]→8；σ=4 使 ±3σ≈±12 覆盖多数 LLM 激活。理想映射用 L1 最优标量量化（Lloyd-Max 类）：归一化高斯阈值 (−0.8217, 0, 0.8217)、代表值 (−1.2657, −0.3778, 0.3778, 1.2657)，可缩放复用到 FP32/BF16/INT8。执行例子（8 个 FP32 值、RG 4b DSC）：写时每值查 RangeMap 得 4-bit RID → 8 个 RID 进 RS(12,8) 编码生成 16-bit parity；读时重新生成候选 RID → RS 解码纠正 ≤2 个错误 RID → 被纠值替换为该范围代表值 → 误差上界 = 范围宽度。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件侧为 flip-flop 实现的 RangeMap（每 32B 访问把 16 个 16-bit 值的 exponent 与 16 个 8-bit 表项并行比较，256 个 8-bit comparator，1 cycle 完成 RID 提取）；多格式支持用多个子映射（32/16/8-bit 值分别用 4/2/1-bit RID，同一 32-bit 区域的 RID 打包进一个 ECC 符号）。使用要点：范围宽则浪费 RID 空间、范围窄则 MAE 收益边际；σ 过大/过小都使保护失效（敏感性实验：最优 σ 下 Llama-3.2 在 BER=10^-6 保持精度，偏离则骤降）；全局映射（σ=4、4-bit RID）已足够，tensor 级 Lloyd-Max 映射精度增益有限且面积开销大。效果：16-bit parity 预算下每 256-bit block 容忍 64+ 个翻转数据 bit（8× bit 级方案）。
涉及论文标题：
- RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs
