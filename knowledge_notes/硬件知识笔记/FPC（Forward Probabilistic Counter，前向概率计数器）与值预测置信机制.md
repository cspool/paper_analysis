## FPC（Forward Probabilistic Counter，前向概率计数器）与值预测置信机制

术语解释
<FPC 是一种置信度计数器：正确事件以概率（由概率向量控制）前向递增，错误事件立刻清零；高位概率递减以减缓置信累积、避免过早信任弱训练的条目。EgDiff 用它（连同 tag 索引、usefulness bits、last misprediction）构成 aggressive confidence 机制。>

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：① 值预测是投机技术，精度不够（<99%）时误预测 squash 惩罚会吃掉收益——所以置信机制决定"什么时候允许输出预测"；② 传统饱和计数器（saturating counter）达到阈值就放行，训练不足/别名干扰时容易"过早信任"；③ FPC（源自值预测文献 [21] Perais & Seznec HPCA 2014、[35] Yang et al. GLSVLSI 2023）的改进：正确时不一定递增——按概率向量 {1, 1/4, 1/4, 1/8, 1/8, 1/8, 1/8} 递增（位置越高概率越低，即越接近饱和越要"慢一点"），错误时无条件清零重建；这样弱训练条目需要更多正确样本才能达到放行阈值，显著降低误预测率；④ EgDiff 的 3-bit FPC 饱和值=7 才允许输出预测（配合 tag 匹配）。
- aggressive confidence 机制的四个组件（互相补充）：① tag-based 索引：每项 14-bit tag（与索引哈希不同的 PC 哈希），tag 不匹配不发预测，减少 aliasing；② usefulness bits（u）：正确预测递增、10-bit TICK 计数器周期衰减、分配替换只允许 u=0 的项，保留长期有用的条目；③ last misprediction（lastmisp）：误预测后 1024 条指令窗口内全局抑制所有预测输出，防级联错误（与 FPC 的"条目级"清零互补——lastmisp 是"全局级"）；④ FPC：条目级概率置信。消融实验：gDiff+sc 平均 3.24%（偶尔负收益，如 511.povray/531.deepsjeng 过早信任）；+tag 提升一致性（减少 aliasing）；+lastmisp 显著改善最坏情况（抑制连续 squash）；+fpc 达最高平均收益（519.lbm/538.imagick 超 5%）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件流程（EgDiff 预测路径）：① 指令 dispatch，PC 经索引哈希查预测表 → ② tag 匹配？不匹配 → 不发预测（以 1/16 概率分配新条目 u=0）；匹配 → ③ FPC 饱和（3-bit = 7）？未饱和 → 不发预测；饱和 → ④ lastmisp 抑制窗口内？是 → 不发预测；否 → ⑤ 取 diff + distance，访问投机 GVQ 取 base → 生成预测值写 PRF/投机 GVQ。
- 更新路径：① 验证正确（预测值=实际值）→ FPC 以概率向量递增、u 递增（至饱和）；② 验证错误 → FPC 清零、u 清零、置 lastmisp（全局抑制 1024 条指令）、触发 squash；③ TICK 计数器（10-bit）周期性对所有条目的 u 减 1（老化淘汰）。
- 在距离轮询中的角色：Algorithm 1 中 δa=δp 时走"if gdEntry.ctr < MAXCONF then if forwardProb() then ctr++"——FPC 的概率递增直接嵌入更新算法；δa≠δp 时 ctr 与 u 清零（重建信任）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：3-bit 计数器 + 概率控制逻辑（伪随机数/哈希位决定是否递增，概率向量决定各位置阈值）；tag（14-bit）、u（2-bit）、TICK（10-bit）为 SRAM 表项字段；在 gem5 O3 CPU 的 value prediction 框架（VPUnit predict/update 接口）中实现。论文未提供公开代码（无法确认）。
- 使用方式（复现）：对照实验 gDiff+sc → +tag → +lastmisp → +fpc 逐级叠加，统计各配置的平均 IPC 提升、覆盖率（26.15% 到 22.45% 的变化）与误预测率，验证"精度优先、覆盖率让位于准确性"的取舍（fpc 下 554.roms 覆盖下降）。论文未明确说明 forwardProb() 的具体随机源。

涉及论文标题：
- Revisiting Global Value Prediction: A Resurgent Complement to Local Predictors
