## Distance Polling（距离轮询）

术语解释
<距离轮询是 EgDiff 的预测表压缩与相关性精化算法：每个表项只保留 1 个 stride（diff）+ 1 个 distance 字段，预测/更新时若实际 stride 与预测不符，就递减 distance 探测值队列中更近的位置、重算 stride，逐步收敛到稳定的全局相关性，从而把 gDiff 每项 32 个 diff 的存储压缩 95.8%。>

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：① gDiff 用"每项多个 diff（order-n）+ distance index"覆盖多个可能的全局相关距离，order-32 时 4K 项达 1.03MB，远超几十 KB 的硬件预算；② EgDiff 的洞察：冗余存储——一个条目实际只需要一个真正相关的 (distance, stride) 对，多个 diff 大多数是噪音；③ 距离轮询机制（Algorithm 1）：每个表项存 diff + distance（distance 指示该 stride 在值队列中回溯多远被观察到）；预测时 diff 作用在 distance 处 base 值上；④ 更新时计算实际全局 stride δa = v_actual − NVQ[d].value，与预测 δp 比较：匹配 → 递增 FPC 置信与 usefulness（u）；不匹配 → 重置置信/使用计数器，若 d>1 则 d←d−1（探测更近位置）、重算 δnew = v_actual − NVQ[d].value 并更新 diff；若 d 失效则重置为默认距离 n；⑤ 通过持续轮询，预测器把 (distance, stride) 收敛到"最能代表该指令全局相关性的位置"，同时每次更新至多计算 distance 与 distance-1 两个 diff（gDiff 要算全部 n 个），降低计算开销。
- 效果：4K 项表从 1.03MB（gDiff order-32）降到 44KB（-95.8%）；距离分布与原始多 diff 配置高度相似（Fig.13），且性能不降反升——IPC 从 3.62% 提到 4.37%、覆盖率从 22.27% 提到 25.87%（轮询顺带过滤了弱/噪音相关性）；order 敏感性：order-8 时 2.67%，order-16 后饱和（4.04%+）。
- 表项构成（88 bit）：14-bit tag + 2-bit u + 3-bit FPC + 64-bit diff + 5-bit distance。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件流程（Algorithm 1 伪代码）：给定当前指令实际值 v_actual、非投机值队列 NVQ、预测 stride δp、表项 gdEntry、预测距离 d：① 计算 δa ← v_actual − NVQ[d].value；② 若 δa ≠ δp：重置 gdEntry.counter=0、gdEntry.u=0；若 d>1：d←d−1、gdEntry.distance←d、δnew←v_actual−NVQ[d].value、gdEntry.diff←δnew；否则：gdEntry.distance←n（默认）、δnew←v_actual−NVQ[n].value、gdEntry.diff←δnew；③ 若 δa = δp：若 ctr<MAXCONF 且 forwardProb() 则 ctr++（FPC 概率递增）；若 u<MAXU 则 u++。
- 硬件架构中的角色：轮询发生在"更新阶段"（commit 路径），每 commit 一次 load 至多访问 NVQ 的两个位置（distance 与 distance-1），配合 FPC（3-bit，概率向量 {1,1/4,1/4,1/8,1/8,1/8,1/8}）与 u 位（10-bit TICK 周期性衰减、u=0 才允许替换）实现"只保留最有价值的相关性 + 过滤噪音"。预测表是直接索引 SRAM，无相联查找。
- 与 gDiff 对比：gDiff 更新需对全部 n 个全局值算 diff 再逐一比较；EgDiff 只算 2 个并选 1 个更新——计算与存储同时降低。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在预测表 SRAM 条目中实现 5-bit distance 计数器 + 1 个 64-bit diff 寄存器 + FPC/u 计数逻辑；更新路径按 Algorithm 1 做 distance 递减与 stride 重算；在 gem5 O3 CPU 的 value prediction 框架（src/cpu/valuepred/，VPUnit 的 update()/squash() 接口）中实现。论文未提供公开代码（无法确认）。
- 使用方式（复现）：对比"全 diff 存储（gDiff order-32）"与"距离轮询（EgDiff）"两配置的 distance 分布（论文 Fig.13，用模拟器统计预测时使用的 distance 分布）与 IPC/覆盖率；扫描 order（8–64）与 stride 宽度（8–64 bit）看收敛与饱和点。论文未明确说明 distance 分布的抽取方式。

涉及论文标题：
- Revisiting Global Value Prediction: A Resurgent Complement to Local Predictors
