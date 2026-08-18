## Multi-Point Temporal Triggers（多点时序触发：FOE/SOE/TOE）

术语解释
STEP 的核心组织原则：把空间足迹预取从"单一固定时间点的一次性决策"重构为"页生命周期内多个时序决策点（FOE/SOE/TOE）的分阶段决策"，每个点按当前累积证据决定是否发预取、否则等待下一触发点。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 传统空间足迹预取器几乎都在单一固定时间点触发（首访问 FO 或后续某点如 Gaze 的第二访问点），把触发时机、精度、存储锁死在固定 trade-off：早触发机会多但上下文少（需更 rich key/更大表维持精度、易污染），晚触发证据足更准但错过早期机会。STEP 观察每个区域三个触发点，因用 offset 作事件键命名为：FOE（first-offset event，仅第一 offset + hashed PC 消歧 + maturity 检查）、SOE（second-offset event，前两 offset）、TOE（third-offset event，满三 offset 的精确 tag）。上下文随时间累积：FOE 1 个 offset → SOE 2 个 → TOE 3 个；每点用 Prefetch-Confidence Evaluator 判断"当前证据是否已够"——够则立即发（早机会），不够则等下一触发点（晚精度）。关键观察：TOE 蕴含 SOE 与 FOE，故单一 PHT 即可（按事件截断 tag），不复制三套历史表。本地证据：主文件（score 1405）、`III.-THE-DESIGN-OF-STEP-PREFETCHER.md`（score 470）、`E.-Ablation-Study.md`（score 452）。
- 消融证据（Fig.12/13）：STEP-D1（禁 FOE）主要伤 SPEC CPU2006（早期机会直接转 IPC）；STEP-D2（禁 SOE）平均最优故为最终配置，但 cactuBSSN-2421/4004、sphinx3-234、roms-294 需要中间触发点；STEP-D3（禁 TOE）伤精度（SPEC17/CloudSuite 更甚）。结论：无单一固定触发点普遍最优——正是多触发动机。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件数据流（D 章）：FT 转发请求（offsets + PC + event ID）给 PHT → PHT 按事件 ID 提取对应数据：FOE 用 FO+PC 取最近 N 个匹配（单匹配查 maturity）；SOE 用 FO+SO 查 tag 上 6 位；TOE 用全 tag 查找（命中即下发、未命中本页无动作）。三步各自独立可被 Evaluator 判定收敛后提前下发，形成"早下发或推迟"的决策序列。
- 案例（E 章）：mcf-192 中单 PC 在短时间窗内快速跨大量页、FT 条目常在其二次访问前被逐出——FOE 在页上下文消失前行动（hashed PC 轻量消歧），这是晚触发 baseline 无法捕获的机会；mcf-484 region 0xfe3 中 SOE 时三个候选 C0/C1/C2 无法区分，TOE 精确消除歧义，避免选错候选的 30/12 条无用预取与 11 条漏取。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：FT 每条目记前两个 offset + issued 标志（1 bit）+ hashed PC（12 bit）+ 64 态第二 offset 无效哨兵；PHT tag 由第 2/3 offset 构成，SOE/TOE 按位截断复用同一表；事件 ID 驱动 PHT 查表分支。使用场景：任意缓存层次（论文验证 L2 与 L1 级、L1+L2 多级组合），与 richer 事件表示/候选聚合正交可叠加（论文强调是正交设计维度，可推广到其他空间足迹预取器家族）。

涉及论文标题：
- STEP: Spatial Footprint Prefetcher with Multi-Point Temporal Triggers
