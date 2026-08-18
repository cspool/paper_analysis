## Deferred Prediction（延迟预测）

术语解释
<延迟预测是 EgDiff 应对"value delay"的机制：预测时所需的 base 值尚未就绪时，不强求立即预测，而是把预测参数暂存到 GVQ 条目并绑定到 base 指令，等 base 值就绪后反向触发（reverse-triggered）生成预测并唤醒依赖指令。>

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：① 全局值预测的难点之一：预测需要"程序顺序上指定距离处的前序指令输出"，但深流水线里该 base 值常常还没算出来（尤其长延迟 load）——这就是 value delay 问题；② gDiff 的应对：用局部预测器（EVES）的投机结果填充值队列、或用 OoO 引擎的投机结果——但局部预测器低置信时不输出（历史不完整），投机结果又可能被误预测污染；③ EgDiff 的洞察：预测不必发生在"请求时刻"，可以推迟到 base 值可用之时——于是预测信息先写进投机 GVQ 条目，通过 Wake 字段与 base 指令绑定（一对一），base 值就绪即反向触发：定位绑定条目、校验 distance、生成预测值写入 PRF 并广播 wakeup；④ 这既避免前端 stall，又不基于不完整/投机信息过早决策，把误预测风险降到最低。
- 理想化 vs 实用：论文先评估理想模型 gDiff+deferred（base 值就绪时对整个值队列反向扫描、触发所有依赖预测，需 CAM-like 相联结构，硬件贵），再提出实用实现 gDiff+wakeuplinst（每条目单 Wake 绑定）——性能几乎相同（4.07% vs 4.05% IPC）；统计显示 >80% 情况下每条 base 指令只触发 1 个依赖预测、>99% 周期至多 1 次唤醒，单绑定设计代价可忽略。
- 延迟代价：normal 预测 3 cycle；deferred 额外 2 cycle（首周期读 GVQ 条目的 Wake 字段定位绑定条目，次周期访问该依赖条目取 diff 并计算预测值）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件流程（论文 §III-C，表 I 例子）：① 指令 2 dispatch，预测表给出 distance=3、diff=20，但投机 GVQ 条目 5（base）的 Avail=0（不可用）→ 条目 2 写入 distance/diff、VP=0，同时把条目索引"2"写入条目 5 的 Wake 字段；② 指令 5 完成、LQ 更新并写入投机 GVQ → 硬件读条目 5 的 Wake=2 定位条目 2 → 校验条目 2 的 distance 仍等于 3（相对位置未变）→ 用条目 5 的新值 + diff 20 计算预测值 → 写入条目 2 的 value、VP=1 → 写 PRF（用 dispatch 时记录的 PRF 名）并广播 wakeup（复用 load 完成端口与 issue-queue/scoreboard 唤醒协议，无需专用端口）；③ 冲突处理：若同一周期并发 normal load 写回占用端口，deferred 写回延迟 1 cycle（>99% 周期至多 1 次唤醒，冲突罕见）；④ 边界：若条目 2 在条目 5 前完成（Avail=1），唤醒时忽略、不生成预测、跳过验证；绑定阶段目标条目已被占用则丢弃新绑定。
- 作用：把"预测时机"与"请求时机"解耦，让全局相关性在深流水线下仍可用——论文隔离评估显示 deferred 贡献 0.6%（BASE）/0.58%（MID）/0.33%（LARGE）的额外 IPC 提升（LARGE 下约占 EgDiff 总收益的 10%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 GVQ 投机条目中增加 wake 字段 + 一条反向触发路径（base 值写入时按 Wake 索引查依赖条目）+ distance 校验逻辑；唤醒复用现有 load 完成端口的 PRF 写回与 wakeup 广播协议，不新增硬件端口。论文未提供公开代码（无法确认）。
- 使用方式（复现/评估）：在 gem5 O3 CPU 的 value prediction 框架中实现：预测时 base 不可用 → 写 pending 条目 + Wake 绑定；load 完成 → 触发 deferred 预测；对比理想反向扫描模型（gDiff+deferred，CAM-like）与单唤醒模型（gDiff+wakeuplinst）的性能与硬件代价（论文图 11/12 的每指令唤醒数、每周期唤醒数分布）。论文未明确说明如何从 gem5 统计中抽取这些分布。

涉及论文标题：
- Revisiting Global Value Prediction: A Resurgent Complement to Local Predictors
