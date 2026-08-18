## EVES 值预测器（EVTAGE + ES，局部值预测 SOTA）

术语解释
<EVES 是 André Seznec 在 2018 首届 Championship Value Prediction（CVP-1）提出的混合局部值预测器，由 EVTAGE（增强型上下文预测器，取 PC+分支历史）与 ES（增强型 stride 计算预测器，只取 PC）两部分组成，EVTAGE 权威高于 ES。>

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：① EVES 是"上下文型 + 计算型"两类预测器的融合：EVTAGE 是 TAGE 式（带 tag 的几何历史长度）上下文预测器，用 PC 与全局投机分支历史索引，预测值是上下文相关的；ES 是 stride/计算预测器，对 PC 的局部值序列做 stride 计算（前一值 + stride）；② EVTAGE 预测的是"指向专用值表（VTABLE）的指针"，让相同值共享表项、降低存储；③ 置信控制用概率计数器（FPC）而非饱和计数器，同宽度下精度更高；④ 预测表为带 tag 的 2-way 组相联结构、按指令类型概率性分配新表项。
- 网页佐证：EVES 在 CVP-1（2018）全部赛道夺冠，至今仍保持 8KB 与 32KB 存储预算赛道的最高 IPC speedup 记录，被视为 SOTA 开源值预测器；CVP 2018 配置约 31.36KB 开销，CVPv6 + SPEC06/17 上实现 11.2% IPC 提升、18.7% 预测覆盖率（web: 电子学报综述；Seznec, "Exploring value prediction with the eves predictor", CVP-1 2018）。
- 在本论文中的角色：作为"局部值预测 baseline 与 SOTA 对照"——EVES 8KB/32KB/unlimited 三档评估，并作为 hybrid 的局部侧（EgDiff 的全局侧与 EVES 部分正交互补）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件工作流程：EVES 在前端/rename 阶段工作：① 指令 dispatch → EVTAGE 以 PC+投机全局分支历史查表（3 cycle 延迟 = VTAGE 表查找 + 条目处理 + 值表访问），命中且置信足够则输出预测值；② 若 EVTAGE 未输出，ES 以 PC 查 stride 预测器，用最近值 + stride 计算预测值；③ EVTAGE 输出优先（权威更高）；④ 预测值写入 PRF 唤醒依赖指令，执行后验证，正确则递增置信（FPC 概率递增），错误则重置/触发 squash 恢复；⑤ 预测器的更新在 commit/后端用架构值完成。
- 在本论文中的两个特殊用法：① EVES 辅助（EVES-assisted）：32KB EVES 的预测值被推入全局值队列作 base 值（不写 PRF、无直接性能影响），缓解全局预测的 value delay——去掉该辅助，EgDiff 平均 IPC 从 3.24% 微降到 3.11%、覆盖从 26.15% 降到 22.45%；② hybrid 模式：EVES 预测同时写 PRF 与 GVQ（作 EgDiff 的 base 值），两者都有效时 EVES 优先——Eves32Hybrid 6.48% > EVES 32KB 单独 4.81%。
- 评估配置（本论文）：EVES 3-cycle 延迟；8KB/32KB/unlimited 三档；Eves8/Eves32/EvesUL 单独 IPC 提升 4.43%/4.81%/5.61%；hybrid（+EgDiff）6.17%/6.48%/7.02%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：EVTAGE（tagged 多表、几何历史长度、概率计数器、2-way 组相联、指向 VTABLE 的值指针）+ ES（stride 预测器）+ 优先级仲裁。官方实现可在 CVP 竞赛框架中获得（CVP-1/CVP-2 规则页 microarch.org/cvp1/，社区实现 eric-rotenberg/CVP github 仓库）。
- 使用方式：在 CVP 模拟器或 gem5（本论文用 gem5 20.0+ 的 O3 CPU，src/cpu/valuepred/ 框架实现 EVTAGE/ES）中实现/移植 EVES，在 OoO 核配置中挂载（tagged 表容量对应 8/32KB 预算），跑 SPEC CPU 2017 等 benchmark 统计 IPC/覆盖率/误预测率，作为局部值预测的 SOTA baseline 与全局预测器（gDiff/EgDiff）对照。

涉及论文标题：
- Revisiting Global Value Prediction: A Resurgent Complement to Local Predictors
