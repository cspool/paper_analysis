## SW-HW 调度接口与轻量硬件反馈（Feedback Counters / Reconfiguration Engine / Tiling Controller）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SW-HW 调度接口（software-hardware scheduling interface）是 Harmonia 把"同构可重构硬件"变成"逻辑异构执行引擎"的硬件桥梁，由三项轻量控制结构组成（Fig.5，合计 3.3% 面积）：(1) Feedback Counters——每 PE 行一组计数器，跟踪 MRN merge 事件、psum spill、SRAM pressure、PE stall（128 行仅 128 个计数器，占 <0.5% PE 阵列面积），经轻量 metadata crossbar 汇聚到 Tiling Controller；反馈路径与主执行数据通路完全解耦，不影响 DN/MRN 时序。(2) Reconfiguration Engine——在 tile 边界执行数据流切换：pipeline flush + DN 路由表/MRN 模式重编程 + AGU 与 buffer 控制器策略（分配/逐出/重置）更新，共 20–50 cycles。(3) Tiling Controller——运行动态 Tuning 策略：读反馈计数器、按成本模型 Gain>α·Cost 决策是否切换数据流/微重切块、执行滞回（异常计数器连续 T=2~4 周期超阈值防振荡）与最坏情况回退（切换失败损失 ≤1 次重构延迟 50 cycles）。其作用是把 microarchitectural 信号翻译成 tile 级调度动作，闭合"静态规划→在线细化→硬件反馈修正"的调度回路。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（一个异常 tile）：PE 行在执行中 psum 溢出 → spill 计数器递增 → 经 metadata crossbar 上报 Tiling Controller → 连续 T 周期超阈值（滞回）→ 控制器用式 (3) 估算切换增益（Gain=T_before−T_after vs Cost=T_reconfig+T_flush+T_buf_reset）→ 若净收益大于 α·Cost 则发重配置命令 → Reconfiguration Engine 在 tile 边界 flush 流水线、重编程 DN/MRN/AGU/buffer → 下一 tile 以新数据流执行 → 若切换无效，回退静态基线（最多损失 50 cycles）。反馈信号包括：SRAM pressure/psum spill（稀疏度与数据流失配的指示）、MRN merge 深度与 stall（merge-tree 行为偏差）、PE stall cycles（操作数不可用、DN 拥塞、merge 背压的聚合效应）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RTL 综合于 TSMC 28nm（Synopsys DC），Schedule 部分（Feedback+Reconfig.）0.25 mm²/3.3%；数据通路/MRN/DN 完全不变。使用场景：任何需要在运行时感知硬件状态并做细粒度重配置的稀疏/不规则加速器——Harmonia 用它实现 tile 级数据流切换与微重切块，在 16 个 SpMSpM workload 上平均 1.75× 加速（orani678 3.46×）、端到端 DNN 1.87×、能耗 -40%、动态开关总 stall <1%，并验证在 16×16~64×64 阵列与不同 SRAM 容量下稳健扩展。论文未提供开源实现；与商用 PMU/performance counter 的区别在于信号直接驱动重配置决策而非仅供观测。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication
