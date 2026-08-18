## Value Prediction（值预测，VP）

术语解释
<值预测是一种微架构投机技术：在指令结果真正算出之前，预测写寄存器指令（尤其 load）的输出数值，用预测值提前唤醒并执行依赖指令，从而打破真实数据依赖、提升指令级并行（ILP）。>

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：① 现代 OoO 处理器的性能受真实数据依赖（RAW）限制——一条指令要等前一条写寄存器指令的数值出来后才能真正执行，深流水线 + 高访存延迟（load 数十周期）让这种等待被放大；② 值预测的观察基础是"值局部性"：同一静态指令（或相关指令组）产生的数值序列高度可预测（如 stride、重复常量、地址模式）；③ 于是可以在寄存器写指令 dispatch/执行前就投机生成其"预测值"注入物理寄存器文件（PRF），立即唤醒依赖指令，让它们与真实执行并行推进；④ 指令真正执行后做"验证"：预测值=实际值则无事发生（白赚 ILP），不等则触发 pipeline squash（丢弃预测后所有指令、按实际值重执行）——因此精度必须极高（论文指出现代设计普遍要求 >99%），否则误预测惩罚会吞掉收益。
- 值预测自 Lipasti & Shen（MICRO 1996，"Exceeding the dataflow limit via value prediction"）提出，分两支：上下文型（context-based，用 PC/分支历史等全局上下文索引预测表）与计算型（computational，对上一个值施加函数如 stride 加法）。所有主流预测器都基于"局部值历史"（单个静态指令的值序列）；全局值预测（跨动态指令的全局值历史，gDiff/EgDiff）长期被忽视，本论文重新审视。
- 网页佐证：EVES 在 CVP-1（2018 首届值预测锦标赛）全部赛道夺冠并保持 8KB/32KB 赛道最高 IPC speedup，是局部值预测 SOTA（web: 电子学报综述，Seznec, "Exploring value prediction with the eves predictor", CVP-1 2018）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 值预测器的硬件生命周期（论文 §III-B 的工作流）分三阶段：预测（prediction）→ 验证（verification）→ 更新（update）。具体流程：① 预测：指令 dispatch 时以 PC 哈希索引预测表，tag 匹配且置信计数器饱和才放行，取出 diff/stride 等预测参数，访问值队列取 base 值，算出预测值写入 PRF（唤醒依赖指令）并写入预测缓冲；② 验证：指令（如 load）在 LQ 完成执行后，把实际结果与预测值比较，匹配则跳过 PRF 回写（省端口）、不匹配则置 misprediction 标志并触发 squash；③ 更新：用架构正确值（commit 后）更新预测表条目（置信计数器、usefulness、预测参数）。论文的 EgDiff 还利用 GVQ 存放预测值做验证，避免额外的 PRF 读/写访问（减少寄存器端口压力）。
- 具体例子（论文 xalancbmk 链表遍历）：H2P 指令 I6（ldr x0,[x1,#16]）依赖 I1（ldr x1,[x22,#8]）的输出。预测器用 I6 的 PC 查表得 stride=40，等 I1 结果（或 base 值）就绪后，预测值 = I1 的值 + 40，提前写入 PRF 唤醒 I7（ldr x1,[x0,x20]）等依赖指令；I6 真正执行后验证，匹配则依赖链已提前解开。
- 价值：论文实测 EgDiff 平均 4.37% IPC 提升、覆盖率 25.87%、精度 >99%；与局部预测器 EVES 混合最高 7.02% IPC 提升。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现载体：现代处理器在 OoO 核的前端/rename 阶段挂接值预测器（与分支预测器并列），预测值写入 PRF、唤醒走现有 issue-queue/scoreboard wakeup 协议，验证在 LQ/执行完成路径上比较，误预测走现有 pipeline squash 恢复机制。预测表多为 PC 索引的 SRAM（tagged，如 TAGE 式或直接索引），配置信计数器（saturating counter / FPC）与 usefulness/替换策略控制投机激进程度。
- 开源参考实现：gem5 O3 CPU 内置值预测框架（src/cpu/valuepred/）：VPUnit 基类暴露 predict()/update()/specUpdate()/squash() 四个接口，CompositeValuePredictor 组合多个子预测器并用可配置仲裁器（固定优先级/随机/轮询/置信度）选结果，ExampleValuePredictor 是新增预测器的模板（web: xs-gem5 文档）；Championship Value Prediction（CVP）官方模拟器 eric-rotenberg/CVP（github.com/eric-rotenberg/CVP）是社区实现/评估值预测器的标准平台（web: GitHub CVP）。
- 使用方式（研究/复现）：在 gem5/CVP 中实现预测器 → 在 OoO 核配置里挂载 → 跑 SPEC CPU 2017 等 benchmark（本论文用 ARMv8-A/AArch64 -O3、SE mode、Simpoints 3.2、150M 指令 = 50M warmup + 100M 统计）→ 统计 IPC、覆盖率、误预测率。工程上需注意：值误预测的最小恢复惩罚（本论文建模 21 cycles，与分支误预测相当）、预测器访问延迟（EVES/gDiff/EgDiff normal 3 cycle、deferred +2 cycle）。

涉及论文标题：
- Revisiting Global Value Prediction: A Resurgent Complement to Local Predictors
