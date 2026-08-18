## Useful / Useless 代码行（commit-based usefulness，Definition 1）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bumper 的核心概念（Definition 1）：一个驻留在缓存中的代码行若在淘汰之前行内没有任何指令提交（commit），则为 useless，否则为 useful。直接推论：对指令行的 hit 不代表该行 useful——错误路径上的 hit 不贡献已提交指令，不能算有用。实测：平均 47%（43%–49%）的 useless L2C 代码行在生命周期内经历过至少一次 L1I hit，所以 L1I hit 不是 usefulness 的可靠代理。量化：移动应用 L2C 中 17.5%–52.8%（平均 33.2%）容量被代码行占据，其中 61.1% 是 useless；即 useless 代码行平均占整个 L2C 的 20.3%（>1.2MB），且其生命周期与 useful 行相当。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
判定流程：代码行插入 L2C → 行内指令进入流水线 → 若有指令 retire 则该行 useful。Bumper 据此实现缓存管理：所有代码行以 RRPV=3 插入，只有行内首条指令提交后经 hint 链（send_hint → ROB 位 → retire VA 回传 → HL1Q/ITLB → L1I tag → HL2Q → L2C）把该行提升到 RRPV=0；无提交证据的行保持 RRPV=3 被快速淘汰。效果：useless 行生命周期 -57.9%、useful 行 +52.5%、其他行（数据/MMU）+21.5%；useless 占用 20.3%→9.5%。论文指出该定义可推广到数据行（列为 future work）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点是把"提交证据"与 cache 行关联起来：Bumper 用 L1I tag 的 send_hint 位 + 填充响应的 l2_vulnerable_fill 标志保证每行 L1I 生命周期内至多一次提升信号（422B 总存储）。与 Dead Block Prediction 的区别：DBP 预测"未来不再使用"，Bumper 依据"过去是否 commit"做事实判定，无预测误判风险，也不随 footprint 增大而膨胀。

涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
