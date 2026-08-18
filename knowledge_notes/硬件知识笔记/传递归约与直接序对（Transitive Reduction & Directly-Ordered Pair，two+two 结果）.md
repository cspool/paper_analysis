## 传递归约与直接序对（Transitive Reduction & Directly-Ordered Pair，two+two 结果）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
有向图的传递归约（Aho-Garey-Ullman [3]）：删除所有"可由其他边的路径推导"的边，得到的图与原图传递闭包相同（无环图下唯一），是"最小但保持全部序关系"的表示。QED 把 MCM 在线程内诱导的偏序图做传递归约，保留边对应的指令对即"直接序对"（Definition 1：两访存指令 a、b 的边 (a,b) 在 MCM 偏序的传递归约中）；定理 1（Directly-Ordered Pair Theorem）证明任何 MCM 序违反必发生在某个直接序对之间（反证：若所有直接序对都保持，由传递性整条路径都保持）。再叠加定理 2（Multiple Event Theorem）与传递折叠引理（Transitive Collapsing Lemma，III-B：两事件间的传递路径折叠成单边不丢环、不造假环）——只需考虑每指令一个相关外部事件、且只在乎事件排序不在乎来源核心——合称 QED 的 two+two 结果。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
two+two 给出两个可扩展性：① in-flight 指令数——不必考虑 n! 种重排，只需所有指令"类型对"×事件类型（O(m²e²) 棵树，m≤5 种指令 load/store/atomic/LR/SC、e<10 种事件 invalidation/外部读/eviction/miss/prefetch；SC 26 棵、TSO 36 棵、RVWMO 167 棵，表 II），与程序长度和 in-flight 指令数无关；② 核心数——外部事件（invalidation 等）只是其他线程指令在本核的代理，实现动作不看事件来源（如乱序 load 对匹配 invalidation 一律 squash），故任意多核任意代码降为"枚举事件排序"。算法 1 对每个直接序对枚举 ordering events 的排列建探索树（图 9 ld-ld 树），叶 trace 做环检测筛选 MCM 违反（成环=违反），违反 trace 生成谓词决策树。TSO 例子（图 4）：TSO 只放宽 st-ld，传递归约后 ld 与其后的 ld、st 与其后的 ld 等直接序对浮现，被松弛的边自然不在其中。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：图算法——对程序序诱导图求传递归约（先算可达性/传递闭包，再删冗余边），得到直接序对集合；QED 用 Algorithm 1（对每个序对的 ordering events 做 enumerate 排列生成 trace）自动化探索树。使用：这是"谓词数与规模无关"的理论根基——每个直接序对生成一棵树，叶 trace 查环后转谓词，谓词数 O(m²e²) 与程序长度、LSQ 尺寸、核心数完全解耦；RTL 侧再配合 fast-forwarding 处理"每个谓词要探索全部 LSQ 状态"的剩余爆炸。对比 baseline：litmus 测试只覆盖特定交错、bounded 穷举实践上限 7 条指令、PipeProof/Kami 仅 in-order——QED 用 two+two 首次把乱序 LSQ 的 MCM 验证做到与规模无关。

涉及论文标题：
- QED Scalable Consistency Verification of Memory Instruction Reordering in Hardware
