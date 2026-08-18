## 自适应 Merge Network（merge network / merger）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Merge network（合并网络）是 SegFold PE 行内的本地互连：每个 PE 含一个 merger（合并器）与四端口 router，merger 把到达的 B 元素列索引 b 与当前 PE 存储的 C 列索引 c 比较，决定：(a) b>c——向右转发（因列序单调，目标必在右侧）；(b) b<c——触发该位置及右侧 C 索引整体右移一格、在空位插入新 C 条目；(c) b=c——就地累加。这实现 SEGMENTBC 的"on-the-fly 交集"：B 元素进入时是 B（带 B 元数据），离开时成为 C（值+元数据已更新）。与 Flexagon/Trapezoid 的 merge-reduction network（MRN，把部分积按模板归约的树/网络）不同，SegFold 的 merge network 以"列索引比较 + 动态插入/迁移"完成 C 位置发现，是动态映射的物理载体；router 四端口（右/上/下/左）但任一时刻仅一个方向激活，方向由 folding 机制决定（默认右向，即最简右传播合并网络）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转例子（论文 Fig.6）：PE 行存 C 列索引 {1,3,4} 于 y={0,1,2}，B 元素 b=2 注入 y=0。(a) y=0 处 b>c(1)→转发 y=1；(b) y=1 处 b<c(3)→因单调性右侧不可能匹配，把 {3,4} 右移、在 y=1 插入 2，f_tout=y=1；(c) 若 b=3 则 y=1 处 b=c→就地累加。保证合法性的前提：注入点左侧所有 c < b（Fig.6d 的违规场景由数据流禁止）；IPM 负责构造满足该条件的注入点。SEGMENTBC 的注入点（segment 起点）由 IPM 二叉搜索近似给出，merge network 仍会纠正到真实位置——即使 LUT 过时（只可能偏左），正确性不受影响，仅可能加长 segment。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：每 PE 集成 ALU（本地计算）、FIFO（暂存已匹配未消费 B 值）、四口 router；PE 行间用 row shifter 水平流式搬移 partial B 行，跨行用 vector multicast network 分发。使用：merge network 的 merge 事件/位移是负载信号，folding 依据 PE 占用调整 router 方向；可扩展性上单行最坏位移 O(P)（P=每行 PE 数），但 IPM 近优注入点使期望位移远短。

涉及论文标题：
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
