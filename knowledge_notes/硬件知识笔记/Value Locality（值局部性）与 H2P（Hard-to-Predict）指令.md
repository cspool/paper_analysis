## Value Locality（值局部性）与 H2P（Hard-to-Predict）指令

术语解释
<值局部性指程序执行中寄存器写指令的数值序列具有可预测规律（重复、stride、由相关指令决定）的现象；H2P（难预测）指令指局部值序列剧烈波动、现有局部预测器难以预测、但可能被跨指令全局规律捕获的指令。>

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：① 值预测能工作的前提是值局部性——如果所有指令的值都像随机数一样，预测就无从谈起；Sazeides & Smith（MICRO 1997，"The predictability of data values"）最早系统研究了数据值的可预测性，发现许多指令产生的值高度可预测；② 值局部性有多种形态：相同值重复出现（equality/重复局部性）、值等差递增（stride 局部性，如地址计算、循环计数器）、上下文相关（同一分支历史下值相同）、计算相关（前值经函数得到）；③ 传统局部预测器（EVES 的 EVTAGE+ES）利用的是"per-static-instruction 局部值历史"——每个静态指令自己过去的值序列；④ 但存在 H2P 场景：某条指令（如链表遍历中访问 next 指针的 load）的局部值序列看似有 stride 却剧烈波动（非周期），局部预测器（上下文型或计算型）都难以捕捉；⑤ 然而这类指令的值可能与其他指令（如链表头指针 load）存在稳定的"全局 stride 关系"（I6 值 ≈ I1 值 + 40，几乎恒定）——这就是全局值局部性（inter-instruction global stride locality），是 gDiff/EgDiff 类全局预测器的工作基础。
- 本论文用 xalancbmk（SPEC CPU 2017）的汇编实例证明：I6（H2P）局部值序列与局部 stride 序列都强波动（Fig.3a/3b），但 I6 与 I1 之间的全局 stride 序列几乎恒定 40（Fig.3c），仅两次微小波动。
- 网页佐证：值局部性概念源自 Lipasti & Shen MICRO 1996 与 Sazeides & Smith MICRO 1997；gDiff 论文（ISCA 2003，Zhou et al., "Detecting global stride locality in value streams"）正式提出"全局 stride 局部性"并证明其存在（web: ACM DL）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件上的角色：值局部性决定预测器用什么"特征"做索引/预测——局部预测器按静态指令 PC 索引、存该指令自己的历史值/stride；全局预测器除 PC 外还记录"全局值序列"（把程序顺序上各动态指令的输出连成队列），预测时取队列中某距离处的 base 值加全局 stride。
- 运转流程例子（论文 Fig.2/Fig.3 的 xalancbmk）：① I1（ldr x1,[x22,#8]）执行得到链表元素地址 A；② I6（ldr x0,[x1,#16]）是 H2P——它的值（x0）是"结构体偏移 16 处的 next 指针内容"，局部来看每次跳变不规则；③ 但 I6 与 I1 存在全局 stride 40（元素按引用顺序分配，next 地址 = 当前元素地址 + 40），于是只要 I1 的投机/实际值可用，I6 的预测值 = A + 40；④ 全局预测器在值队列里维护这种"距离"关系（distance 字段 = I6 相对 I1 在值队列中的位置差），预测时按 distance 取 base、加 stride。
- 度量：值局部性强弱用预测覆盖率（coverage，被成功预测的指令占比）与准确率（accuracy/misprediction rate）刻画；本论文 EVES 局部预测器覆盖增长受限，EgDiff 全局预测补充 9.4% 局部预测器完全看不见的覆盖率（EvesULHybrid 覆盖分解：EgDiff-only 9.4%、Ev-only 38.3%、Es-only 0.8%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：局部值局部性用 PC 索引的预测表（存历史值/stride/上下文）+ 置信计数器实现；全局值局部性用"PC 索引表 + 全局值队列（GVQ）"实现：表项存 stride 与 distance，队列按程序顺序保存近期指令输出，预测 = base(distance) + stride。检测全局相关性是难点——本论文用 distance polling 在运行中动态收敛到稳定的 (distance, stride) 对。
- 使用/评估：用 SPEC CPU 2017（本论文 ref 输入、ARMv8-A -O3、Simpoints 3.2）跑 gem5 模拟，统计覆盖率与误预测率；H2P 指令是驱动"为什么需要全局值预测"的典型 workload 特征，可用工具定位（本论文以 xalancbmk 链表遍历为例，论文未明确说明是否有自动化 H2P 检测工具）。

涉及论文标题：
- Revisiting Global Value Prediction: A Resurgent Complement to Local Predictors
