## RBias（寄存器值相关性分支预测机制，含 digest / Log-RBias / Seq-RBias）

术语解释
RBias 是 RUNLTS 的核心新机制：作为 SC 的扩展组件，用后端流水线产生的逻辑寄存器值摘要（digest）作为 hashed-perceptron 输入，直接学习"寄存器值 ↔ 分支结果"相关性，不显式跟踪数据依赖链。Log-RBias 按逻辑寄存器组织（需 checkpoint 恢复），Seq-RBias 按动态指令距离的 ring buffer 组织（无恢复、可利用 wrong-path 值）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：(1) 动机：一批 hard-to-predict 分支的结果由近期计算出的数据值决定，历史机制（TAGE/SC）无法捕获；已有基于值的方法要么显式分析分支的数据依赖图（Chen 等哈希源寄存器值集合、Heil 等用操作数差值），要么依赖 pre-execution（Branch Runahead、Opportunistic Early Re-steering），实现成本高、普适性差；(2) 核心思想：把所有逻辑寄存器（AArch64 65 个）一视同仁作为候选，用紧凑 digest 表示每个寄存器的值，让感知机自己学会"哪个寄存器的值、什么样的值模式与分支结果相关"；(3) digest 生成（按寄存器类别不同哈希）：整数 = trailing zeros/ones 与 leading zeros/ones 计数按位 XOR + 值低 6 位（区分对齐指针、舍入值、小整数）；FP = 符号位+指数高位（粗略量级）；flag = 位模式重复拼接（增大 Hamming 距离、减少 digest 空间冲突）；(4) digest table：每逻辑寄存器一项，存"最新值的 digest"或"将产生该值的在飞指令 ROB index"+valid；decode 时写 ROB index、执行完成时执行单元广播 ROB index+结果值、匹配则替换为 digest 并置 valid；(5) 预测：按 bank（逻辑寄存器分 8 bank、WT 单端口化）并行，每 bank 用 UT（gskew 3-hash、仅按 PC 索引）选出 usefulness 最高的有 digest 寄存器，正 usefulness 才访问共享 WT（3 表 3-hash、同一 digest 三种哈希，gskew 式抗干扰），加权和 ×2.5 gain 输出，所有 bank 输出相加加入 SC；(6) 训练：预测错误或正确但 |SC sum|<阈值时触发；被访问 bank 训练预测所用寄存器，未访问 bank 随机选有 digest 的寄存器训练（利用空闲 WT 端口探索新相关性）；UT 按"假设 RBias 输出 0 vs 2.5×sum 时 SC 总符号是否翻转"来增减；(7) 能捕获的两类相关：隐式值相关（值在依赖图之外，如 qsort 中 n_caller 的中间结果与 n_callee 粗略成半分成相关）与误预测后可见（首个误预测不可避免，但被 squash 的 producer load 不消失、其值及时可见供后续分支用，阻止误预测级联）。
- 两个变体：(a) Log-RBias——按逻辑寄存器索引 digest table，可引用数百条之前未被覆盖的值，但中间结果常被同一寄存器覆盖；flush 需 checkpoint 恢复 digest table（每 checkpoint ~1 Kbit，持续监听 completion 通知比较 ROB index）。(b) Seq-RBias——fetch 顺序给每条指令分配 ring buffer 条目，预测时观察分支前固定动态窗口（如最近 64 条指令）的 digest，UT 按"动态距离"而非寄存器号索引（路径变化会改变距离，可在 digest 哈希中混入寄存器号缓解）；能观察被同一逻辑寄存器覆盖的中间结果，但观察不了很老的值；恢复只需回滚 ring buffer 指针，buffer 内容保留——因此 wrong-path 上生成的值可被后续分支利用；初步实验显示 buffer 可缩到 ROB 的 1/4 而无精度损失。
- 效果：RBias 平均降低 MPKI 2.46%（673 条 CBP2025 trace）；64 KiB TAGE-SC 预算下替换 6.6 KiB 容量仍 -2.83%；Log-RBias 两模拟器一致（对 wrong-path 不敏感），Seq-RBias 在 gem5（建模 wrong-path）下收益显著高于 CBP simulator。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件位置：SC 的扩展组件，与各 hashed-perceptron 组件并行求和；依赖后端执行单元的结果广播（ROB index + 值）。运转流程例子（fp_8_trace 的 X1>0 分支）：① 执行单元完成 X1=*p1 的 load，广播 {ROB index, 值} → digest table 中 X1 条目从 ROB index 变为 digest(X1) 并置 valid（Log-RBias）或写入 ring buffer（Seq-RBias）；② 之前 R32>R33 分支误预测触发 flush：Log-RBias 从 checkpoint 恢复 digest table（保留已完成执行的更新、撤销 decode 阶段的年轻更新），Seq-RBias 回滚 ring buffer 指针、X1 的 digest 留在 buffer 内；③ X1>0 被预测时：RBias 读 digest table 发现 X1 的 digest 可用 → 在 X1 所在 bank 查 UT（按 PC、3-hash）得 usefulness 为正 → 用 digest(X1) 访问共享 WT 三表 → 加权和 ×2.5 → 加入 SC 总输出 → 翻转/确认方向 → 预测正确、避免级联误预测。
- 例子（qsort n<7，隐式相关）：n<7 的结果由 callee 参数 n_callee 决定（预测时不可见），但 RBias 学到的是 n_caller 或其 line 8/10 的中间结果与分支结果的统计相关——算法层面 qsort 每次把数组约对半分，n_callee≈n_caller/2；Log-RBias 学到 line 8 中间结果（动态距离远、递归多次但不被覆盖），Seq-RBias 学到 line 10 中间结果（动态距离近但被同一寄存器覆盖、恰在 ring buffer 可见窗口内）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现组件：digest 计算单元（每类别不同哈希：整数 TZ/T1/LZ/L1 计数 XOR + 低 6 位；FP 高 8 位；flag 位重复）、digest table（65 项 Log / 256 项 ring buffer Seq，注意 Table II 注 * 指出两者 digest table 尺寸不同）、UT（每寄存器/每距离一份、gskew 3-hash、按 PC 索引）、共享 WT（3 表、8 bank 分片、单端口 SRAM）、gain 乘法（2.5）。存储：sR(Log/Seq) = 6.576/7.112 KiB（192 KiB 总预算内）。恢复：Log-RBias 用 checkpoint（每 ~1 Kbit、数量由 flush 恢复粒度决定、需持续广播监听），Seq-RBias 只回滚指针。开源情况：RUNLTS 随 ISCA 2026 artifact 开源（Zenodo 10.5281/zenodo.19453058，BSD 3-Clause Clear License），含 CBP simulator 与 gem5 实现，可用 `./run.sh` 复现 Figure 10。
- 使用场景：作为 TAGE-SC 类预测器的 SC 扩展提升对数据依赖型分支的预测；对 4/8/16-wide 核心均有效（小前端宽度/深度下因能捕获更近指令的相关而收益更大，Figure 17）；对 64 KiB 及以上容量预测器有效。局限：需要后端值在预测前可见（依赖指令时序），且 digest 是压缩表示、存在信息损失。

涉及论文标题：
- RUNLTS Branch Prediction with Register-Value Correlations and Hierarchical Table Orchestration
