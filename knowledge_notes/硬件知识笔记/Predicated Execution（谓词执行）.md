## Predicated Execution（谓词执行）

术语解释
谓词执行：操作携带谓词位，谓词为假时结果不写回，从而把短分支 if-convert 为无条件数据流执行，避免控制流发散与重收敛开销。DICE 每 PE 有 1-bit 谓词输入，同时兼作输出控制（写回使能 + 访存请求 valid 标记）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
与分支改控制流不同，谓词执行改数据流：两条路径的操作都执行（或时钟门控），由谓词位选择哪条路径的结果提交。在 CGRA 上下文中（论文 II-B）：支持谓词执行的 CGRA 可把带控制流的更大 CDFG 片段映射为单配置（如 AHA [23]、RipTide [14]）。DICE 用法：p-graph 划分约束①（p-graph 内无控制发散）可被谓词放宽——编译期把因发散拆开的路径合并回单 p-graph，两条路径的操作都布局在同一配置内，谓词位选择性使能；兼作输出控制：使能/禁用寄存器写回、标记访存请求 valid/invalid。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
例子：`if (c) x = a+b; else x = a-b;` 谓词化后：
```
add r1, a, b, pred=c    // c 为假则 r1 不写回
sub r2, a, b, pred=!c   // c 为真则 r2 不写回
```
Annotations：两操作同周期并行执行，写回门控由 1-bit 谓词控制；访存变体将谓词连到请求 valid 位，谓词假的请求不发出。效果：发散路径合并为单 p-graph，减少配置数与重配置开销、提升利用率。代价：mask 掉的线程寄存器仍被读出（scale-up 实验 DICE-U 中 BFS/GE-1 的 RF 访问微升的原因之一）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：每 PE 一条 1-bit 谓词输入线 + 写回使能门/请求 valid 门；编译器做 if-conversion 并决定哪些发散可合并。使用：消除短控制流发散、合并 CGRA 配置；与 PDOM 栈/分支预测互补（长路径与嵌套发散仍靠控制流机制）。论文未明确说明谓词信号的具体布线来源（由 PE 间直连还是控制逻辑提供）。

涉及论文标题：
- DICE: Enabling Efficient General-Purpose SIMT Execution with Statically Scheduled Coarse-Grained Reconfigurable Arrays
