## In-Transit Computation（在途计算）与 Curry ALU（Currying 单操作数 ALU）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在途计算 = 数据在互连/内存层级中移动的同时就地完成计算，避免"搬到专用单元→算→搬回"的往返（同族概念：in-network compute、in-switch computing、Active-Routing）。CompAir 把 LLM 里三类跨 bank 数据移动全部改为在途完成：(i) RoPE 的粒度失配（向量阵列 vs 标量交换）；(ii) 非线性函数（RMSNorm/SiLU/Softmax 的 exp、sqrt）；(iii) 集合通信（切分算子的归约/广播）。支撑机制 Curry ALU 借鉴 Lambda 演算 Currying（多元函数柯里化为链式一元函数）：ALU 由单操作数驱动——flit 动态携带一元算子 InputOp 与左值 InputVal，内部 ArgReg 静态存右操作数，IterArg/IterOp 支持 ArgReg 迭代更新（如 InputVals+=ArgReg；或 ArgReg+=IterArg），因此无需多 flit 操作数匹配，单个 flit 即可独立触发计算；对比传统 dataflow（数据动态流动、算子静态绑在 ALU 里）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CompAir 的三个用法：(1) exp（Softmax/Sigmoid 核心）：ArgReg=6 作迭代轮计数器，IterArg=1、IterOp='-'，逐轮 acc=acc×X/IterRound+1（Taylor 截断，自最内层向外），每通道 16 bank × 2 ALU = 32 路并发；(2) sqrt：Newton 迭代；(3) 归约树：16 宽归约为 4 层二叉树，ArgReg 作非叶节点累加器；广播树为其逆。执行流程：packet（Type 4b/Data 16b BF16/IterNum 4b/Path[0..3]×12b）进入 router → flit compute 级 Curry ALU 就地改 Data → 继续路由或写回 DRAM 行。收益：非线性总延迟 −30%、长上下文延迟 −25%、Curry ALU 面积仅为 router 的 2.94%（4×Curry ALU 资源少于一个定制 16 输入 Softmax 单元）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：路由器内每 ALU 1 adder/multiplier/divider + ArgReg/IterArg/IterOp 寄存器（CompAir：每 router 2 个 BF16 Curry ALU、flit 72b、DC+UMC 28nm 综合）；类似概念在交换机级有 NVLink SHARP。使用要点：计算级与 ST 并行保证零额外跳延迟；迭代类算法靠 IterTag 触发寄存器更新；path generation（见编译框架库）把依赖的标量操作融合成单 packet，避免逐步 DRAM 写回（33–50% 延迟优化）。

涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
