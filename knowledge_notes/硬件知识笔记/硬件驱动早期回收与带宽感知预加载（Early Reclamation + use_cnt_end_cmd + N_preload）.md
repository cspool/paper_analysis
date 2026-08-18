## 硬件驱动早期回收与带宽感知预加载（Early Reclamation + use_cnt/end_cmd + N_preload）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是 SMOOTH（ISCA'26）两大核心能力中的"运行时回收 + 预取"组合。① 早期回收（Early Reclamation）：传统 SPM 等软件显式释放信号或等整块计算完成才复用内存；SMOOTH 由硬件跟踪 tile 的使用：编译器静态 lifetime 分析为每个 tile/块标注剩余使用次数 use_cnt，buffer 跟踪 ISA 执行进度，当对某个块的末次数据访问发出 load 请求时置 end_cmd=1，DMC 随即递减该块 use_cnt，归零即提前回收，无需等待整层结束。② 带宽感知预加载（Bandwidth-aware Preloading）：DMC 在无待处理访存请求的 idle 周期，周期性识别 use_cnt=0 的块并立即预取后续数据，预取块数 N_preload = ⌊(U × BW) / Block_size⌋，其中 U 是当前空闲 compute cycle 数、BW 是硬件在运行期动态测量的可用带宽（应对统一内存下 CPU/GPU/NPU 争用导致的带宽波动）。预取按 block 粒度进行，prefetch 寄存器记录最后取到的块索引，buffer 访问时查寄存器判断数据是否已入片、未取完则回主存取数。回收与分配的顺序保证：先更新 block table 再清 bitmap，防止新分配在回收完成前覆盖数据。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子（decode 期低 OI 线性层与高 OI 非线性层交替）：Q 投影 GEMV 消耗完 S×V 中的 V_3、S_3 块后，buffer 发 end_cmd、use_cnt 归零；此时 softmax/GELU 等高 OI 运算占满向量单元、内存带宽空闲——DMC 检测到 idle 周期，用 find_zero 找到刚释放的块，按 N_preload 公式把下一 tile（如 W1_1）预取进 SRAM；S×V 计算进行中，V_cache 的单个 block 一空出就被用于预取，实现"数据一到就被用、带宽一空就被预取"的连续数据流。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Verilog RTL（DMC 的 free/alloc/find_zero 模块 + buffer 的 end_cmd 逻辑），在 LLMCompass 中建模；Yosys+ASAP7 7nm 综合。效果：该机制让 SMOOTH-ER 较只有 block 分配无早期回收的 SMOOTH-Base 平均再降 TTLT 24.0%、ITL 平均 11.1%（带宽受限时最高 47.0%）；16GB/s 低带宽下收益最大。开源：https://github.com/skkim-caslab/SMOOTH。

涉及论文标题：
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference
