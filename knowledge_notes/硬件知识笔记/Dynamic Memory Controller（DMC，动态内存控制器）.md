## Dynamic Memory Controller（DMC，动态内存控制器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DMC 是 SMOOTH（ISCA'26）中负责片上 SRAM 运行时动态管理的硬件模块——把传统上由编译器完成的"何时分配、何时回收、预取多少"的粗粒度静态决策，改为硬件在运行期以 block 粒度动态执行。DMC 由五个轻量模块组成：find_zero（在 bitmap 中定位最长连续空闲块区）、alloc（为预取/新 tensor 分配块）、free（回收 use_cnt 归零的块）、block_table_lookup（虚拟地址→物理块地址翻译）、address_check（判断访问是否落在已翻译连续区、是否需要翻译）。DMC 的三个设计原则：① 固定大小 block 分配消除外部碎片、简化空闲空间追踪；② direct-mapped block table + bitmap 提供低开销翻译，连续区直通；③ 基于 use_cnt/end_cmd 的硬件驱动 early reclamation（不等软件显式释放）。编译器只做静态 lifetime 分析并标注 use_cnt，DMC 负责全部动态分配/释放——这是"把复杂内存调度从编译器转移到运行时硬件"的编译器-硬件协同设计。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子：buffer 对块 0x2400–0x27FF 的末次访问发 end_cmd=1 → DMC 在 block table 中递减该块 use_cnt；idle 周期 DMC 周期性扫描 bitmap，找出 use_cnt=0 的块——先更新 block table 标记失效、再清 bitmap（顺序保证回收完成前不会被新分配覆盖）→ find_zero 找空闲区 → alloc 为预取分配块并在 bitmap/block table 登记 → 从主存预取数据并记录最后预取块索引到寄存器。buffer 访问时查询该寄存器判断数据是否已入片。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Verilog RTL，集成进 LLMCompass cycle-accurate 模拟器（https://github.com/PrincetonUniversity/LLMCompass）；Yosys + ASAP7 7nm 综合验证面积（SRAM 侧控制逻辑 13,050 μm²，约 0.095% 相对 NPU 面积）与时序（alloc 1508.2ps 最慢、find_zero 364.4ps、free 654.6ps、bt_lookup 615.2ps、addr_check 83.7ps），功耗 pW 级。开源：https://github.com/skkim-caslab/SMOOTH。

涉及论文标题：
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference
