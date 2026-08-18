## Tensor Walker Unit（TWU，张量遍历/地址生成单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TWU 为 TPB 功能单元生成流式访问 HBSM 的地址序列：配置嵌套循环层数（每层 Initial/Step/Final 三个计数器），每迭代输出一个地址 = 各层 Value 计数之和；内层 Value 达 Final 触发外层按 Step 递增（外层达 Final 后下次重置 Initial）；最内层每迭代无条件 +1。支持双缓冲：在外层设带 buffer offset 的 Step，即可在两块 buffer 区间交替访问。典型配置：功能单元 2+ 个输入 TWU + 1 个输出 TWU；是卷积等复杂非线性地址模式（嵌套循环）高效访问的关键。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 例子（3 层 TWU，Fig.10）：卷积输入访问经 3 层嵌套循环描述（如 batch/out_h/out_w 或通道/高度/宽度），输出地址 = 三层 Value 之和；每迭代产出一个地址，供 TCU 流式读激活/权重；双缓冲（外层 Step 带 buffer 偏移）让 producer（DTDU 搬运下一 tile）与 consumer（TCU 计算当前 tile）在两块 HBSM 区间间无缝交替。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：多级计数器（每级 Initial/Step/Final 寄存器）+ 加法树 + 双缓冲控制，逐 cycle 生成地址、无指令开销。使用：TPB 指令配置各层循环参数，编译器按算子访问模式生成；与 HBSM 结合实现单元间通信无需专用 datapath。未开源。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
