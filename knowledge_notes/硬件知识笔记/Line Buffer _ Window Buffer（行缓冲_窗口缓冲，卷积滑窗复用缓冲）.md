## Line Buffer / Window Buffer（行缓冲/窗口缓冲，卷积滑窗复用缓冲）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 卷积/模板计算流式加速器的经典片上复用结构（Web 证据：AoCStream arXiv:2212.11438、Cornell ECE5760 等）：line buffer 存输入特征图最近若干行（K×K 核需 K−1 行历史），window buffer 维护当前 K×K 滑窗，相邻输出像素复用大量输入元素，避免重复读存储。CODO 自动生成的版本（Fig. 7）：line buffer lb[n][ci][kh][w]（深度=核高 kh、保留 kh−1 行历史，新元素写入最新位置）；window buffer wb[n][ci][kh][kw]（每新列 w 进入时横向移位、从 line buffer 装入新列）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- CODO 的 violation-free reuse buffer generation：检测乘加（MAC）模式识别卷积/矩阵乘等计算密集 kernel → 提取目标数组输入/输出访问模式 → 把出现在数组索引中的循环维度分类为 FIFO 维度、其余为 reduction dimension → 在 reduction 维的迭代域上构造 line/window buffer（归约维即卷积核滑窗维）。同时重构包围循环使 FIFO 访问的循环维度与数组索引精确对齐（既不缺维也不多无关维），防止生成缓冲本身引入新违例。
- 并行化合法性指引（Fig. 7 色彩标注）：最外层红色循环不可并行（会展开全部三个内部区域、引入复杂数据依赖与控制问题）；中间橙色循环对应 FIFO 索引、并行化会改变 FIFO 访问模式需配套措施；最内层绿色循环与 FIFO 无关、可安全并行——这一分析直接供后续 DSE 剪枝。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 经典手工实现：K−1 个行深 BRAM + K×K 移位寄存器窗口 + 并行 MAC 阵列，warm-up 后 II=1（每周期一个输出）。CODO 把它们做成 MLIR pass 自动生成（对 ping-pong 缓冲场景同样适用），消融显示其对高数据复用模型（ResNet-18、YOLO）增益显著（Opt3 vs Opt2）；生成后重跑正确性 pass 防新违例。

涉及论文标题：
- CODO: An Automated Compiler for Comprehensive Dataflow Optimization
