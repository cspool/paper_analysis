## Row-Stationary / Output-Stationary 数据流分类（stationarity taxonomy）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Stationarity（驻留性）分类（源自 Eyeriss 的数据流分类法，[31] 在 SegFold 中引用）按"哪个操作数在 PE 内保持不动（stationary）"划分矩阵乘法数据流：weight-stationary（权重驻留，systolic array 经典形态）、output-stationary（输出驻留，psum 在 PE 内累加到底）、input-stationary（输入驻留）。SegFold 自称在 stationarity 分类下"在 PE 行粒度是 row-stationary、在 PE 粒度只是近似 output-stationary"：每条 PE 行终身拥有 C 的一整条虚拟行（row-stationary at PE-row granularity），但输出可能在 PE 之间被重映射（spatial folding 或 spad spill），所以不是严格的 output-stationary。这种两层视角正是 SegFold 的核心权衡：在最细粒度牺牲严格驻留，换取运行期再平衡负载的灵活性。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SegFold 的驻留结构：A 元素级流式（SELECTA 每周期广播选中的 A 及其 B 行）、B 行级流式（经 vector multicast + row shifter 注入）、C 虚拟行驻留 PE 行（PE 存 c 寄存器 + spad 溢出）。运转例子：一个 C 虚拟行分配驻留 PE 行 r，其各列索引分布在 (r,0..P-1)；B 元素注入后经 merge network 找到位置就地累加（output-stationary-like），但若该行超容量，多余列经 spatial folding 折到 PE 行 r±1（违反严格 output-stationary），提前完成的部分和经 temporal folding 卸入 spad 让 PE 复用于其他行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Row-stationary 由"PE 行拥有虚拟 C 行 + merge network 按列序维护"体现；近似 output-stationary 由"元素就地累加 + 动态迁移兜底"体现。使用：作为设计定位的语言（SegFold 论文以此解释与 TPU（weight-stationary systolic）、Flexagon 等的关系），也用于消融归因——fixed-k 实验使数据流退化为约束外积（static），量化了驻留/动态粒度对性能的贡献。

涉及论文标题：
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
