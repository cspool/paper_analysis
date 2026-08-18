## Symmetric Tensor Allocation（对称张量分配 / SymMalloc）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Symmetric Tensor Allocation（对称张量分配）是 RoCC 论文提出的 GPU 内存分配策略，把参与 CC 的 tensor tile 映射到所有参与 GPU 上的相同本地物理地址（对称物理帧），从而在跨 GPU 的 ROP 间通信时免去地址翻译/换算。动机：ROP 直接访问 L2 cache（通常用物理地址），若各 GPU 把 tile 映射到不同物理地址，跨 GPU 门铃就需经 MMU 翻译虚拟地址（运行时开销大）。实现为自定义 GPU memory allocator SymMalloc：分配时检查参与通信的各 GPU 未用物理帧池，选择各 GPU 上具有相同本地物理地址的帧；由于 DL 训练通常占满整个 GPU，对称帧一般可找到（无需连续）。找不到对称帧时回退：在 descriptor 中记录 tensor 虚拟地址，走虚拟地址方案。与 OpenSHMEM/nvshmem 的 symmetric heap 及 BarreChord 的对称虚拟地址不同，RoCC 的对称是"物理地址级"对称（虚拟地址可以不对称，图 19），且只在 CC 涉及的 tile 上使用，专门服务于 ROP 间的门铃路由。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在硬件中的运转流程：① DL 框架启动 kernel 前用 SymMalloc 分配 CC 用的输入/输出 tensor → ② SymMalloc 在各 GPU 上选择相同的本地物理帧（图 19：绿色 Src 页与灰色 Dst 页在两 GPU 上物理地址一致，虚拟地址可不同）→ ③ GEMM 结果写进本地对称地址 → ④ ROP 触发 CC 时，门铃/数据包只需携带"与本地相同"的目标物理地址 + 目标 GPU rank 即可寻址（源/目的物理地址除 rank ID 外完全相同）→ ⑤ 接收 GPU 的 ROP 直接在本地对称地址读写，无需地址换算或 MMU 翻译。作用：把"下一跳去哪"从地址翻译问题简化为 rank 索引问题，节省门铃包处理时间；代价是分配时一次性的对称帧查找（初始化阶段、不在关键路径）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SymMalloc 作为 GPU driver 的自定义内存分配器（CUDA 风格 cudaMalloc 替代/扩展），分配时做跨 GPU 物理帧协商；与 RoCC descriptor 配合（descriptor 记录对称地址基址）。使用方式：仅对参与 CC 的 tensor 使用（非 CC tensor 走常规分配）；常见 DL sharding 策略（data/tensor parallelism）采用均匀分配，天然利于找到对称帧。相关已有概念：OpenSHMEM symmetric heap（对称虚拟地址）、nvshmem 动态对称堆分配、BarreChord（对称虚拟地址优化 MCM GPU 虚拟内存）——RoCC 与之的区别是物理地址级对称 + 面向 ROP 门铃路由的轻量使用。no note evidence；web 补充：OpenSHMEM/nvshmem 规范与 BarreChord (ISCA'24) 论文。

涉及论文标题：
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication
