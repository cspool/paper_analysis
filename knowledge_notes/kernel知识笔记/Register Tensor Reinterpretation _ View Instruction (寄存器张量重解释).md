## Register Tensor Reinterpretation / View Instruction (寄存器张量重解释)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Register Tensor Reinterpretation（View指令）是Tilus VM中的零开销操作，允许在不进行任何数据移动或拷贝的情况下，同时改变register tensor的数据类型（dtype）和layout。其正确性条件：两个tensor分布在同一数量的线程上（32 threads），且每个线程持有的总bit数相同。例如，32个线程各持有24 bits，既可解释为3×uint8（3×8 bits），也可解释为4×int6（4×6 bits），通过View指令在registers内即时切换（Figure 2c）。View是Tilus消除Triton shared memory layout conversion瓶颈的核心机制：Triton必须通过shared memory中转来改变register tensor layout，而Tilus直接在registers内完成。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。
在低精度weight loading pipeline中View指令的使用：
```
# b_tile在32 threads的registers中，每线程持有 3×u8 = 24 bits
b_tile = LoadShared(shared_tile, dtype=u8, layout=local(3).spatial(32), offset=0)

# View: 零开销将24 bits/thread reinterpret为 4×int6，同时改变layout
# 原layout: local(3).spatial(32) —— 每线程3个连续u8
# 新layout: local(2,1).column_spatial(4,8).local(2,1) —— Tensor Core兼容layout
# 不产生任何PTX指令（纯编译器metadata操作）
b_tile = View(b_tile, dtype=i6, layout=local(2,1).column_spatial(4,8).local(2,1))
```

对比Triton（Figure 1a）weight loading pipeline中的Step 4（shared memory layout conversion）和Ladder（Figure 1b）的Step 3-4（shared memory中转），View指令将两者都消除为registers内的零开销操作。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
View指令的实现基于Tilus的代数layout系统。编译器在IR层面维护每个register tensor的dtype+layout属性，View指令仅修改这些metadata属性——不生成任何实际的PTX指令或数据移动。在代码生成阶段，后续指令（如Cast、Dot）根据更新后的dtype+layout信息生成正确的per-thread操作序列。开发者使用方式：`b = View(a, [dtype], [layout])`，其中dtype和layout参数至少提供一个。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation
