## PIM 运行时数据布局管理（bit-sliced / column-contiguous / baap_set_mode drain-flip-reload）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BAAP 三模式各有"自然布局"（论文 §III-B2）：Scratchpad 模式字节寻址；SIMD 模式位切片（bit-sliced：第 i 子阵列存所有向量元素第 i 位，利于位并行/位串行算术）；DirectAP 模式列连续（column-contiguous：整词单周期 CAM 匹配）。布局责任分四层：① host 用扩展的 scatter/gather/broadcast（带 layout 参数选 stride）做初始摆放；② DPU 运行时库 baap_set_mode(mode, m) 把 m 个向量寄存器工作集 drain（若脏则排回 DRAM bank）→ flip → reload 成新布局，纯 DPU 侧 DMA strided 访问实现、无新硬件；③ kernel 主模式编译期静态声明，ISA 允许隐式换模（跨模式指令）或显式换模；④ 硬件写 CSR 切换灵敏放大器（控制级、无微架构状态排空）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
成本模型（论文）：最坏重排 t = 2×6KiB ÷ 72.58MB/s ≈ 0.171ms ≈ 6×10⁴ cycles@350MHz（72.58MB/s 为最坏细粒度访问的每 bank 有效带宽 [21]），相当于 2–15 次 SIMD 乘法。
```
baap_set_mode(new_mode, m):        # DPU 侧运行时例程
  for v in working_set[0..m-1]:
    if dirty(v): dma.store(bank_addr(v), v)   # 排空到 bank
  flip_layout()                      # CSR 切换 + 布局元数据更新
  for v in working_set[0..m-1]:
    dma.load_strided(v, bank_addr(v), new_mode.stride)  # 按新布局重载
```
决策规则：频繁换模不摊平 → 应单模式映射 + 特定阶段 host 回退/协同（同固定功能 PIM 与 UPMEM 基线的做法）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
类比 Castle/CAPE 的 mode reconfiguration。使用：BAAP 评估的 PrIM/Phoenix kernel 均为固定模式（免换模）；基因组 k-mer 计数与图遍历两阶段都可用 DirectAP 布局连续执行。布局选择即算子选择：位切片适合算术、列连续适合匹配——编译器/程序员在 kernel 粒度静态声明，运行时按需 drain-flip-reload。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing
