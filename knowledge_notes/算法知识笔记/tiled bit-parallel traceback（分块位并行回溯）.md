## tiled bit-parallel traceback（分块位并行回溯）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SWG 比对有前向（分数矩阵）与反向（traceback）两趟：traceback 从最优分数单元出发、按"上一步决策决定下一步"逐单元回溯出比对路径——**每步依赖前一步**、串行紧依赖，需 GHz 级高频才高效；FPGA 时钟仅数百 MHz，逐单元 traceback 极慢。此前工作或省略 traceback（只加速分数矩阵，端到端收益有限）、或用 2-bit 编码路径（Cheng24 [8]）、块单元级间并行（G³SA [25]、Nawaz [61]）。Lembas（ISCA'26）的 **tiled bit-parallel traceback**：把矩阵切成 8×8 tile，前向时在每个 tile 边缘单元（灰格）**完整编码"从相邻 tile 外起点到本 tile 各边缘单元的最优路径"**——每步 2-bit（x/y 偏移），8×8 tile 内最长 16 步、15 个边缘单元 → 每 tile 32×15=480 bit，正好对齐 512-bit HBM 接口；反向时读编码、**popcount x/y 位**即可算出下一 tile 入口单元，**每 cycle 前进一整 tile** 而非单单元。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 实现细节（论文 VI-B，图 7b/8）：每 PE 带**历史寄存器（Hist Reg）**——维护最近两个访问 cell 的滑动窗口最佳历史；PE 算出当前 cell 分数并决定下一步时，从三个输入历史寄存器按方向选一个、把新确定的路径步（2-bit）追加进去；前向算分与历史编码同流水完成。反向流程：读 tile 边缘单元编码（例 4×4 tile 的 `11111010`，每 2 bit 一步 x/y 偏移）→ popcount x 位得横向位移、popcount y 位得纵向位移 → 定位下一 tile 入口 cell（一步跳过整个 tile）→ 重复直到出 band。8×8 是 timing 约束下的最大 tile（更大则历史寄存器访问不满足时序）；每 tile 480 bit 与 512-bit HBM 接口匹配；每次加载 tile 预取 4 个缓解 HBM 延迟。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：RTL 在 PE 微架构内（16×16 PE 脉动阵列 + 历史寄存器），扩展阶段资源占用 Extend 431,957 LUT (49.61%)/105 BRAM (7.81%)（表 IV，tile 寄存器面积占比小）。效果：traceback 从"每 cycle 1 单元"变"每 cycle 1 tile"（每 cycle 前进 8 单元）；W=2048 时 traceback 开销比次优设计低 1.77×；窄 band（W=1024 研究常用）下收益显著；extend 总计 48 GCUPS/FPGA。使用场景：任何"低时钟硬件 + 需高质量 traceback"的 DP 比对加速器（FPGA、PIM、近存）。

涉及论文标题：
- Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration
