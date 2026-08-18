## Smith-Waterman-Gotoh（SWG，仿射 gap 罚分的动态规划序列比对）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SWG 是序列局部比对的动态规划算法：Smith-Waterman（1981）定义局部最优比对分数；Gotoh（1982, J. Mol. Biol. 162(3):705-708）引入**仿射 gap 罚分**辅助数组把复杂度从 O(mn²) 降到 O(mn)。仿射罚分 $g(k)=\alpha+\beta k$（$\alpha$=gap 打开罚分、$\beta$=gap 延伸罚分）。递推三值（Lembas 论文的 S/E/F 即此 H/E/F）：$E[i,j]=\max\{E[i,j-1], S[i,j-1]-\alpha\}-\beta$（水平 gap）、$F[i,j]=\max\{F[i-1,j], S[i-1,j]-\alpha\}-\beta$（垂直 gap）、$S[i,j]=\max\{S[i-1,j-1]+Z[A_i,B_j], E[i,j], F[i,j], 0\}$（0 截断保证局部性）；计算顺序严格按 $i,j$ 递推（每 cell 依赖左/上邻居）。**banded SW** 忽略远离对角线的区域降计算量（Minimap2 默认 20 kbp band，研究常用 W=1024 折中精度/性能）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 正向分数矩阵的**反斜对角线（wavefront）依赖结构**与脉动阵列/GPU 天然契合：同一反斜对角线上的 cell 可并行计算。Lembas extend 加速器（ISCA'26）：16 kernel × 16 PE 的 1D systolic array，每 PE 算矩阵一行、交错推进，每 cycle 用一个 cell 的四个输入（$S[i-1][j-1]$、$E[i-1][j]$、$F[i][j-1]$、$b^j$）算 S/E/F 三值——$F$ 随 PE 从左向右移动而缓存在 PE 内（无需跨 PE 传递）、$E/b$ 走 E,b 寄存器链（差 1 cycle）、$S$ 走 2 元素 FIFO（对角线差 2 cycle）→ 全流水。反向 traceback 每步依赖前一步、串行紧依赖，FPGA 低时钟（数百 MHz）下低效——Lembas 用 8×8 tile 位并行 traceback（见"tiled bit-parallel traceback"条目）解决。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CPU 上 ksw2（Minimap2 内嵌）以 SIMD 条带化（striped SW，Farrar 2007）向量化；GPU/FPGA 上以 systolic/wavefront 阵列映射（AGATHA、Logan、PipeBSW 等）。Lembas 实测 extend 48 GCUPS/FPGA（双 FPGA 96 GCUPS，mm64 为 27.21 GCUPS，≈4×）；W=2048 时 traceback 开销比次优设计（PipeBSW/Li21）低 1.77×；W=512–25K 全范围总延迟最低（图 16/17，对比 Cheng24/Li21/Liao18/Turakhia18/Teng23）。使用场景：任何需要最优局部比对的序列比对/组装（read mapping 精化、重叠检测、数据库搜索）。

涉及论文标题：
- Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration
