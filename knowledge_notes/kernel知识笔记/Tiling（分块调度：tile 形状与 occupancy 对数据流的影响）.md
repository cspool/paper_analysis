## Tiling（分块调度：tile 形状与 occupancy 对数据流的影响）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tiling（分块）是把无法整块驻留片上 SRAM 的稀疏矩阵按块（tile）切分执行，块形状 (T_M,T_K,T_N)、块占用（tile occupancy，SRAM 中同时驻留的 tile 规模）与块间遍历顺序共同决定数据复用、buffer 压力与 psum 行为。Harmonia 的核心洞察（Insight 1）是：inter-tile 参数会从根本上重塑 intra-tile 数据流的相对性能，因此 tile 形状与数据流必须联合优化而非分层孤立。论文用 16×16 PE 阵列、16KB 本地 buffer、1MB SRAM 的 cycle-accurate 模拟验证：(a) tile 形状改变最优数据流——ResNet-0.1 与 Llama-0.2 负载下，保持操作量不变改变 (64×K×N)，K 小 N 大时 OutP 最优，K 大时 OutP 因 buffer 溢出性能骤降而 InP/Row 受益于更高 K 的复用与 PE 并行度，(64,128,64) 使 Row 优于 OutP；(b) tile occupancy 重塑 SRAM 访问——从 16×16（1×PE 阵列）到 256×256（256×）缩放，OutP 最早到最小流量但对超大 tile 因 psum 溢出变差，InP 受益于大 K 但超大 tile 增加重载成本，Row 最容忍大 tile（逐行处理）但极端大 tile 放大 B 冗余访问。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Harmonia 静态层选块过程（Algorithm 1）：候选块数 n_M=⌈M/T_M⌉、n_K=⌈K/T_K⌉、n_N=⌈N/T_N⌉；独立近似下 E[nnz_A]=ρA·T_M·T_K、E[nnz_B]=ρB·T_K·T_N；可行性约束 (2)：s_val(E[nnz_A]+E[nnz_B])+s_psum·T_M·T_N ≤ 0.8·S_SRAM（留出索引/元数据/运行时变化余量）；选取最大 OI=OPs/Bytes 的 (T_M,T_K,T_N)。动态层按 tile 实际密度细化：低密度扩张 tile 提升 PE 利用率与复用、高密度收缩防 psum spill 降 merge 深度、聚类稀疏时把 tile 边界对齐非零簇防局部热点。微重切块（micro-retiling）只在块内进行：spill/SRAM pressure/深 merge 背压触发收缩（降 K 降 merge 深度、限 M/N 降 buffer/DN 负载），持续低密度无异常时扩张提复用。遍历顺序启发式：M,N 小（C 可驻留）时用 k-outer 保持 C 驻留、流式 A/B 块；K 小则强调 M/N 复用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
既有方案对照：Tailors 用离线平均稀疏度静态 overbooking 分块；DRT/HARP 运行时重构 tile 但控制开销高；HYTE 用静态分析+运行时细化调整 tile 边界但数据流固定。Harmonia 的差异：静态层保证所有候选块在"合理稀疏模式"下 buffer 可行（不依赖逐 tile nnz），在线层只改静态层显著偏差的维度（轻量、只改 tile 切分与分发逻辑、无硬件重构），动态层以反馈为据做块内微调——三层 tiling 决策逐级细化。论文未提供开源实现；评估用 bcsstk10.mtx、email.mtx、orani678、rajat19（SuiteSparse）与剪枝 DNN 权重。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication
