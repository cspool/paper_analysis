## Scale-Value 交织布局（Interleaving Block / itBlock，NDP 的量化矩阵 DRAM 布局）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Scale-Value 交织布局是 FlexQ-NDP 针对分组低比特 FP 的 DRAM 数据布局：把多个 QGroup 的 scale 区域与对应 value 区域打包成"交织块"（itBlock），scale 区在前、value 区在后连续排布，取代传统 GPU/NDP 中 scale 与 value 分离的连续存储（Triton、Cost-Effective 惯例）。动机：细粒度分组下 scale 数据量不可忽略，计算中频繁在 value 区与 scale 区间切换 DRAM 行——scale 相关额外延迟 75% 来自行切换、仅 25% 来自 scale 访问本身。交织块设计两目的：① 小组尺寸下单个 QGroup 的 scale 填不满一个 DRAM 列（最小访问粒度），多 QGroup scale 拼成一区提高列利用率；② scale 与 value 相邻存放减少交替访问时的行切换。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
属于编译流程的 DRAM Mapping 步骤，三步构造（权重矩阵 W(M,K)）：
1) 交织比 Ratio = Size(ValueRegion)/Size(ScaleRegion) = Prec.(Value)·G_M·G_K / Prec.(Scale)（Eq.1）——按 QConfig 自适应；
2) 交织步长：scale 区列数 Col_S 受 scale 缓冲约束（编译空间可探索项），itBlock 内 QGroup 数 #QGroup = Col_S×len/Prec.(Scale)（Eq.2，len 为 DRAM 列位宽）；沿 K 的 QGroup 数 #QGroup_K = ⌈K_Tile/G_K⌉ 对齐循环切块（Eq.3）；跨 tile 边界取整 #QGroup' = #QGroup_K×⌊#QGroup/#QGroup_K⌋（Eq.4），value/scale 实际列数由 Eq.5/6 计算；
3) 物理映射：按内层循环迭代序选 QGroup 进 itBlock（保证 tiled 循环数据集中存放、减少行切换），计算 itBlock id 与块内 id（Eq.7）→ 逻辑列号（Eq.8）→ 物理行列号（Eq.9）。
运转例子：QGroup(1,16)、K_Tile=4 个 QGroup、Col_S=2（每列存 4 个 FP8 scale，共 8 QGroup/itBlock）→ 每 itBlock 含 scale 区（2 列）+ 8 QGroup 的 value 区 → 内层循环遍历 K 时只在一个 itBlock 内顺序扫 value、scale 预取在前，跨行切换大幅减少。仅对权重矩阵离线排布，避免运行时重排中间结果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译器 DRAM 映射 pass 中的纯离线重排（Eq.1–9 的整数运算），映射结果直接编入指令中的行列地址；Col_S 与 K_Tile 由 DSE 决定。使用：与循环 tiling（决定 #QGroup_K）和缓冲分配（决定 Col_S）强耦合——buffer refill 频率决定交织步长；效果：DRAM 行切换平均降约 2×（消融 ×1.36），且使 FlexQ-NDP 在 8×32B 缓冲下接近最优（baseline 需 >20×32B）。适用场景：任何"细粒度元数据 + 数据"混合访存的 NDP/GPU 布局问题（如分组量化的 scale、稀疏格式的 index）。

涉及论文标题：
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
