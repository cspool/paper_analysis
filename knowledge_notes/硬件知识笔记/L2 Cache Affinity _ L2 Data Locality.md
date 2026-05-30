## L2 Cache Affinity / L2 Data Locality

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

L2 Cache Affinity（L2 缓存亲和性 / L2 数据局部性）是指 GPU 上不同 SM 执行 tile 时通过共享 L2 cache 复用数据的能力。GPU 的所有 SM 共享统一的 L2 cache，当多个 SM 在同一 wave 中访问重叠的数据区域时，后续 SM 的请求可以直接从 L2 cache 命中，避免昂贵的 DRAM 读取。HyTiS 通过两机制优化 L2 cache affinity：(1) tile layout scheduling——选择合适的 Group-M/Group-N 布局使同一 wave 内相邻 SM 访问数据有最大重叠；(2) wave 粒度分析——量化每一 wave 的 DRAM→L2 流量 V_i 并最小化 total V_tol。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

H100 GPU 硬件架构中 L2 cache 的运转（HyTiS 场景）：
- L2 cache 容量 50MB（H100），132 SM 共享
- GEMM 执行流程：SM 通过 TMA（Hopper）或 ldmatrix（Ampere）请求 A/B 矩阵 tile → L2 cache 检查 → cache hit 直接返回 → cache miss 通过 memory controller 从 HBM 读取
- 同一 wave 内相邻 SM 访问的数据重叠：若 SM_0 处理 tile (m=0, n=0)、SM_1 处理 tile (m=0, n=1)，则两者需要的 A 矩阵行 (m=0) 完全相同——SM_1 可从 L2 复用 SM_0 已加载的 A 数据
- Column-major layout：沿 M 维相邻的 SM 共享 A 行数据
- Row-major layout：沿 N 维相邻的 SM 共享 B 列数据
- 第一 wave 最关键：L2 初始为空，所有访问 DRAM；后续 wave 可复用前 wave 缓存数据
- 量化指标：Nsight Compute dram_bytes_read.sum 测量 DRAM→L2 流量

DRAM read 量差异实测：同一 GEMM 在不同 tile layout 下 V 差异最高 64%；HyTiS 自适应 layout vs fixed group-M(s=8) 将 H100 上 low DRAM read 区从 46% 降至 20%，high 区从 15% 升至 28%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

优化 L2 cache affinity 的一般方法：(1) tile layout selection——HyTiS 分析模型选择 GM/GN 布局和 group size；(2) CTA scheduling——CCDS 通过 predictor 估算 peer SM 缓存块，减少 L1 miss；(3) LLC-bypass management——SMILE 将部分 LLC 作为扩展 SMEM 管理；(4) column-major in MoE——Adnan Hoque et al. 采用 column-major tile layout 改善 MoE kernel L2 数据局部性；(5) wave-granularity modeling——HyTiS 是第一篇在 wave 粒度建模和优化 L2 cache affinity 的工作。在 GEMM 场景，可以通过改变 tile dispatch 顺序（即 tile layout）以几乎零开销的方式获得显著 L2 cache affinity 改善。

涉及论文标题：
- HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality
