## 带宽利用中心数据流（Bandwidth-Utilization-Centric Dataflow：tile 分组 / plane-aware 布局 / 输出输入驻留 / GEMV 配对）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 带宽利用中心数据流是 SHyLA 的运行时算子映射与数据布局策略，目标是在 3D 堆叠混合内存（NVM 存 Weight/KVCache、DRAM 存 IA）的放置约束下最大化 areal bandwidth 利用，避免 2D/2.5D 风格访问模式在 3D 堆叠下带宽利用不足。组成：(1) memory-plane-aware 数据布局——tile group 内 NVM plane 存其 NTile 与同组 DTile 的 Weight 均匀切块、DRAM plane 存 IA 行，防 plane starvation 与 compute-tile-memory-plane 失配；(2) tile 分组——#DTile<#NTile 时每组 1 个 DTile + 若干 NTile（同相对位置配对），组内专用高速链路、跨组 AXI fabric；(3) intra-chiplet 输出驻留+输入驻留（output-stationary with input-stationary）强调 Weight 读复用、最小化 IA 流量，tiling 因子 B_I/B_K 由片上 buffer 容量决定；(4) decode GEMV 配对 + KVCache 单 plane 放置；(5) GQA attention-group/request 级并行。仿真标定达到 DRAM 读/写 90%、NVM 读 70%、NVM 写 10% 的带宽利用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 以 FFN GEMM（IA∈DRAM、Weight∈NVM、B_I×B_K tiling）为例的调度伪代码：
```
for i_batch in range(0, I, B_I):               # IA 行块（DRAM 全局输入 buffer）
    load IA_rows[i_batch] -> global_in_buf     # DRAM 读，双缓冲与计算重叠
    for k_block in range(0, K, B_K):           # Weight 列块（NVM）
        load W[:, k_block] -> tile_local_buf   # NVM 读（每 NVM plane 存 NTile+组内 DTile 切块）
        compute C[i_batch, k_block] = IA_rows[i_batch] @ W[:, k_block]   # tile MAC 阵列，output/input-stationary
        write C -> DRAM（或片间 all-reduce 于 Attention Output/FFN2 输出）
    # decode 阶段 ATTN：每 tile 一对 GEMV（QK^T→SV 连续、QK^T 中间结果不写 DRAM），KVCache 单 plane 本地访问
```
时间模型即解析模型 Eq.(2)：IA 批读占 DRAM 读带宽、每批内 Weight 块占 NVM 读带宽、IA 写回占 DRAM 写带宽。消融（SHyLA-D 无 tile 分组/细粒度切分、one-to-one NTile-DTile 配对）显示该数据流贡献 1.35× geomean 系统吞吐提升（更高带宽利用、更低 NVM 访问开销）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 GPGPU-Sim（https://github.com/gpgpu-sim/gpgpu-sim_distribution）中以 CUDA 实现：CUDA 管理片上双缓冲与 plane-aware tile 映射，GPGPU-Sim 内存控制器地址映射被修改为把连续地址跨 channel 分布（提高带宽利用）、channel 数按 CACTI-3DD 推导带宽配置、DRAM/PCM 时序按 workload 配置。SHyLA 本体（CUDA 映射/解析模型）未开源（联网未找到仓库）。使用流程：输入 LLM 层算子形状与混合内存配置 → GPGPU-Sim 逐周期模拟 tile 计算与 Weight 加载重叠、按 plane 分布的内存请求 → 输出每层执行周期与带宽利用 → 汇总系统 token 吞吐。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
