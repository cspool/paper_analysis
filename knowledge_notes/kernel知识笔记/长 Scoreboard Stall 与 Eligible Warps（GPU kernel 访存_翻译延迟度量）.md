## 长 Scoreboard Stall 与 Eligible Warps（GPU kernel 访存/翻译延迟度量）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Nsight Compute 的 SM 调度度量：long-scoreboard stall = warp 等待长延迟内存操作（TLB miss、页走查、全局访存）的 stall 占比，值越高说明可用算术不足以掩盖访存延迟；eligible warps per active cycle = 每活跃周期可发射的 warp 数，衡量调度器可用并行度。两者配合定位 kernel 是"延迟受限"还是"吞吐受限"：stall 高 + eligible 低 ⇒ 内存系统（含地址翻译）是关键路径；stall 低 + 吞吐低 ⇒ 指令供应/计算问题。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ConServe 的用法（A100，Llama-3-8B，8K prefill+1K decode，batch=8）：FlashInfer-paged 长 scoreboard stall 84.64%、eligible warps 0.718；native 79.37%、0.825。解释链：散页 KV 布局 → TLB 工作集超容量、页走查增多、页走查缓存（PTE cache）复用下降 → 更多 warp 挂在翻译上 → 可发射 warp 变少 → 每周期发出的访存/计算指令变少 → SM/L2/DRAM 吞吐 −22.4%/−16.7%/−21.1%。即用这两个指标把"布局差异"归因到"地址翻译延迟"，支撑 contiguity-preserving 设计动机。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Nsight Compute（ncu）profiler 采集（论文引用 docs.nvidia.com/nsight-compute/NsightCompute/index.html）。使用：对比 kernel 实现、验证访存布局优化（如 KV 连续化）的微架构效果；配合 SM/L2/DRAM 吞吐、TLB/页走查统计使用。局限：百分比指标必须在同配置下对比才有意义——ConServe 强调两组配置同上下文长度、同逐 token 算数、仅 KV 放置不同。
- HyperDrive 补充视角（ISCA'26，FHE NTT kernel 的 warp stall 分类与应用）：用 Nsight Compute 的 warp state 分类定位 FHE kernel 的访存瓶颈，六类状态——Stall Long Scoreboard（长延迟数据依赖，如 GMEM 访问）、Stall LG Throttle（global/local 访存指令队列满，GMEM 访问密集）、Stall Short Scoreboard（短延迟 I/O 依赖，如 SMEM 访问）、Stall MIO Throttle（内存 I/O 指令队列满，SMEM 访问密集）、Compute（等待功能单元完成）、Eligible（就绪等待发射）。量化结果（N=2^16、36 limbs）：baseline FP64-TCU NTT 的 Inner-NTT stage1/2 中 SMEM 相关 stall 占 32%/48%、总 stall 占比 77%/72%；Outer-NTT stage1/2 的 GMEM 相关 stall 占 45.9%/22.6%；跨多项式 kernel 的 stall long scoreboard 占 BConv 60.6%、IP 74.6%。TLMOP 使 SMEM stall -33.2%、scheduler stall cycle -44.2%（加 TransOP 累计 -50%）、occupancy 55.0%→76.8%；TFOP+RowMaj 使 GMEM stall -59.5%、scheduler stall -39.3%。用法：对每个 kernel 分别统计 warp state 占比并叠加消融配置，把"stall 种类"归因到"具体内存优化"。
- MNEMOS 补充视角（ISCA'26，TFHE PBS kernel 的 stall 分解与优化归因）：对 PBS 最耗时 kernel（盲旋转）的 stall 统计显示 baseline 中 stall_long_scoreboard 占 >50%（多数参数集），与 stall_MIO_throttle（共享内存/L1 操作争用）合计超 60%，证明 PBS 为 memory-bound；根因是 BSK 作为热数据被多 SM 并发访问、仅 L2 级复用（GMEM→L2 与 L2→SM 流量失衡）。优化后（BSK 分块复用 + Tensor Core FFT + 跨迭代融合 + 片上 Fourier 矩阵生成 + swizzle）stall_long_scoreboard 降至约 20%（Para-E 下 <15%），主导 stall 变为 Stall Math Pipe（compute-bound）；FFT kernel 的 stall_MIO_throttle 延迟降 3.2×/2.9×/1.8×（N̄=256/512/1024）；GMEM→L2 流量平均降 15.7%、L2→SM 降 69.4%。用法：以"stall 种类占比变化"作为内存优化是否生效的判据——memory stall 让位于 math pipe stall 即说明访存瓶颈已解除。

涉及论文标题：
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
