# 实验_Serving调度

## Approaching Shannon Bound with Lossless LLM Weight Compression

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：修改开源 serving 框架 SGLang，把其 dense 矩阵乘（投影层）默认的 CUTLASS GEMM kernel 替换为本文 ANS-enabled GEMM backend（plugin 式投影算子，离线压缩权重直接作为 drop-in 权重加载），使权重静态内存大幅缩小，在同一 GPU 显存预算内腾出容量给 KV-cache，从而支撑更大 batch、提升吞吐；Mixtral-176B 以专家并行（EP）部署于 4×A100。论文不改调度算法本身，而是通过压缩权重 footprint 扩大调度器可用的内存预算（batch 上限由显存决定）。
  - 实验比较：SGLang 默认（未压缩权重 + CUTLASS GEMM）vs 本文（压缩权重 + ANS GEMM backend），固定显存预算（Qwen-14B：80 GB 单卡；Mixtral-176B：320 GB 四卡），序列长度 1024/2048。指标：权重/KV/总内存分解、最大可行 batch size、吞吐 tokens/s、median TPOT。
- 硬件平台是什么，配置是什么。
  - 8× NVIDIA A100 80 GB（HBM2e 2 TB/s）服务器（SGLang 端到端）；NVIDIA Hopper H200 用于 kernel 级对比。PyTorch 2.5.1、CUDA 12.1，吞吐为 SGLang batching scheduler 下实测执行时间。
- 开源Serving框架是什么。修改了什么。
  - SGLang（论文引用 [50]，https://github.com/sgl-project/sglang）。修改点：dense 矩阵乘（W_Q/W_K/W_V/FFN 等投影算子）的默认 CUTLASS GEMM 替换为 fused rANS 解压 + GEMM 后端；权重以"tile 压缩 bitstream + 4B/tile offset 表 + 每层共享 codebook"形式加载。多 GPU：Mixtral-176B 用 EP 跨 4 张 A100。调度算法本身论文未说明有修改。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文自身 SGLang 集成代码未找到公开仓库（arXiv 2606.15789，无官方 repo）；SGLang/CUTLASS/DietGPU 均开源（链接见上）。
  - 全过程（一个 decode 请求）：SGLang 前端接收请求 → RadixAttention/调度器按剩余显存决定可加入的 batch 上限并组批（Qwen-14B 权重 27.5→18.1 GB，释放约 9.4 GB 显存给 KV-cache，batch 上限随之提高）→ 执行投影层时调用 ANS-enabled GEMM kernel（不再先整层解压）→ kernel 内 warp 0 从全局内存按 offset 表取压缩 tile、rANS 解码进 shared memory，其余 warp 用 tensor core 与激活 tile 做 GEMM，双缓冲流水重叠 → 输出激活、KV-cache 增长由调度器记账 → 返回 token。效果：Qwen-14B（seq 1024）最大 batch 47→60（Table II，1.3×；摘要与 C 节正文写作 47→75）、吞吐 1131→1217 tokens/s（1.1×）；seq 2048：23→30（1.3×）、548→651 tokens/s（1.2×）。Mixtral-176B（4×A100，seq 1024）batch 20→95（4.8×）、吞吐 241→391 tokens/s（1.6×）；seq 2048：10→47（4.7×）、190→257 tokens/s（1.4×）。median TPOT 因解压开销略有上升（Qwen-14B 71→81 ms、112→125 ms），以轻微延迟换吞吐。kernel 级对比（H200）：vs NeuZip 吞吐最高 ~10×、vs DFloat11 ~6–7×（摘要称最高 11×）。

## Breaking Barriers in Atomic Scaling: A Hardware–Software-Collaborated Framework to Deconstruct RDMA Atomic (Fusa)

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Fusa 框架在 Mellanox RNIC 用户态驱动层（mlx5 驱动）实现多请求分发调度——拦截并重写 RDMA Atomic（CAS/FAA），按"细粒度争用感知分发策略"把请求调度到两条执行路径：无争用请求走 RNIC 内部锁定表（硬件路径），热点争用请求经 Fusa-RPC 选择性卸载到服务端 CPU 线程（软件路径）。调度单元从锁定表槽细分为 group（按地址多取 g 位，默认 8,192 组），每组 64-bit 计数器 + 1-bit 分发位；以 watermark（全组平均请求数）判定争用组，按请求数降序扫描并卸载最热组，直到超出服务端 CPU 处理容量 C。策略切换由客户端 lazy 同步（multi-QP epoch + Wait_sync，图 9 伪代码）与服务端 consensus（新旧策略 XOR 一致位，变更组 reject + 客户端自动重传）保证一致性；OrderedFusa 变体在 WRITE 后追加 RDMA WAIT verb，保证 per-QP 线性化。
  - 实验比较：RNIC-Only（全部 RDMA Atomic 走 RNIC 锁定表）、HERD（HERD RPC 全量卸载到服务端 CPU）、Static（随机一半 CPU 一半 RNIC）、Fusa、OrderedFusa。指标：原子吞吐（Mops/s）、平均/P50/P99 延迟。
- 硬件平台是什么，配置是什么。
  - 5 节点集群：每节点 2× 2.4 GHz Intel Xeon Silver 4314（共 32 核）、256 GB DRAM、100 Gbps Mellanox ConnectX-6 InfiniBand RNIC，经 100 Gbps Mellanox SN2700 交换机互连；Ubuntu 20.04 LTS、Linux 内核 5.4.0、MLNX OFED v24.10-2.1.8、2 MB 大页。1 server + 4 clients；server 端 4 个 RPC 线程绑核。另在 Intel Xeon Gold 5420（Sapphire Rapids）与 AMD EPYC 7281 平台复验 PCIe Atomic 结论。
- 开源Serving框架是什么。修改了什么。
  - 框架：Mellanox 用户态驱动（rdma-core / libibverbs，https://github.com/linux-rdma/rdma-core）。修改 mlx5_create_qp()（为每 QP 分配元数据内存区）与 mlx5_post_send()（按分发策略拦截并重写请求），并用 LD_PRELOAD 覆盖 libibverbs API，上层应用零修改。新增 Fusa-Driver（拦截分发）、Fusa-RPC（coroutine-friendly RPC，移植自 HERD RPC）、Fusa-Agent + Fusa-SHM（65 KB 共享内存维护 8,192 组计数/策略位）、Fusa-Server（聚合统计、计算并广播策略）。论文开源：https://github.com/xmusys/fusa（已确认：rdma-core fork，改动限于 providers/mlx5/，含 recorder.{c,h} 与 fusioncas 实验框架、run_ycsb.py）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 全过程（一个 RDMA CAS 请求）：应用经 libibverbs 调 ibv_post_send → Fusa-Driver 在 mlx5_post_send 拦截：取 qp_id、group_id = address % 8192，计数器++、置 QP running、epoch++，读该组 strategy bit → bit=0：inflight[group]++，原样发 one-sided CAS 给 RNIC，RNIC 按地址哈希取锁定表槽、PU 经 PCIe RMW（Read 旧值 + Write 新值）完成，CQE 返回后驱动从 WR_ID 提取 13-bit group_id 递减 inflight → bit=1：Fusa-RPC 把 CAS 转成 RPC（客户端 post RECV + WRITE 到 server 请求缓冲，控制权立即交还应用，异步等 SEND 的 CQE）→ server CPU 线程出队解析、在 CPU 上执行原子 → 结果经 SEND 返回。每 1 s 一个 stage：Fusa-Agent 上报统计 → Fusa-Server 聚合、按 watermark + 容量 C 计算新策略并广播 → 客户端 CAS 切换策略指针、Wait_sync 等全部 QP epoch 推进或退出 running；CPU→RNIC 切换时 server 直接 reject 在途请求（客户端检测 reject 标志自动重传），RNIC→CPU 切换时等 inflight=0。开销：ibv_post_send 123→141 ns（+18 ns）；策略切换共识仅 48 µs（push 10 µs + 本地应用 10 µs + Wait_sync 28 µs）。结果：YCSB 微基准吞吐最高 4.6×（Exp#2 各更新比下 Fusa 平均 2.8×、OrderedFusa 2.0×；U10R90 下 43.9 vs 14.8 Mops/s）；RACE 上最高 14.7×、P99 延迟 -97.8%；DrTM 上最高 7.1×、P99 -89.0%。

## CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM（近似层次匹配：调度优化多请求并行，但未修改开源 Serving 框架，运行于自研模拟器）

- 属于Serving调度的实现是什么？实验比较什么？
  - 近似匹配（论文是软硬件协同 AFD 系统，未修改任何开源 Serving 框架、无真实部署，调度实现在模拟环境中；最接近"Serving 调度"中"优化多请求调度提高吞吐量"的定义）。实现 = CHIME-sys 的 alignment-predicting scheduling：对 GPU 与 CHIME-PIM 两侧运算延迟建模预测，按"预测延迟对齐"原则选择请求组成两个 sub-batch，使两设备的并行执行延迟对齐、空闲气泡最小。调度策略：1) 每个 sub-batch 先加 1 个 prefilling 请求（无 prefill 则跳过）；2) 每加入一个 prefill 请求（T_GPU 增大）后，向每个 sub-batch 追加 N 个 decoding 请求并按负载均衡分配到各 rank，随即预测 T_PIM 与 T_GPU，若 T_PIM < T_GPU 继续加 N 个 decoding 请求直到 T_PIM > T_GPU；3) 若仍有 prefill 请求则重复直到 PIM 显存耗尽；显存饱和且气泡在 PIM 侧时，把每个 sub-batch 的最后一个 prefill 请求 chunk 化，动态调整 T_GPU 逼近另一 sub-batch 的 T_PIM。N 越大批时间调节粒度越粗、rank 间负载均衡越好：MHA N=1、GQA N=16。延迟建模：T_GPU 用 Random Forest Regression（RFR，增量学习/低延迟/高精度），T_PIM 用线性模型（PIM 执行时间与计算/传输 token 数线性相关，t_d 由最慢 rank 决定）；运行时 profiling 增量更新数据集，预测相对误差 <~1%（中位数 <0.5%）。sub-batch 调度技术本身继承自 NeuPIMs [28]、NEO [30]。
  - 实验比较：调度消融 baseline = 优先填满 CHIME-PIM 容量的调度策略 vs CHIME 的 alignment-predicting scheduling；指标 = 平均吞吐与 Time-between-Tokens (TBT)。结果（OpenR1 trace）：TBT 最高降 70.93% 且吞吐不降略升；MHA 模型下 baseline 随容量增大选更大 batch 导致 TBT 升高而无吞吐收益，CHIME 可避免气泡并抑制 TBT 增长。
- 硬件平台是什么，配置是什么。
  - DGX-A100：GPU 侧 8× NVIDIA A100（每卡 80GB HBM2e，FP16 合计 156 TFLOPs，NVLink 互连）；加速器侧 16 通道 × 2 DIMM（DDR4-3200，2TB）装备本文 DIMM-PIM（CHIME 等效带宽 13.0TB/s；rank-level R-PIM 1.6TB/s），PCIe 连接 GPU 与 CHIME-PIM（DGX-A100 上 4 ranksets）。评估环境为模拟器：AttAcc（GPU roofline 模拟）+ CHIME-PIM-sim（修改版 DRAMSim3，trace-driven cycle 级）。
- 开源Serving框架是什么。修改了什么。
  - 论文未修改开源 Serving 框架（论文未明确说明）。CHIME-sys 继承 NeuPIMs/NEO 的 sub-batch 调度思路，在其自研模拟环境中实现 alignment-predicting 调度（RFR+线性模型+运行时 profiling+chunked prefill 对齐），对 GPU 上的 QKV Gen/投影/FFN 与 PIM 上的 decoding attention 做跨设备并行编排。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未给出 CHIME 源码链接；联网搜索未发现官方仓库（SJTU IPADS 暂未公开），开源状态无法确认。基座组件开源：DRAMSim3（https://github.com/umd-memsys/DRAMSim3）、AttAcc 模拟器（https://github.com/scale-snu/attacc_simulator）。
  - 全过程（一次迭代）：调度器从请求队列按策略选请求组成 sub-batch 0/1 → 每个 sub-batch 的 QKV Generation 在 GPU 批处理 → decoding attention 在 CHIME-PIM 以 rank 粒度执行（t_d 取最慢 rank，跨设备传输 t_comm 与另一 sub-batch 重叠隐藏）、prefill attention 在 GPU 执行 → CHIME 聚合所有请求的 attention 输出经 PCIe 返回 GPU → GPU 批处理投影/FFN 等 FC 操作。对齐原理：RFR 预测 t_p(chunk 数, 已完成 token) 与 t_batch，线性模型预测各 rank t_d 与 t_comm，选择使 sub-batch 0/1 的 T_GPU≈T_PIM 的请求组合，chunked prefill 微调 T_GPU。端到端效果：吞吐较 HBM-PIM 最高 5.15×、HBM-PIM-EXT 3.45×、GPU-only 3.94×、R-PIM 7.21×。

## ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：ConServe —— 面向多轮对话（multi-turn）LLM serving 的 contiguity-preserving 虚拟内存分配器，替换 vLLM/vAttention 类 serving 框架的 KV-cache 内存管理。每个 conversation 独占一段连续虚拟地址（VA）slice（内部按 transformer 层分段 back-to-back、layer-major 布局），物理页用 CUDA VMM 按需映射（2 MB 组粒度），attention kernel 用纯算术 base+offset 寻址（VA(t,l)=base+seg_off[l]+t×B_layer+δ），消除 block table 查表与跨块 gather；slice 弹性增长用 copy-free remap（新 slice 绑同一物理页、KV 数据不搬）与层间计算重叠、迭代边界切换 base，旧 slice 以 PENDING_UNMAP 状态隔离后批量延迟回收（cuMemUnmap + 批量 TLB invalidation）；dual-index free-list（length-ordered 视图做 best-fit O(log M)、address-ordered 视图做相邻区间 eager coalesce，merge flag 标记可合并区间）管理 VA 碎片。默认参数：VA group G=2 MB、初始/扩容 headroom k=8 turns、每轮 token 预算 Q 取 ShareGPT 离线统计的 r=80% 分位数、利用率阈值 θu=0.90、几何增长因子 γ=1.5；连续两轮触发 resize 时临时用最近一轮实测 token 作为 Q 防抖。
  - 实验比较：vLLM（PagedAttention，block size=16 tokens，软件 block table）、vAttention-Turn（每 turn 视为独立 request，跨 turn KV 落在不相邻虚拟区）、vAttention-Conv（整个 conversation 视为一个长期 request，按 max-context 预留 slice）。在线 serving 指标：mean TTFT、p99 TTFT、decode throughput、SLO attainment（SLO=无争用基线 25×TTFT/TPOT）；离线推理指标：端到端吞吐（含 prefill+decode 的 tokens/s）。
- 硬件平台是什么，配置是什么。
  - NVIDIA A100-80GB：Yi-6B 与 Llama-3-8B 单卡；Yi-34B 用 2×A100 NVLink 互连、TP=2。BF16 权重；长上下文推理开启、关闭 sliding-window/truncation 使完整 KV 常驻；CUDA VMM 默认 2 MB 映射粒度。动机量化实验：Llama-3-8B、8K prefill + 1K decode、batch 1–16；多轮场景每轮 512 输入 + 64 decode。
- 开源Serving框架是什么。修改了什么。
  - 框架：vLLM（PagedAttention baseline，https://github.com/vllm-project/vllm，block=16 tokens）；vAttention（ASPLOS 2025，CUDA VMM 管理单请求 KV，其两种多轮适配变体作 baseline）；attention kernel 用 FlashInfer（native 与 paged 两实现，https://github.com/flashinfer-ai/flashinfer）。修改了什么：把 KV-cache 管理从"固定 block 池 + 软件 block table"替换为 conversation 级连续 VA slice + 按需物理映射（进程启动时 cuMemAddressReserve 预留一个大 48-bit VA arena，cuMemCreate 2 MB / cuMemMap / cuMemSetAccess 在 token 边界按需执行、与上一迭代 kernel 重叠）+ 弹性 resize；attention kernel 从 block-table gather 改为 variable-length 模式下从 descriptor（每序列 KV base 指针 + live 长度）加载 base 后连续流式访问。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文（ISCA 2026，作者 Bingyao Li，UC Riverside）未提供 ConServe 源码链接；联网搜索未找到该工作的公开仓库（arXiv 2410.01228 的 ConServe 是另一篇统一在线/离线 serving 论文，非本文），无法确认开源。基座组件开源：vLLM、FlashInfer、vAttention。
  - 全过程（conversation 的新 turn 到达）：调度器把该 turn 加入 micro-batch → runtime 为该 conversation 分配/复用连续 VA slice，prefill 只算新增 token，attention kernel 从 descriptor 加载该序列 base 指针，按 VA(t,l)=base+seg_off[l]+t×B_layer 对历史 KV 连续流式读（无 block table 查表、无跨块边界判断，warp 发对齐连续全局访问合并成大批量事务）→ 每 token 进入 layer 0 时检查利用率 η=U/R，>0.90 触发 resize：按 R_next=max(⌈γR⌉, ⌈T_target^(new)·B_tok/G⌉) 从 length-ordered free-list best-fit 选新区间 → 当前迭代内 layer ℓ 执行完后立即对新 slice 该层段 cuMemMap/cuMemSetAccess 绑同一物理页（与剩余层计算重叠，重叠窗口约 (L−1)/L·T_iteration）→ 迭代边界更新 descriptor 的 base 与 seg_off，旧 slice 置 PENDING_UNMAP 隔离 → 空闲或 free-list 不足时批量 cuMemUnmap + TLB invalidation，回收区间与邻居合并回 free-list。物理页分配：后台线程在迭代 i−1 运行期间监控上下文长度、预判迭代 i 需求并映射下一组 2 MB 页；conversation 结束时的空闲页组保留并直接重分配给新 conversation。效果（ShareGPT，A100）：TTFT 较 vLLM 低 64.1%–74.4%、较 vAttention-Turn 低 31.4%–43.4%、较 vAttention-Conv 低 8.3%–19.1%；decode 吞吐 +19.4%–25.6% / +6.1%–9.5% / +4.2%–6.1%；SLO attainment +11%–19% / +7%–9% / +3.2%–4.9%；离线端到端吞吐 +17.7%–35.1% / +8.6%–15.6% / +7.2%–12.1%（batch 1–32，增益随 batch 与模型 KV footprint 增大）。Qwen-Bailian 生产 trace（43,058 turns、约 2 小时、平均 2 turns/conversation）：TTFT −36.6%/−23.8%/−10.9%、p99 TTFT −30.3%/−19.5%/−8.3%、decode 吞吐 +19%/+13.6%/+3.4%。LongBench v2 长对话 trace（每对话截断到 8K–32K token）：各 batch 下均最优。

## DynoPipe: Heterogeneous Edge-Cloud LLM Serving with Dynamically Orchestrated Pipeline Boundaries（近似层次匹配：边云 serving 系统，自研实现、论文未明确说明修改开源 Serving 框架）

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：DynoPipe —— 面向异构边云连续体的自适应 LLM serving 系统，用动态 pipeline parallelism + 可移动计算边界（split point）跨边云编排推理。三层协同：(1) Boundary-constrained pipeline construction（§4.1）：把边云划分建模为边界约束优化（min max{T_exec(l_i) + I_boundary(l_i)·T_boundary(l_i)}，Eq.1），DP 求解（Eq.2）预计算 3-5 个边界配置组合（bandwidth/compute/memory-constrained），split point 只放在完整 transformer block 之间（残差在块内解析，跨域只传 fully-resolved hidden state，无跨域残差同步）；组合规模上界 |K|≤min(资源 regime 数, 层数)，LLaMA 类均匀架构实测 |K|=4 即可覆盖全部最优 split point（{4,8,12,16}）；(2) Proactive multi-configuration orchestration（§4.2）：LRP（Latency-Regulated Placement，Algorithm 1）算法，500ms 采样的轻量遥测（带宽/GPU 利用率/内存压力），按触发阈值（bandwidth<τ_bw→activation-minimal、edge_load>τ_compute→early-cloud、memory_pressure>90%→memory-aware）从预计算组合选边界，hysteresis δ=15-20% + cooldown 防振荡，决策延迟 sub-ms；(3) Hierarchical state management（§4.3）：pipeline 状态按迁移关键度分三层（KV cache/中间激活/元数据），参数重叠缓存 + 带宽感知状态分区（利用下行富余带宽异步流式）+ 差分 KV cache 传输，带宽<1Gbps 时切 adaptive recomputation fallback（计算开销 +15-25%、带宽 -90%），迁移开销从秒级降到毫秒级（P99 迁移 85ms）。
  - 实验比较：DynoPipe vs FlexNN（edge-only 静态层分配、内存约束）、EdgeShard（固定边云划分 + 离线优化）、Cloud-only（SP=0）、Edge-only（SP=32）；受控负载（QPS=2/4/8/16/24）+ MAF 生产 trace + 并发多任务（2-4 invokers）。指标：TTFT、token 吞吐、E2E 延迟（mean/P50/P99）、TPOT。结果：LLaMA3.1-8B 下 TTFT 较 FlexNN 降 98.5%、吞吐 4.4×；LLaMA2-7B 吞吐较 edge-only 10.1×、较 cloud-only 1.6×（SP=12 处 3.43 vs 2.09 rps、E2E 292 vs 478ms）；MAF 下 P99 较 CloudOnly/FlexNN/EdgeShard 降 54%/60%/16%（LLaMA2-7B）；支持 128K+ context、16+ 并发请求、70B 模型（LLaMA3-70B 830 tok/s、TPOT 160ms）。
- 硬件平台是什么，配置是什么。
  - Cloud：12 台服务器、16× NVIDIA A40（48GB HBM2 each），100 Gbps RDMA InfiniBand 互连，每台 Ubuntu 22.04 + 512GB DDR4。Edge：NVIDIA RTX 3090（24GB GDDR6X）地理分布式，经共享 10 Gbps 上行链路连云端（实测 RTT 5-50ms），edge-to-edge 1 Gbps LAN（RTT<3ms）。动机实验另提 H100（989 TFLOPS）vs NVIDIA Thor（2070 TOPS）、edge DRAM（64GB/51.2GB/s）vs cloud HBM（140GB/3.35TB/s）等不对称对比。
- 开源Serving框架是什么。修改了什么。
  - 论文未明确说明基于/修改哪个开源 Serving 框架（自研系统，未提及 vLLM/SGLang 等集成；论文未说明）。修改/新增内容：边界感知 pipeline 构建（离线 profiling 生成 <30KB/模型的 per-layer 执行时间 + 激活大小查找表，组合生成约 3 min/设备对）、LRP 动态编排器（遥测监控 + 组合切换 + 触发阈值 + hysteresis/cooldown）、层次化状态管理（三 tier 状态分解、L1 GPU 热边界缓存 ~150MB、L2 host-RAM ~500MB、差分 KV 同步、recomputation fallback）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：未找到 DynoPipe 官方仓库（ISCA 2026，作者 SIAT CAS / UCSD / University of Macau；联网搜索未发现公开 repo，开源状态无法确认）；注意与 NeurIPS 2025 的 DynaPipe（同构 GPU 集群动态层重分配，非同一项目）区分。Baseline EdgeShard 有 arXiv 链接：https://arxiv.org/abs/2405.14371v1。
  - 全过程（一个 LLaMA2-7B 推理请求，split point SP=12，QPS=5）：请求到达边缘 → RTX 3090 执行 embedding + 前 12 层（隐私敏感部分留边缘），遥测每 500ms 采样带宽/负载/内存 → LRP 的 SelectBoundary 按当前瓶颈从预计算组合选目标 SP（网络自由 QPS=5 最优 SP=12；网络争用下 SP=8）→ 边界 stage 把 fully-resolved hidden state 激活张量经 10 Gbps uplink 传云端（优先 attention sparsity/quantization 压缩点，图 7）→ 云端 A40 执行 layer 13-32 → 需迁移边界时：差分 KV cache 传输（P99 72ms）+ 参数重叠缓存 re-staging（13ms），带宽<1Gbps 时 recomputation fallback（<120ms）→ 云端返回 token 到边缘。pipeline 使边缘（96ms）与云端（112ms）阶段并行，容量从 cloud-only 单 GPU ~5.6 rps 提到 ~8.9 rps，排队占比从 62%（295/478ms）降到 26%（76/292ms）。

## Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference（近似层次匹配：本层取其请求到达/batch 调度分析，自研事件驱动流量生成器，非修改开源 Serving 框架）

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Raptor 部署下的 batch-size/请求到达调度分析。用事件驱动流量生成器 + batched compute engine 建模：请求按 Poisson 过程到达（基准 110 req/s ≈ 9.5M requests/day，与 Azure OpenAI 公开配额一致，并扫描更高到达率），请求在 compute engine 前排队；每当 engine 空闲，就从队列中取 q 个待处理请求组成 batch size b，恒定 per-batch 延迟 T 下处理 q 个请求耗时 T·q/b，每轮模拟 1s (1000ms)。结论：1ms per-batch 延迟下即使 1000 req/s 到达率最大 batch 也仅 27（平均 5.12），各到达率与 per-batch 延迟（0.1-100ms）下的推荐 batch 均 <32 → 固定 batch=32 作为实际工作点。同时比较两种部署调度的 tok/s/card 与 interactivity (1/TPOT)：minimal-card（能装下权重+KV cache 的最少卡，暴露各基板容量边界）与 iso-card（固定为 3D-DRAM minimal-card 的卡数）；dense 模型用 unified 部署（TP≤8，PP 仅在单卡装不下时加卡，DP 增加并发序列）、MoE 模型用 disaggregated 部署（TP=4 注意力组 + EP 专家池）。
  - 实验比较：不同到达率（110/200/500/1000 req/s）下的 max/avg batch 与稳态队列深度；per-batch 延迟 T（0.1/1/2/30/100ms）下的推荐 batch；batch 32 下各内存基板（XPU+SRAM 150TB/s/4GB、XPU+HBM 18TB/s/192GB、RP+3D-DRAM 100TB/s/32GB 及 2×/4× BW 与 Full 缩放）在 minimal-card/iso-card 下的 tok/s/card-interactivity 曲线；网络延迟（0.01-10µs）/带宽（32GB/s-4TB/s）敏感性。结果：3D-DRAM 4.71×/2.44× 高于 HBM/SRAM throughput，9.96× 低 TPOT vs HBM，4K 上下文 0.5µs/1TB/s 下 4.38× vs HBM、3.15× vs SRAM；SRAM 因容量小迫使高 TP/PP 对网络最敏感，HBM 因小 TP 度基本不敏感。
- 硬件平台是什么，配置是什么。
  - Raptor 卡（XPU 逻辑 10 PFLOPS + 3D-DRAM 100TB/s/32GB，2-4 MCM/卡、4 chiplet/MCM、D2D 32Gbps/lane、PCIe Gen7/ESUN 网络）vs XPU+SRAM（150TB/s/4GB）vs XPU+HBM（18TB/s/192GB）；网络现实点 0.5µs / 1TB/s；batch=32。模型并行配置见 Table II（如 Llama3.1-70B：SRAM 8TP/1PP/128GB 八卡 vs 3D-DRAM 1TP/1PP/32GB 单卡；DeepSeekV3-671B：3D-DRAM 4 卡 1/32/2 配置 1216GB）。
- 开源Serving框架是什么。修改了什么。
  - 论文未明确说明基于/修改某个开源 Serving 框架（自研事件驱动流量生成器 + batched compute engine 分析模型；上下文引用 PagedAttention [35]、Orca [89]、DistServe [92]、MegaScale-Infer [93] 的调度/并行概念，但未声称修改它们）。模型化为：请求到达（Poisson）→ 排队 → engine 空闲时成批 → 处理（batched compute engine 给出 tok/s/card 与 TPOT）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未提供流量生成器/分析模型源码，联网无法确认（商业产品论文）；评估模型（Llama-3.1-70B、DeepSeek-V3、GPT-OSS、Kimi K2、Whisper、Canary-1B）公开。
  - 全过程（110 req/s，1ms per-batch 延迟，Raptor 3D-DRAM）：请求按 Poisson 到达排入 engine 前队列 → engine 空闲时取 q 个请求成 batch（110 req/s 下 max=1，即实际串行服务）→ 每 batch 经解码流水线：各 slice 的 TE 经 3D-DRAM channel 流式读权重/KV tile（stream blocking 128B flit + stream flipping 翻转）执行注意力与 FFN GEMV → 层内/层间按并行配置做 collectives（dense unified：每层 2 次 all-reduce，TP 越低量越小；MoE disaggregated：注意力 all-to-all ~16KB/card + all-reduce + dispatch/combine many-to-many 数百 KB 至 MB/卡）→ 产出 token，累计 tok/s/card 与 TPOT。作用：证明在 1ms 级 batch 延迟下 32 的 batch 已足够（3D-DRAM 高带宽使 decode 以低 batch 即达高吞吐），并量化内存基板 + 并行度对吞吐/交互性/网络敏感性的影响。

## HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding（近似层次匹配：调度实现在自研事件驱动模拟器 + 流片原型上，未修改开源 Serving 框架；采用 vLLM 的 benchmark_serving.py 生成工作负载）

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现为三套面向动态在线 serving 的异构调度策略（在事件驱动模拟器/原型上实现，非修改开源框架）：(1) 异步 batching——把连续批处理 [73] 扩展到异构架构：每请求分解为 prefill/decode/verification 三类任务，XPU 与 HB 栈各维护 task pool（存任务类型/上下文状态/token 数等元数据），空闲单元立即从池中组批执行、忙碌单元延迟到当前迭代结束；memory 侧用 watermark 机制（内存利用率超阈值时挂起 prefill 任务以限制 KV cache 增长），compute 侧 HB 栈 decode 联合批（共享权重、仅 attention KV cache 按请求区分）、XPU 上 target prefill 与 verification 联合批（draft prefill 因参数不同单独执行、开销可忽略）；(2) utilization-aware speculation（Algorithm 1）——用 SVR 拟合的查找表 T 记录各 draft budget 下最优 tree width 拐点 p，运行时按 HB 栈/XPU 的实际算术强度（每迭代处理的 token 数）相对各自 roofline 动态调 draft tree width（[1,p]）与下一轮 draft budget（[1,B]，B=32），在提高利用率的同时避免在拒绝 draft 上浪费计算；(3) prefill-verification arbitration——PFS（Prefill-First Scheduling）优先 prefill 任务以扩大可批请求池，CHK 把长 prefill 按序列维 chunk 化（需调整 attention mask 保持序列对齐）以匹配 XPU 计算-内存比；XPU 批类型分 verification-only / prefill-only / mixed 三类。
  - 实验比较：动态投机 vs 固定 budget 投机（平均 1.81× 加速，随请求率升高下降，长序列更敏感）；batching 消融（PFS vs FIFO 平均 1.10×，PFS+CHK vs FIFO 平均 1.29×，增益随请求率增长）；FIFO vs PFS 的 TTFT-TPOT 权衡（FIFO 不抢占 verification、TPOT 略好但 TTFT 明显更差，高请求率下 FIFO 把新请求堵在 verification 后面、减少可批请求数，PFS 反超）；不同请求率 1/2/3/4 per device 下平均 draft_budget/tree_width 从 (30.74,3.72) 降至 (9.25,1.58)。
- 硬件平台是什么，配置是什么。
  - 评估载体为事件驱动模拟器（HB 栈 silicon-derived 参数 + XPU 分段线性性能模型）；硬件配置同硬件架构条目：HB 栈 408mm²/10GB/4TB/s/400MHz/4×80×64 MAC，XPU GA100 + 512GB 1.1TB/s LPDDR5X；baseline GPU 为 A800-80GB-PCIe（vLLM）。模型对：Llama2-13B/TinyLlama1.1B、Qwen3-32B/Qwen3-1.7B、OPT-66B/OPT-2.7B，部署于 1/2/4 设备（数据并行、无需 tensor parallel 通信）。
- 开源Serving框架是什么。修改了什么。
  - 论文未修改开源 Serving 框架（论文未明确说明）；工作负载生成沿用 vLLM 的 benchmark_serving.py（https://github.com/vllm-project/vllm）。自研新增/修改：异步 batching（双 task pool + watermark 内存水位）、utilization-aware speculation（Algorithm 1 + SVR 查找表）、PFS/CHK 仲裁，实现在其扩展 SplitwiseSim 的事件驱动模拟器里。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：HybridSpec 无官方仓库（论文未给链接，联网搜索未发现），开源状态无法确认；SplitwiseSim（https://github.com/Mutinifni/splitwise-sim）与 vLLM 开源。
  - 全过程（请求率 4 req/s 下并发请求）：benchmark_serving.py 从 ShareGPT 生成请求序列（含 prompt/输出长度分布与到达时刻）→ 事件驱动调度器按到达时刻注入请求 → XPU task pool 收到 prefill 任务，PFS 优先 prefill，长 prefill 被 CHK 沿序列维切块与 verification 组混合批 → prefill 完成即把请求作为 decode 任务注入 HB 栈 task pool（watermark 检查内存余量）→ HB 栈按当前 tree width 组批迭代 decode（utilization-aware 策略每轮估算算术强度、对照 HB roofline 调 tree width）→ 达 draft budget 后 token 列表经 wire-bonding 链路传回 XPU → XPU 上 verification 与下一批 prefill 按仲裁并行执行 → accepted token 回传、清除误推测 KV cache、进入下一轮。作用：在固定硬件下通过运行时调度把两侧单元利用率推向各自 roofline 拐点、减少气泡与排队延迟，同时改善 TTFT/TPOT 与吞吐。

## Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现（Case Study 2，prefill-guided decode expert placement）：利用 profiling 得到的 Insight 1（prefill 与 decode 阶段专家选择高度相似，Spearman ρ≥0.7、top-20 专家重叠 90%），解决大 scale MoE serving 中初始 decode 阶段（~前 1000 token）的负载不均问题——现有 EPLB 每 3000+ step 才基于周期 profile 动态调整专家放置，短输出请求永远收集不到足够数据。提出两个专家放置算法（Algorithm 2）在 decode 前用 prefill 阶段的专家选择 trace 确定初始放置：(1) Remap-based：保持每 GPU 专家数不变（容量 E/G），按 roofline cost 降序排序专家、贪心分配给最轻负载 GPU，重排专家实现负载均衡；(2) Duplication-based：每 GPU 预留 R 个额外专家槽，从默认连续布局开始，贪心迭代 R·G 次、每次选使瓶颈负载 max_g load_g 下降最多的 (expert, GPU) 对复制热门专家，复制专家的 token 在副本间均分（DeepEP dynamic dispatch）。两者都用 roofline cost model 估计每 GPU 负载。Serving 调度层面：在解码早期无历史数据时用 prefill 信息设置专家放置，属调度/负载均衡优化。
  - 实验比较：Default（Qwen/SGLang 标准连续放置：专家 0-15 在 GPU0、16-31 在 GPU1……）、Best/Worst（用 oracle decode 阶段选择生成的理论最优/最差放置，实际不可得）、Remap、Dup（R=1，128+8=136 专家/层）。指标：MoE 计算时间（三个专家线性层 up/gate+down，不含 attention、all-to-all、top-k）。结果：Remap +15.5%、Dup +12.5%（相对 Default，即最高 1.25x speedup），两者均 >2x 于 Worst，与 Best 差距 <10%；EP8 规模下 max/min 执行时间比仅 ~1.3x，更大 EP 规模预计收益更大。
- 硬件平台是什么，配置是什么。
  - 8× NVIDIA H100 80GB GPU，NVLink 互联（8×H100 DGX 服务器）。
- 开源Serving框架是什么。修改了什么。
  - Serving 框架：SGLang（论文引用 [51]），MoE 后端 DeepEP（ep_dispatch_algorithm 设为 "dynamic" 使复制专家 token 均分），并提及 DeepGEMM 软件依赖。修改：(1) 在 SGLang 中插入 cuda. Event timer 构建分布式 profiler，在每 GPU 独立测量 attention、top-k、all-to-all、MoE 各操作耗时；(2) 通过 SGLang 的 init_expert_location 接口操纵专家在 GPU 上的放置（按 Algorithm 2 计算结果加载专家权重到指定 GPU）；调度算法本身（batching 等）未改。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：https://github.com/zhongkaiyu/moe_exp_placement（DOI 10.5281/zenodo.19617695，Apache-2.0）。需要 8×H100 80GB、CUDA 12.0+、约 300GB 磁盘、PyTorch、修改版 SGLang fork、DeepEP、DeepGEMM；main_ae.py 下载 traces、跑实验、输出 CSV 生成 Figure 17，耗时 12-16 小时，真实 GPU 测量波动 ±5%（热、系统负载、NCCL 非确定性、SGLang micro-batching），预填感知放置使 MoE kernel 性能提升约 5-25%。
  - 全过程（Qwen3-235B，94 个 MoE 层、每层 128 专家、top-8 路由，SGLang + DeepEP on 8×H100）：请求从 MMLU / Global-MMLU 进入（batch 64-16384）→ prefill 阶段：分布式 profiler 采集每层每个专家的选择频率 f_{l,e}，SGLang 执行注意力 + gate + all-to-all + MoE（DeepEP）；放置计算：用 Algorithm 2 从 prefill trace 为每层算出专家到 GPU 的放置 S_g（remap 重排或 dup 复制热门专家）→ 通过 init_expert_location 让 SGLang 把专家权重加载到指定 GPU（复制专家出现多副本）→ decode 阶段逐 token 生成：gate 选 top-8 专家 → DeepEP all-to-all 把 token dispatch 到持有对应专家的 GPU（dynamic 模式下复制专家 token 均分到各副本）→ 每 GPU 运行该专家三个线性层 GEMM → all-to-all 返回结果继续下一层/下一 token；profiler 用 cuda. Event timer 记录每 GPU MoE 层耗时作为指标。效果：更均衡的负载使 MoE 计算最多提速 25%（相对默认连续放置）。

## Power Sloshing in Compound Servers for Large-Scale AI Inference Workloads（近似层次匹配：本层取其部署在 AI serving 服务器上的运行时电源管理控制器；未修改开源 Serving 框架）

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：server/module-level power sloshing 控制器（Algorithm 1）。把整台 compound AI inference server 视作功率受限模块（可配置功率上限 P_M），闭环控制循环每 100ms 采样 GPU 利用率 u_G（应用无关硬件计数器，作为负载与"GPU 是否功率受限"的代理信号），按目标利用率区间 [u_min, u_max] 动态在 CPU 与各 GPU 之间"slosh"（舀取并重分配）功率预算：u_G < u_min 时逐步降 GPU 频率 f_G 收获闲置功率，u_G > u_max 时一步拉满 f_G=f_GM 保护突发。功率受限模式（budget reallocation）：用功率模型 P_G(f_G, u_G) 估算 GPU 稳态功率，反推 CPU 预算 P_C = max{0, P_M - ΣP_Gi} 并施加 CPU power cap（SetCpuPowerLimit）；非功率受限（energy optimization 模式）：用线性映射 f_C = F(f_G) 协调 CPU DVFS（SetCpuFreq）。两种变体：SLO-Optimized（目标区间取历史利用率 P75，如 C1 上 40-50%）与 Power-Optimized（P90，如 60-70%）。
  - 实验比较：Baseline（静态最大频率、无动态电源管理）vs SLO-Optimized vs Power-Optimized vs Theoretical Minimum（§IV 理想下界，每负载点最优频率）。两组实验：(1) 服务器功率上限 sweep 下的 Performance/Watt（性能 = 不违反 SLO 的最大 QPS / 总服务器功率）——最高 1.83× 优于 Baseline，紧功率封顶下收益更大（Fig.12-14，Model-A 上策略把预算从 CPU 转给 GPU、GPU 维持 ~75% 利用率）；(2) 非功率受限下 1 小时动态负载 trace（含 idle→峰值→突变 stress 场景）的平均功率与 SLO 违反比例——C1 上 Power-Optimized 省 24%、SLO-Optimized 省 11%（贴近 Theoretical 的 30%），C2/D1 上 SLO-Optimized 8-19%；SLO 违反区间比例 C1 上 Baseline 4%、SLO-Optimized 5%、Power-Optimized 14%，D1 全部 0%。
- 硬件平台是什么，配置是什么。
  - NVIDIA Grace Hopper 系统（集成 H100 GPU + Grace CPU，[31]）。平台功率分配示例为 8 GPU × 1kW + CPU 300W（文中 TDP 指软件可配置功率上限，非厂商峰值）。单服务器细粒度测量：GPU / GPU memory / CPU / CPU memory 分量功率（C1 占比约 65%/10%/22%/3%）。频率扫描范围：GPU 53%-100% f_GM（driver 离散档），CPU 71%-100% f_CM。另有 8-GPU 服务器多服务 colocate 的 fleet 数据分析（Fig.4-6：60% 服务器有 20-40% TDP 闲置可收割）。工作负载为 Meta 生产推荐/排序模型 A、B、C1、C2、D1（不可公开）。
- 开源Serving框架是什么。修改了什么。
  - 未修改任何开源 Serving 框架（如 vLLM/Triton 等）。实现是独立于 serving 框架的服务器级电源管理控制器，运行在 host，仅依赖标准 GPU 利用率计数器和频率控制接口（如 NVIDIA NVML 或类似 API）与 CPU 频率/power-cap 接口（cpufreq / power capping），要求无应用代码修改、无应用级性能指标（QPS/SLO 不可见）。控制器为每 GPU 每 100ms 控制区间 ~100µs CPU 开销的轻量循环（8 GPU 可并行；论文建议固件级实现可进一步降低开销、收紧周期）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未提供代码/artifact 链接（联网搜索未找到公开仓库）。数据来自 Meta 内部生产 fleet、生产工作负载与生产流量，模型权重/流量/SLO 不可公开，难以完整复现；论文 PDF 公开于 Edinburgh Research Explorer（https://www.research.ed.ac.uk/files/654047698/ChoEtalISCA2026PowerSloshing.pdf）。控制机制本身仅依赖公开的 NVML/cpufreq 接口，可在任何 NVIDIA 平台上按 Algorithm 1 重新实现。
  - 使用例子（控制器输入→硬件执行全过程）：以部署 Model C1 的 Grace Hopper 服务器为例——①输入信号：每 100ms 经 NVML 采样 GPU 利用率 u_G（实测 f_G·u_G ≈ Q_w，即 QPS 代理）与当前各 GPU f_G；②决策：把 u_G 与目标区间 [u_min, u_max] 比较（SLO-Optimized 如 40-50%），u_G < u_min 则 f_G 降一档（收获功率），u_G > u_max 则立即置 f_G = f_GM（保护 P99 SLO）；③执行：功率受限模式下按功率模型估算稳态 P_G(f_G, u_G)，算 P_C = P_M - ΣP_Gi，经 CPU power-cap 接口写入 CPU 上限；非受限模式下 f_C = F(f_G) 线性映射经 cpufreq 设置 CPU 频率；④硬件层：GPU 频率迁移硬件在 100µs-数 ms 内完成，GPU 功耗随频率变化（动态功耗超线性），CPU/GPU 总功率保持 ≤ P_M；⑤反馈：下个 100ms 区间重新采样 u_G 形成闭环。端到端效果：同一固定模块功率下，把闲置组件的功率转给受限组件（或反之降低整体频率），实现 11-24% 功率节省且 SLO 基本不恶化，Performance/Watt 最高 1.83×。

## PowerGrad: Hierarchical Power Management for Power-Limited ML Inference Clusters（近似层次匹配：本层取其运行在 ML serving 集群上的分层电源管理控制器；自研用户级软件框架，未修改开源 Serving 框架）

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：PowerGrad 分层电源管理框架，面向功率受限（power-limited）ML 推理集群。三个组件：(1) Gradient Estimator——每 100ms 读取硬件性能计数器（BIPS、核心利用率 util、内存 stall 周期 ldm_stalls、频率 f）与电压/频率值，在线构建每个核心的功率模型 P(V)（三次多项式 idle 功率 + 活动功率）与性能模型 CPI(f)（线性），微分得到性能梯度 ∂BIPS/∂P（通过链式法则把 V、f、计数器间的耦合近似分解）；(2) Local Controller——每 100ms 在节点内处理器之间用 Algorithm 1 重分配功率预算（PL'[i] = PL[i] + lr×G[i] − α(PL[i]−P[i])，再均分调整使总和等于节点上限，并用 fmin 防饿死保护）；(3) Hierarchical Controller——每 1–4s 异步在节点/子集群之间用同一算法重分配功率预算，可递归扩展层次。核心思想：把功率从低梯度（功率不敏感，如内存密集 decode）工作负载移到高梯度（功率敏感，如计算密集 prefill）工作负载，获得净性能增益。
  - 实验比较：baseline 为 Fair（各处理器均分功率）、SLURM [35]（用不完分配时把 50% 超额功率均匀分给其他节点）、DPS [8]（按功率消耗历史定优先级，但全体功率饥饿时退化为均分）；PowerGrad 变体为默认两级（Local + Cluster）、PG-central（单集群控制器直管所有处理器）、PG-multi（四子集群三层）。指标：平均与 P95 响应延迟（各应用 geomean，归一化到 Fair）。结果：Legacy 双 CPU 16 节点集群上 PG-multi 相对最强 baseline 平均延迟降 22.9%、尾延迟降 23.0%；Accelerated 单 CPU 16 节点集群降 9.0%/9.9%；严重受限时收益更大（55W/节点 Legacy 降 23.6%/27.4%；115W/节点 Accelerated 降 18.3%/20.2%）。
- 硬件平台是什么，配置是什么。
  - Cloudlab testbed [10]。两类平台：(1) Legacy：双 CPU Intel Haswell E5-2660 v3，无 ML 加速，代表存量数据中心系统；(2) Accelerated：单 CPU Intel Emerald Rapids（Xeon Gold 5512U），带 AMX（Advanced Matrix Instructions）加速。每平台 17 节点集群（16 个推理节点 + 1 个控制器节点）。功率受限场景按集群总功率预算 sweep（如 Legacy 每节点 55–75W，总 880W；Accelerated 每节点 115W 起，总 1840W）。控制周期：节点 Local 100ms、子集群 1s、集群 4s（由最坏往返网络延迟 100ms 推导）；lr=2.0、α=0.3。
- 开源Serving框架是什么。修改了什么。
  - 未修改任何开源 Serving 框架（论文未提及 vLLM/SGLang 等）。PowerGrad 为自研用户级软件：节点内 Gradient Estimator 与 Local Controller 为 Java 线程（共享内存同步，100ms 周期）；Hierarchical Controller 为 Python 进程（网络 socket 与子控制器通信，1–4s 周期）。功率上限通过 Intel RAPL 接口强制执行（控制 V-f 状态）；功率模型回归沿用 PPEP 框架 [39] 的方法学（六个计数器：instruction-count、cycle-count、uops.executed、cache-misses、branch-misses、ldm_stalls_pending；Accelerated 额外读 exe.amx_busy 与 fp_arith_inst_retired.vector，一个 AMX busy 周期计为 16 条指令）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未提供代码/artifact 链接，联网搜索未发现公开仓库（NSF PAR 全文 2027-07-01 才公开；ISCA 2026 PDF 见 https://iacomaweb.web.engr.illinois.edu/iacoma-papers/isca26_3.pdf），开源状态无法确认。PPEP 框架 [39]（MICRO 2014，https://ieeexplore.ieee.org/document/7011408）亦未公开源码。
  - 全过程（Legacy 双 CPU 节点，Llama-high 与 Llama-low 各占一个处理器，集群总预算 880W=55W/节点）：①输入信号：每 100ms 每个核心读取 BIPS、util、ldm_stalls 与频率 f，算出 BCPS=f×util、CPI=BCPS/BIPS、MCPI=ldm_stalls/BIPS、CCPI=CPI−MCPI；②在线建模：用 (5)(6) 的离线回归系数 a_i、w_i、γ 构造 P(V) 与 CPI(f)=CCPI+MCPI×f/f^(t)；③梯度：微分得每核心 ∂BIPS/∂P，再用链式法则 (13) 聚合为每处理器梯度 G；④决策：Local Controller 跑 Algorithm 1——按 PL'[i]=PL[i]+lr×G[i]−α(PL[i]−P[i]) 调各处理器上限（把功率从低梯度内存密集的 Llama-low 移到高梯度计算密集 prefill 的 Llama-high），均分校正到节点上限，低于 fmin 的处理器保底增加 incmin=1W；Hierarchical Controller 每 4s 在节点间做同样的梯度分配；⑤执行：RAPL 把新功率上限强制到各处理器（调 V-f 状态）→ 处理器在该状态下继续执行 ML 推理请求（Poisson 到达，High/Low 平均 CPU 利用率 60%/30%）；⑥反馈：下个 100ms 重新采样形成闭环。作用：在不可预先 profile 的动态 ML serving 集群中，仅凭硬件计数器在线识别功率敏感工作负载并转移功率，降低平均/P95 响应延迟。


## PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management（近似层次匹配：本层取其 OS 级电源管理控制平面——Governor 作为可导入 serving 框架的库监控 RPS/尾延迟/SLO 并协调每域频率；Interposer 位于 serving 框架与 GPU driver 之间，透明拦截 kernel launch，未修改开源 Serving 框架本身）

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：PowerWeave，首个面向 GPU 的空间 DVFS（spatial DVFS）系统，构建在 LithOS GPU 操作系统 [12] 之上（Interposer ≈5500 行 Rust + Governor ≈250 行 Python）。Interposer 透明拦截所有 CUDA driver API 调用（kernel launch 路径），无需修改应用、serving 框架或 GPU driver；在每次 kernel launch 记录函数句柄、grid/block 维度、shared-memory 大小与 CUDA stream（共同构成 kernel identity key），用 CUDA event 对异步测 elapsed time。四个组件：(1) Online Kernel Profiler——先以最大频率建立 baseline 性能画像，再在多个频率点扫描 kernel（1965MHz 到 915MHz，12 个频率步 × 2 轮，约 150 个请求），对新 kernel 配置用 latency predictor 泛化 l = waves × (l_old/waves_old)（waves = 总 launched blocks / (SM occupancy × 分配 SMs)）；(2) Frequency-Latency Scaling——一阶 Taylor 近似构建 per-application 模型 f(k)=f_max/S, S=1+k/Σ(s·w)（w=每 kernel 占运行时间权重、s=频率敏感度因子），profiling-threshold 设为 5%（偏离超阈值则重启 profiling）；(3) DVFS Controller——根据 governor 提供的 performance slack 为每个频率域选择工作频率；(4) Governor（用户态库，可 import 进 vLLM/SGLang 等 serving 框架，仅需几行代码取用户态指标）——按可配置监控窗口计算每域允许的 performance degradation slack，slack 更新公式 s₂=((1−s₁)×l₁)/SLO，SLO 违反时快速纠正（频率拉满直到延迟回到安全余量），支持多 SLO（TTFT/TPOT）取最保守 slack。频率变更经 NVML 下发。
  - 实验比较：baseline 为 LithOS（SOSP'25 SOTA，单设备级频率域 DVFS + 空间多租户）与 default GPU DVFS policy（默认最大频率、超 TDP 降频/热节流）。三组实验：(1) disaggregated prefill（prefill/decode 各占一半 TPC，4 模型 Llama-3.2-1B/Llama-3.1-8B-Instruct/Qwen3-14B/Qwen3-32B-FP8，Azure trace，TTFT/TPOT 独立缩放）——PowerWeave 平均节能 28% vs LithOS 13%，最高 38%，Qwen3-32B-FP8 上比 LithOS 高 8×；(2) spatial multitenancy（4 配置 × 3 模型租户，Table II：18/74、19/74、37/74 TPC 分配，MLPerf interactive/server 场景，Azure trace 随机分 3 split、请求率 1/3）——PowerWeave 平均 28%（最高 35%）vs LithOS 平均仅 10%（最差 6%）；第五场景：两个 Qwen3-14B（1 TPC interactive + 73 TPC server）PowerWeave 仍省 40%、LithOS 几乎无收益；(3) agentic workflow（AutoGen 编码 pipeline，3 个 Qwen3 模型 4B/8B/14B 各占 10/27/37 TPC，Table III，自定义 1024-token prompt + 512-token 输出上限，open-loop 持续灌入，batch 4→10 步进 2）——平均节能 19%（最高 22%）、吞吐不降，TPJ（throughput per Joule）最高 +20%、平均 +15%，LithOS 退化为 default policy 无节能。另有 load sensitivity sweep（Poisson、ShareGPT Vicuna + scientific papers、MLPerf SLO）：Llama-3.2-1B 低 RPS 最高 41% 节能；最坏场景（Qwen3-32B-FP8、Llama-3.1-8B）LithOS 落后 PowerWeave 最多 25%。
- 硬件平台是什么，配置是什么。
  - NVIDIA DGX B200：Ubuntu 24.04.2 的 DGX Server 7.0.2 软件栈、NVIDIA driver 580.82.07；8× B200 GPU，每 GPU 192GB 显存、148 SMs（9 GPC、74 TPCs、每 TPC 2 SM）。频率扫描范围 1965MHz–915MHz。空闲功耗 ≈140W。能量测量用 NVIDIA DCGM 4.2.2（与 DCGM 功率×时长交叉验证，不一致则重跑）。
  - 由于当前 GPU 不暴露细粒度频率域，论文用多 GPU + 分配 TPC 使总分配恰好等于一个完整 GPU 来仿真空间 DVFS，并对比 (i) isolated 执行、(ii) 同 GPU 计算分区、(iii) MIG 分区三种方式；争用使 TTFT/TPOT 平均 +3%、最差 <+7%，因此所有仿真实验将 SLO 目标保守放大该比例。
- 开源Serving框架是什么。修改了什么。
  - Serving 框架：vLLM 0.10.2（[26]，+ PyTorch 2.8 for CUDA 12.8 on Python 3.12）；disaggregated prefill 用 LMCache 0.3.6 [9] + NIXL 0.6.0 [27]；agentic pipeline 用 AutoGen [57]。论文未修改这些框架的内部调度逻辑；PowerWeave 的 Governor 作为极小库被 import 进框架（从 vLLM/SGLang 取每域请求率与尾延迟只需几行代码），Interposer 在框架之下透明拦截所有 kernel launch，二者均对框架透明。与 GPU 共享机制兼容：MIG 硬件分区下 Governor 无需修改即可工作；MPS 部署沿用 LithOS 的 TPC assignment；或用 NVIDIA Green Contexts 实现 MPS 内空间隔离；AMD 侧对应 SPX/DPX/CPX（MI300X/MI355X）与 ROCm CU masking。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未提供 PowerWeave 代码链接，联网搜索未能确认公开仓库（ISCA 2026，CMU CAOS 组；构建于 LithOS 之上，LithOS 亦无已确认公开仓库，截至 2026-08）；底层依赖 vLLM（https://github.com/vllm-project/vllm）、LMCache（https://github.com/LMCache/LMCache）、NIXL（https://github.com/NVIDIA/nixl）、AutoGen（https://github.com/microsoft/autogen）、LithOS 相关材料（slides: http://www.cs.cmu.edu/~dskarlat/slides/lithos_sosp25_slides.pdf）均可获取。
  - 全过程（disaggregated prefill on B200，Azure trace 驱动）：Azure trace 按 1/8 缩放 inter-arrival（Llama-3.2-1B 用 1/4）注入请求 → vLLM 0.10.2 将请求拆为 prefill/decode 阶段（LMCache 管 KV cache、NIXL 做跨实例传输）→ 每 kernel launch 被 Interposer 透明捕获：记录 identity key（函数句柄+grid/block+shared mem+stream）、注入 CUDA event 对 → Online Kernel Profiler 先全速测 baseline 再扫 12 个频率点，为新 kernel 配置按 wave 比例预测延迟（l = waves×l_old/waves_old）→ Frequency-Latency Scaling 合并各 kernel 曲线成 per-application 模型（权重 w 随 prefill/decode 比例在线更新、敏感度 s 固定）→ Governor 每监控窗口算 performance slack（s₂=((1−s₁)×l₁)/SLO）→ DVFS Controller 据此为 prefill 域与 decode 域选各自频率 → 频率经 NVML 写入 → 每 kernel 完成后 Interposer 对照预测延迟，偏离超 5% 阈值则重启 profiling；SLO 违反时 Governor 立即要求拉满受影响域频率。作用：prefill 域跑高频率满足 TTFT SLO、decode 域跑低频率利用 TPOT slack，实现 28% 平均节能且零 SLO 违反。


## R2D2 Robotized Reconfigurable Network for Disaggregated Datacenters

- 属于Serving调度的实现是什么？实验比较什么？
  - R2D2 software runtime 的联合任务分配与网络重构调度（近似匹配本层：非请求级 serving 框架，而是 datacenter VM/任务分配调度）。两层设计：系统控制器（datacenter orchestration 层，全局资源管理）+ 机器人控制器（嵌入式，低层运动规划/闭环控制/故障处理）。核心是 Joint Allocation and Reconfiguration 算法（Alg.1）：两阶段分层（先选 datacenter row，再选 row 内 compute-memory 节点）；优先复用已建立链路避免 reconfiguration、fitness 函数考虑资源匹配与链路利用率、必要时允许 reconfiguration 并按实时空闲机器人列表并行分发（异步不阻塞，避免系统控制器串行化瓶颈）；重配成本计入分配决策以主动工程化流量稀疏/稳定。
  - 实验比较：与 best-fit baseline（云常用策略，同跑 R2D2 硬件）——联合算法平均与 p99 allocation latency 低 10-20×（best-fit 机器人无关、触发过量重配级联延迟）；与 fat-tree/OCS 相比 allocation latency 高 41-51%(avg)/27-30%(p99)（512 节点）、5-39%/21-29%（2048 节点 + spine），但仅增加 VM 总运行时间的 0.49%；利用率达 99% CPU / 69% memory；机器人并行度 37-45% 重配重叠 2+ 机器人（2/4 robot 配置）。

- 硬件平台是什么，配置是什么。
  - 512/2048 节点、400 GbE；Broadcom P1400GD 400Gb NIC（$2198）；R2D2 fabric 为 2-robot 或 4-robot 配置（400G 拆 2×200G / 4×100G breakout transceivers），每 R2D2 unit 576/2304 端口、120W。分配延迟用自定义 discrete-event simulator 在 R2D2 runtime 上测量。VM trace：Protean [32] 生产集群 trace（2064 机器、48 核/384GB 每机，每次请求含 VM CPU/内存/存储需求与时长，排除存储与应用网络需求，含 1000s 请求/秒 burst）。

- 开源Serving框架是什么。修改了什么。
  - 论文未修改开源 serving 框架（非 LLM serving 论文）：R2D2 runtime 是自研两层调度系统，通过标准 API（兼容 Azure Protean rules [32]）暴露给上层 orchestrator（hypervisor/scheduler），上层无需修改编排逻辑即可请求资源分配与网络配置。最接近"多请求调度"的层面是 VM/任务分配调度，而非请求级 serving——近似匹配本层次。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：R2D2 runtime 未提供公开仓库（联网搜索无法确认）；输入 trace 公开：Protean VM allocation traces [32]、Gao et al. [25]/Shoal [62] 内存流量 trace。
  - 使用例子（一次 VM 分配）：hypervisor 通过标准 API 提交任务（compute 需求 C + memory 需求 M）→ 系统控制器执行 Alg.1：①FEASIBLEROWS(C,M) 按 best-fit fitness 选最高分行；②先试 row 内已连接（无需重配）的 compute 节点（FEASIBLECNODESNORECONF 排序、COMMIT 校验资源并绑定，成功即返回）；③无则枚举可行 compute-memory 对，对每对查 AVAILABLEROBOTS 实时空闲列表，向机器人控制器分发"断开端口 A、连接端口 B"命令（多机器人可并行），RECONFIGURENETWORKBYROBOT 成功后 COMMIT 并返回节点分配；④失败标记机器人 down、换下一候选；全部失败则 QUEUE FOR RETRY。
  - 输入到硬件执行全过程：输入=VM 分配请求流（Protean trace，含 1000s 请求/秒 burst）→ 系统控制器联合调度（异步分发、不阻塞后续分配）→ 机器人控制器（嵌入式板）把高层命令翻译成 stepper 轨迹/G-code，闭环控制（编码器位置、插入力反馈）执行物理插拔 → 光纤 latch 后链路变被动直连，数据面由 NIC 直通跑 400 GbE（无逐包交换）→ 完成通知回系统控制器，hypervisor 调度 VM 运行。作用：联合优化任务放置与物理拓扑，促进流量稀疏/稳定、最小化机器人移动，使分配延迟增量仅占 VM 运行时间 0.49%；故障时（机器人 down、重配失败）快速 failover（备选机器人/备选放置），已分配应用性能不受影响。

## Rearchitecting the Datacenter Lifecycle for AI（近似层次匹配：本层取其 LLM 推理 serving 的 workload 级效率评估与 operation 阶段 SLO 驱动的调度/迁移/分解优化；论文本体是跨 build/IT provisioning/operation 三阶段的 TCO 生命周期框架，非请求级 serving 框架修改）

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：TCO 驱动的跨阶段 AI 数据中心生命周期管理框架（AI Lifecycle Compass），覆盖 build、IT provisioning、operation 三阶段、15 年（2015–2030）规划期。性能侧核心是 roofline 分析模型：从 LLM 架构与参数量解析推导算术强度与内存占用，结合硬件峰值 FLOPS、内存带宽/容量、互连带宽/延迟、TDP 预测 TTFT 与 TBT 延迟；在固定 SLO（TTFT 400ms、TBT 100ms）下不断增加负载直到违反 SLO，得到 goodput（可持续 RPS）作为利用率点，从而按需最小化 GPU 供给。operation 阶段评估的 serving 侧软件优化包括：smooth model migration（新模型发布后平滑迁移）、model quantization、KV-cache management、prefill/decode disaggregation（把 compute-bound prefill 放新 GPU、memory-bound decode 放旧 GPU）、model routing、heterogeneity-aware scheduling（把负载映射到最优/可用硬件代际）、infrastructure-aware scheduling（利用功率/冷却余量）。refresh 决策用蒙特卡洛模拟枚举各硬件代 0–10 年生命周期策略的 TCO 分布。
  - 实验比较：(1) workload 级效率——用 vLLM 在真实 GPU 上跑 Llama3 1B–405B 测 TTFT/TBT（2K 序列、batch 8、TP1/TP4/TP8，按 H200 归一化），与 roofline 模型对比误差 <5%；比较 dense Llama3 vs sparse Qwen3（30B-A3B、235B-A22B）与 transformer Llama3-3B vs state-space Mamba-2.8B 在不同代际 GPU 上的延迟缩放；(2) baseline 为传统 5 年刷新周期（CPU 数据中心惯例 [111]），图 11a 显示大部分策略落在 baseline 右侧，AI 场景下多种替代策略可降 TCO 15–20%；(3) build 阶段对比 flat vs hierarchical 电源（flat 降 4.2%）、air/hybrid/liquid 冷却（75/25 hybrid 降 9%）、Ethernet/InfiniBand/NVLink/hierarchical 组网（hierarchical 降 6%）；(4) operation 阶段单策略 TCO 降 12–39%，组合超 60%；(5) 跨阶段联合优化整体 TCO 最高降 40%；(6) 四种结构性突变场景（需求冲击 α=3 降 31%、模型收缩 β=0.8 降 43%、硬件能力跳变 γ=3 降 38%、硬件价格冲击 δ=0.6 降 36%）下框架仍稳健。
- 硬件平台是什么，配置是什么。
  - workload 级效率实验：NVIDIA T4、V100、A100、H100、H200 GPU，Llama3 1B/3B/8B/70B/405B（TP 取能装下模型的最小值：TP1/TP1/TP4/TP8/TP8；部分旧 GPU 因显存限制无法跑大模型）。参考成本数据：P100 约 $9K/GPU、H100 约 $30K/GPU（DGX H100 8×H100 整机 >$350K、TDP 10.2kW）；CPU 侧 Intel Emerald Rapids 64 核 385W、Haswell $7K/server、Granite Rapids $12K/server。TCO 场景：10MW 数据中心、75% 利用率、约 500 台 H100 server、年耗电 70GWh；模拟从 2015 年 50 台 P100 支撑 100K RPS（年 TCO ≈$0.2M）开始，2024 年流量达 350K RPS 触发 H200 大刷新、server 数达 25K、年 TCO $0.3B。DCPerf [106] 用于 Intel server 代际性能/Watt 评估。
- 开源Serving框架是什么。修改了什么。
  - 论文未修改任何开源 Serving 框架。评估用 vLLM [59]（https://github.com/vllm-project/vllm）在真实硬件上测 TTFT/TBT 并校准 roofline 模型；DynamoLLM [105] 的输入 trace（含 diurnal 模式）作为负载输入；DCPerf [106]（https://github.com/intel/dcperf）评估 Intel server 代际吞吐与性能/Watt。论文自身的 deliverable 是 TCO 模拟框架 AI Lifecycle Compass（https://github.com/Azure/AI-Lifecycle-Compass，MIT 协议，Python 包 dc_tco/），非 serving 框架。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：AI Lifecycle Compass 已开源（https://github.com/Azure/AI-Lifecycle-Compass），含 dc_tco/ 模块（config/hardware/models/performance/demand/tco/policies/simulation/scenarios/rental/plotting/cli.py）、YAML 配置、notebooks、tests、docs；README 声明对应 ISCA 2026 论文。安装 `pip install -e .`，CLI 示例：`dc-tco run --config configs/default.yaml --policy baseline`、`dc-tco sweep`、`dc-tco monte-carlo --trials 10000`；Python API：`load_config()` + `run_simulation(cfg)` 返回 `result.total_tco`。依赖 vLLM/DynamoLLM trace/DCPerf 均公开可复现 workload 评估。论文本身的 TTFT/TBT 实验脚本未单独开源（论文未明确说明）。
  - 使用例子（框架输入→决策→硬件部署全过程）：①输入：YAML 配置（硬件 roadmap P100–B200 含 NVLink 规格与 FLOPS/带宽/TDP/价格、workload 模型大小趋势（Epoch AI 数据）、需求增长 15%/年、PUE、电价、折旧、8 个关联随机变量分布）；②roofline 性能模型：对每个 (模型, GPU) 组合从 LLM 参数量与架构解析出算术强度与内存占用，用硬件峰值 FLOPS/带宽算出 TTFT/TBT 延迟；③SLO 约束（TTFT≤400ms、TBT≤100ms）下按 DynamoLLM trace（100K RPS、diurnal 波动）拉高负载求 goodput，据此决定该季度最少需采购多少台 server；④蒙特卡洛引擎（10,000 trials、8 个相关随机变量经逆 CDF 映射到 log-normal/triangular 等边缘分布）遍历各刷新策略（每代生命周期 0–10 年、可跳过代、可多代共置）得到 TCO 分布与置信区间；⑤输出最优刷新时间线（图 13：跳过 B100/B200、延长 H100/H200 寿命、2027 年初因 GPU 效率提升 server 数短暂下降）→ 转化为实际采购/退役计划部署到数据中心。作用：把模型演进、硬件代际、基础设施约束与运营优化统一到 TCO 优化目标下，跨阶段联动（如 build 期投更大 powersharing 域换取 IT provisioning 刷新灵活性与 operation 效率），实现相对传统割裂式管理最高 40% 的 TCO 降低。

## SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution（近似层次匹配：本层取其 MoE 专家级 offloading/缓存/预取/CPU-GPU 调度；论文是自研 Python 推理运行时，而非修改开源 Serving 框架）

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现是面向边缘 GPU 显存受限场景的 online MoE expert offloading 调度器 SMoE：所有专家常驻 CPU 内存，按需经 PCIe 加载到 GPU 或直接在 CPU 计算，完全在线运行、无需 offline 准备。调度策略四部分：①expert-cache router 按 gate score 把激活专家分成 top-score/low-score，low-score 替换为 GPU 已缓存的近似 score 专家（提升 GPU 命中率、减少 PCIe 传输与 CPU 计算）；②score-aware cache eviction（按近 n 次平均 activation score 淘汰最低分专家、替代 LRU，带 protection shield）；③top-score prefetching（用 GPU 常驻共享专家+缓存专家预测下一层 top-score 专家，与当前层计算重叠）；④CPU-assisted task load scheduling（Algorithm 2，双指针按 C_load/C_CPU 成本最小化 max(T_load, T_CPU)）。目标：低 batch（1–3，边缘单设备场景）下最小化 TPOT 与 TTFT。
  - 实验比较：与 MoE-infinity（预取+历史 router 数据）、llama.cpp（纯 CPU 推理）、DeepSpeed（layer-wise 加载、静态缓存）、HybriMoE（CPU-GPU 调度+缓存管理）比 TPOT/TTFT/GPU cache ratio/精度，batch=1/3；S3 下 TPOT 平均降 48%（batch=3）/34%（batch=1），GPU cache ratio 相对提升 ≥65%、命中率 >60%；多 batch 时额外用同 batch 其他 token 的 top-score 专家替换 low-score 专家（top-score 反正要算，免去额外加载）。
  - 硬件平台是什么，配置是什么。
  - 单卡边缘/低端 GPU：S1 RTX 3080 Ti 12GB（PCIe 3.0 + Intel E5-2683 v3）；S2 RTX 4060 Ti 16GB（PCIe 3.0）；S3 A6000 48GB（PCIe 4.0 + Intel Xeon Gold 6444Y）。S3 复现需约 150GB CPU 内存（Qwen2-57B 权重 107GB）。模型：deepseek-moe-16b-base / Qwen2-57B-A14B-Instruct / XVERSE-MoE-A4.2B-Chat。
  - 开源Serving框架是什么。修改了什么。
  - 论文未修改开源 Serving 框架：SMoE 是自研 Python 推理运行时（HuggingFace Transformers 加载模型 + 自实现专家缓存/预取/调度），非 vLLM/TGI 等 Serving 框架改造。调度决策（expert-cache router 轻量搜索、cache eviction、CPU-assisted 调度）全在运行时内完成，与 GPU 计算/PCIe 加载流水线重叠。baseline 中 HybriMoE 基于 ktransformer 架构（自带量化），论文称已移除量化效果做公平对比；DeepSpeed 与 llama.cpp 仅作 baseline 对照，未修改。
  - 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：https://github.com/goingshr/SMoE（figshare: https://doi.org/10.6084/m9.figshare.31982136）。环境：Python 3.13 free-threading（no-GIL）、Ubuntu 专属依赖脚本（dependency.sh：apt/snap/sudo + Rust 工具链 + tokenizers 源码编译）；run.sh 以大写环境变量传参，main.py 为直接入口；论文文档中的 --alpha 控制替换阈值（S3 默认 0.25），仓库实际 config 字段为 replaceScoreRatio/window_size（null=LRU）/if_prefetch/if_usecpu/if_replace。artifact 用 shell 脚本自动跑 Gaokao/triviaqa/WiC/Race-mid/gsm8k 五个数据集并自动解析日志提取 TPOT 与 GPU cache hit ratio（对应 Fig.12/13）。
  - 使用例子（一次 token 解码跨相邻两层，框架输入到硬件执行全过程）：
    ```
    # 输入：batch token 序列（如 gaokao_math_ii 数据集，input_num=20, output_len=60, batch=3）
    # 层 i 执行：
    # 1) GPU 计算 attention + gate（common params 常驻 GPU）→ 该层全部 expert 的 gate score
    # 2) CPU 上 expert-cache router（与 GPU 计算/PCIe 重叠）：按 α 阈值分 top/low-score，
    #    low-score 替换为 GPU 已缓存的近似 score 未激活专家；cache eviction 更新 +
    #    当层保护 shield（防用前被淘汰）
    # 3) CPU-assisted task load scheduling（Algorithm 2）：剩余未命中专家按双指针分配——
    #    高 score 的走 PCIe 加载到 GPU（load 侧），低 score 的留 CPU 计算（CPU 侧），
    #    最小化 max(T_load=n_load×C_load, T_CPU=n_CPU×C_CPU)，C_load 为主成本、C_CPU 视为常数
    # 4) PCIe 操作：预取层 i+1 的 top-score 专家（GPU 上用共享专家+缓存专家预测），
    #    同时加载层 i 需要的新专家；CPU 计算层 i 的部分专家
    # 5) GPU：先算已驻留专家（直接计算），PCIe 完成后继续算新加载专家；CPU 结果传回 GPU 合并
    # 6) 进入层 i+1（层间串行，bubble 被 PCIe 延迟掩盖；层 i+1 的预取已在第 4 步开始）
    # 输出：逐 token TPOT 与 GPU cache hit ratio 日志；hit rate 提升（>60%）→ TPOT 下降
    ```
    作用：以专家替换+命中率优化把 TPOT 中占比 42% 的 low-score 专家加载时间转为 GPU 命中计算，降低边缘设备 MoE 解码延迟且精度无损。

## SPEC CPU: The Next Generation（近似层次匹配：本文是 CPU 基准套件与异构多程序负载调度方法论论文，本层取其 RRR（Rolling Round-Robin Rate）异构多程序调度运行风格；"框架"为 SPEC CPU 的 runcpu 工具集而非 LLM Serving 框架）

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现是 SPEC CPU 2026 新增的 RRR（Rolling Round-Robin Rate）运行模式：把原有 SPECrate 套件（intrate/fprate，N 个 benchmark）当作异构多程序负载来运行的标准调度方法。核心规则：M 个核（copies）各自持有一条包含全部 N 个 benchmark 的队列，队列起点错开（按进程号 mod benchmark 数取初始 benchmark，随后按固定步长 --rrrrate_inc 循环），每个核把整条队列顺序跑完 N 个 benchmark、重复 --iterations 轮。由此每个 benchmark 在每个核上恰好各跑一遍，任何 benchmark 的指令总量在所有核上完全相等——从用户态 CPU 执行角度看"等量工作"，从机制上消除 sample imbalance（各程序独立运行时长不同造成的不均衡），只保留 schedule imbalance（并发争用造成的非对称干扰）。
  - 实验比较：同机同核数下对比同质负载（refrate，每个 copy 跑同一个 benchmark 的经典 SPECrate 同质容量方法，自 1992 年沿用）与异构负载（rrrrate，所有 copy 并发滚动调度全部 intrate benchmark）的时间图/性能差异；以及不同 --rrrrate_inc 步长（默认 1，可自定义）对轮转编排的定制。Figure 2 给出 AmpereOne 系统 48 copies 下三种调度风格对比。RRR 仅处于 exhibition 状态（评分方法未定稿，SPEC 公开征求评测指标——累计 IPC、平均吞吐、调和平均、公平性指数等仍在学界讨论中），因此 RRR 结果不产生官方 SPECratio/geomean 可比分。
  - 硬件平台是什么，配置是什么。
  - 特征分析机：AMD EPYC 9755（Zen 5，2.7 GHz、最高 Boost 4.1 GHz，L1 32 KiB I + 48 KiB D、L2 1 MiB、L3 512 MiB，2.3 TiB DDR5-6400），GCC 15.2 -O3，Ubuntu 24.04 LTS / Linux 6.8.0-44-generic（Table IV）。RRR 演示：AmpereOne 系统 48 copies（Figure 2）。基准参考机（用于计算 SPEC ratio 的归一化基准）：Lenovo ThinkSystem HR330A，3.0 GHz Ampere eMAG 8180（ARMv8 aarch64，32 核，2018 年产品）。评估工具链：PMC（Performance Monitoring Counters）+ Top-down 微架构分析、BBV（Basic Block Vector，Valgrind 提供）自相似性 recurrence 图、perf 时间序列图（IPC/frontend-bound/backend-bound）。
  - 开源Serving框架是什么。修改了什么。
  - 论文未修改开源 Serving 框架（本文不涉及 LLM Serving）。最接近的匹配是 SPEC CPU 自带工具集 runcpu（随套件分发，C++/Perl 实现，社区有第三方构建工具 jiegec/spec2026）。修改/新增：runcpu 新增两个标志——`--rrrrate`（启用滚动轮询 rate 模式）与 `--rrrrate_inc[=N]`（队列步长，默认 1，0 表示只跑初始 benchmark 的快速验证模式）；`--copies`（进程队列数）、`--iterations`（每队列重复轮数）复用既有参数。实现要点：RRR 模式下 copies 由 runcpu 直接 fork（而非 specinvoke），`$SPECCOPYNUM`/`$BIND` 由 runcpu 展开使既有 submit 命令无需改动；结果校验推迟到独立并行阶段（每 copy 一个进程）；每个 benchmark×copy×iteration 各建一个运行目录（磁盘占用显著增加）；结果表无 SPECratio/geomean，每个 benchmark 报 per-copy 中位数再取均值的时间、CV（标准差/均值）、以及 min/max/mean/median/variance/sigma/cv/四分位/IQR/Tukey fences 等统计字段（RSF 中 spec.cpu2026.results.<benchmark>.<tune>.*）；RRR 模式不报能耗字段、`--minimize_rundirs` 不兼容。
  - 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：SPEC CPU 2026 不是自由开源软件——是商业授权套件（新客户 $3,000、2017 老客户优惠 $2,000、合格非营利 $750、认证学术机构免费，须经教授申请），分发源码、不供二进制，用户自行编译；ISO 内含 redistributable_sources/（第三方开源组件源码与许可）。官方文档：https://www.spec.org/cpu2026/docs/runcpu.html（RRR 见 section 1.8）、https://www.spec.org/cpu2026/。社区工具：https://github.com/jiegec/spec2026（需自备授权副本的 benchspec/ 与 bin/）。论文 arXiv: 2605.01575。
  - 使用例子（一次 RRR 多程序调度运行，框架输入到硬件执行全过程）：
    ```
    # 输入：runcpu 命令 + 选中的 benchmark 名 + copies/iterations/inc 参数
    runcpu --rrrrate -c oct29a --iterations=2 --copies=4 727.cppcheck_r 714.cpython_r 750.sealcrypto_r 734.vpr_r
    # → runcpu 为 4 个 copy 各建一条队列（含全部 4 个 benchmark），起点按进程号 mod 4 错开：
    #   copy0: 714→727→734→750 | copy1: 727→734→750→714 | copy2: 734→750→714→727 | copy3: 750→714→727→734
    #   每轮按 --rrrrate_inc=1 步进，队列重复 --iterations=2 轮
    # → runcpu 直接 fork 4 个编译好的 benchmark 进程（$SPECCOPYNUM/$BIND 由 runcpu 展开），
    #   4 个进程同时启动、独立运行不再同步，期间它们并发射出 CPU 上的共享资源
    #   （取指带宽、ITLB、分支预测器、L1/L2/L3、DRAM 带宽、预取器）→ 产生跨程序 cache/TLB/带宽争用
    # → 每个 benchmark 在每个 copy 上恰好完成 2 次（4 copies × 2 iterations = 8 次运行/benchmark），
    #   结果校验在独立并行阶段逐 copy 进行；每 benchmark 先取各 copy 内 2 次运行的中位数，
    #   再对 4 个 copy 的中位数取均值作为该 benchmark 的运行时间，并报 CV 等统计量
    # → 输出：每 copy 每 benchmark 的运行时间与 CV（如 220/225/228/231s → mean 226s, CV≈0.018）、
    #   无 SPECratio/geomean；配合 scripts.misc 中的 rate_timeplot.py 生成 Figure 2 式时间图
    #   （refrate 同质 vs rrrrate 异构对比），供研究者按自选指标（累计 IPC/吞吐/调和平均/公平性）分析
    ```
    作用：以标准化、确定性、可复现的轮转调度产生异构多程序负载，使多核争用/OS 调度/资源划分研究摆脱"自定义 benchmark 混合 + 临时调度策略"的碎片化现状，可直接对比不同研究的结果。

## Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现是 RESONATOR，一个面向 MLLM 推理服务的共生式 Serving 运行时（构建于 SGLang 之上），沿两条轴做细粒度运行时调度：(1) **Intra-GPU Sharing Engine**——在单 GPU 上管理视觉编码器与 LLM 之间的 SM/HBM 共享，双模式调度：complementary 场景（LLM chunk 为 memory-bound 的 decode-heavy 阶段，Tag(c)=mem 且 decode token 占比 ρ≥ρ0）用显式 SM 分区（给 decode 保留 SM_dec=⌈SM_total·SM_dec_min(c)⌉ 的 SM 切片，其余给 encoder）；contending 场景（compute-bound 的 prefill-heavy chunk）用 per-kernel stream binding（Alg.1：把 compute-bound kernel 路由到可占全部 SM 的 wide stream、memory-bound/低占用 kernel 路由到窄 SM 子集的 narrow stream，wide/narrow 流由 green-ctx 或 libsmctrl 绑定 SM 子集，运行期只查 kernel profile 表选流）。两场景带迟滞切换。(2) **Inter-GPU Parallelism Engine**——跨 GPU 动态选择 encoder 的 DP/TP 并行方案：PRISM 调度算法把请求队列建模为 Multiple-Choice Knapsack Problem（最大化 ∑1/T(R_i,k)，DP 递推 dp[i][j]=max(dp[i-1][j], max_k{dp[i-1][j-k]+1/T(R_i,k)})，回溯出最优 batch 与各请求 TP 度），配合 **logical sharding** 实现近零开销并行切换（启动时每 GPU 预载完整未分片的 encoder 权重，运行时只改 cuBLAS/CUTLASS 的 leading dimension(ld) 参数做 strided GEMM 逻辑分片，TP 切换从数据面 weight 重分布降为控制面元数据更新）。
  - 实验比较：①端到端对比 vLLM、SGLang（text-only 强 baseline）与 EPD-Serve（MLLM 专用 encoder-prefill-decode 分池系统），三个模型、递增请求率（RPS）下比吞吐、mean/P99 TTFT、mean TPOT、mean E2E latency；②并行策略 landscape 分析（8 GPU、8DP/8TP/4DP-2TP/2DP-4TP 四固定策略 × 3 分辨率 336/1024/2048 × 3 RPS 1/4/8，验证无单一静态策略最优）；③Intra-GPU 消融（SM Partitioning Only vs Stream-based Sharing Only vs 完整引擎，归一化到 Partitioning；另测 encoder-decode 共存下 TPOT P99 窗口内 SLO 违例率）；④高异构 batch 案例研究（20 请求、4 种分辨率混合，动态调度 vs 静态策略 vs Oracle）；⑤系统消融（Static Baseline / +Intra-GPU Sharing / 完整 RESONATOR）；⑥与 EPD-Serve 对比（RESONATOR 4×A100 vs EPD-Serve 6×A100，省 33% GPU）；⑦logical sharding 计算效率微基准（3 类 encoder GEMM：QKV/FFN-up/FFN-down，L_seq∈{1k,4k,8k,16k}、TP∈{1,2,4}，contiguous shard vs strided logical shard 比延迟与 MFU，中位差 0.7%、91% 配置 <2%）。
  - 主要结果：相对 SGLang/vLLM，mean TTFT 最高 5.1×、TPOT 最高 3.0×、mean E2E 最高 4.9×、吞吐最高 3.4×（如 Qwen2-VL-7B 吞吐 876 vs 462 vs 257 tokens/s；Kimi-VL-16B@10RPS mean TTFT 11.6s vs 43.5/59.7s）；相对 EPD-Serve 用少 33% GPU 仍提升 TTFT 2.31×、E2E 1.58×、TPOT 1.75×；消融下 Intra-GPU 引擎把 TPOT@2RPS 从 155ms 降到 60ms、完整系统到 42.7ms，Inter-GPU 引擎带来 13.7× TTFT@4RPS 提升；共存场景 TPOT P99 SLO 违例率从 stream sharing 的 20%（21/103 窗口、峰值 28s）降到 5%（5/100 窗口、峰值 479ms）。
- 硬件平台是什么，配置是什么。
  - 单服务器 8× NVIDIA A100 SXM 80GB，GPU 间 NVLink 互联，CPU 为 Intel Xeon Gold 6430。LLM backbone 用 TP=4 on 4 A100（Qwen2-VL-7B、Kimi-VL-16B）、TP=8 on 8 A100（Qwen2-VL-72B），RESONATOR 与 baseline 用相同 LLM 并行度，仅 encoder 并行度运行时动态调整。微基准中 A100 FP16 峰值按 312 TFLOPS 归一化 MFU。
- 开源Serving框架是什么。修改了什么。
  - 基于开源 Serving 框架 SGLang-0.4.7（论文明确给出版本号）。修改/新增：①新增强化 chunked-prefill 的 LLM chunk 运行路径（chunk 特征向量 c=(n_p,n_d,L_c)，L_c 为平均 KV cache 深度 bucket）；②新增 Performance Atlas 离线 profiler 与在线查询接口（encoder 多项式模型 T_enc(r,k,SM_enc) 以 L_seq=⌈H(r)W(r)/P²⌉ 为唯一复杂度参数、LLM 随机森林模型 T_llm(c,SM_llm)，存储合法 TP 集 K(r)、decode SM 最小配额 SM_dec_min、memory/compute 标签与 kernel profile 表 P）；③Intra-GPU Sharing Engine：CUDA 流级 SM 配额控制（wide/narrow stream + SMCTRL.SetQuota），SM 分区在 chunk 边界切换（兼容 CUDA Graph 重放的 decode 路径），contending 路径用 eager 执行逐 kernel 选流；④Inter-GPU Parallelism Engine：encoder 请求批量形成 + PRISM DP 调度器 + logical sharding（strided GEMM 按 ld 参数分片，权重全量预载每 GPU）实现零开销 DP/TP 切换。调度查询只读 Performance Atlas（offline 一次 profiling 约 6 小时，Qwen2-VL-7B 上 profiled 范围平均预测误差 4.7%、外推 8.1%）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未明确说明 RESONATOR 是否开源或给出仓库链接（正文只引用 SGLang https://github.com/sgl-project/sglang 与 libsmctrl [26] 论文）。可复用组件：SGLang-0.4.7（开源 Serving 框架）、libsmctrl（开源，论文 [26] Bakita & Anderson "Hardware compute partitioning on NVIDIA GPUs" 提供 CUDA 流→SM 子集绑定）、green-ctx（论文提到的另一 SM 绑定机制）、cuBLAS/CUTLASS strided GEMM（ld 参数，CUTLASS 仓库 https://github.com/NVIDIA/cutlass）。SGLang 的 LLM serving 使用方式：requests 经 RadixAttention/Scheduler 进入 chunked-prefill 执行，prefill/decode 以 chunk 形式调度到 GPU；RESONATOR 在其上叠加 encoder 与 LLM 的共享与 encoder 的动态并行。
  - 使用例子（一个高分辨率图像请求到达 8×A100 集群，SGLang/RESONATOR 框架输入到硬件执行全过程）：
    ```
    # 输入：图像请求 R（如 2048×2048）+ 文本 prompt，Poisson 到达，batch 排队
    # 1) Preprocessor（CPU）：resize/归一化/分 tile → 生成 tile 序列 → 序列长 L_seq=⌈H·W/P²⌉
    # 2) Inter-GPU Parallelism Engine（控制面）：从队列 Q 取 m 个请求，
    #    对每个 R_i 查 Atlas Λ：合法 TP 集 K_i（按显存容量过滤）+ 各 k 的延迟 T(R_i,k)，
    #    PRISM 跑 DP dp[i][j]=max(skip, max_{k≤j}{dp[i-1][j-k]+1/T(R_i,k)})，
    #    回溯得到最优 batch 及每请求 TP 度（高分辨率→TP 大、低分辨率→DP 多）
    # 3) Logical Sharding（零开销切换）：每个 GPU 已预载完整 encoder 权重；
    #    对 TP=k 的请求，worker 用 cuBLAS/CUTLASS strided GEMM（改 ld 参数）只算自己的 1/k 逻辑分片，
    #    控制面更新 launch 参数即可，无需搬运/重分片权重
    # 4) Intra-GPU Sharing Engine（数据面，单 GPU 内）：encoder kernel 与 LLM chunk 共跑——
    #    若 LLM chunk 为 decode-heavy/memory-bound（Tag=mem 且 ρ≥ρ0）：SM 分区，
    #    decode kernel 固定跑 SM_dec 切片（TPOT 保护），encoder 用其余 SM；
    #    若为 prefill-heavy/compute-bound：每 kernel 查 profile 表 P 选流，
    #    compute-bound kernel → wide stream（全部 SM），memory-bound/低占用 kernel → narrow stream（q_narrow·SM）
    # 5) GPU 执行：A100 上 encoder 的 ViT 自注意力/FFN GEMM（quadratic 计算、低 HBM）与
    #    LLM prefill（chunked-prefill，FlashAttention 类 kernel）填满对方 SM/HBM 空洞；
    #    decode 阶段在 SM_dec 切片上连续跑，TP 通信等待间隙的 SM 被回收给 co-located 任务
    # 6) 输出：逐请求 TTFT/TPOT 与系统吞吐、E2E latency 日志（MMMUPro/TextVQA trace）
    ```
    作用：把 MLLM 的 vision encoder 从干扰源变成 LLM 的合作者——单 GPU 内按阶段互补性与 kernel 级空洞做 SM/HBM 细粒度共享，跨 GPU 按请求分辨率与并发度动态选 DP/TP，同时解决 encoder 进入 prefill 关键路径造成的 SLO 违例与静态并行度带来的过/欠供给，用相同 GPU 预算获得最高 5.1× TTFT、3.0× TPOT、4.9× E2E、3.4× 吞吐提升。

## Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现是 Tetris，一个面向在线长上下文 LLM Serving 的调度系统，核心是 CDSP（Chunkwise Dynamic Sequence Parallelism，逐块动态序列并行）：把每个请求的 prompt 拆成多个 chunk，为每个 chunk 分配不同的 SP 大小（前段小 SP 用空闲资源碎片提前开跑、后段大 SP 满足长序列计算需求，类似俄罗斯方块填空隙），替代既有"整请求统一 SP 大小"的粗粒度分配。Tetris 由三个组件构成：①**CDSP 推理引擎**——在 prefill-decoding 解耦集群上把全部 prefill 实例连成统一 SP 组（小 TP），decoding 实例用大 TP 的 DP 部署；扩展 zigzag ring attention 支持跨 chunk 的负载均衡（历史 KV cache 在 chunk 实例组间均匀重分布，并复用 ring 通信器做跨层 cache balancing 重叠）；引入 handshake 式 backend 分配管理 prefill→decoding 的 KV cache 传输。②**CDSP 调度器**（C++，递归算法 Algorithm 1/2/3）——基于 FLOPs 延迟模型 T_s(R)=a_s+b_s·L+c_s·(C·L)+d_s·L²（最小二乘离线拟合）递归搜索最优 chunk 划分与实例分配；用 improvement rate（改进率阈值）按实时负载调控 SP 扩张程度，避免过度 SP 扩张。③**simulator 式 improvement rate profiler**（~2.1K 行 Python）——离线按请求长度分布 + Poisson 到达率模拟不同负载，为每个到达率选出最优 improvement rate，在线每 30 秒按观测到达率自刷新。
  - 实验比较：①压力测试（把真实请求 trace 的时间戳缩放模拟不同负载，归一化到 25× light-load 延迟）对比 LoongServe、LoongServe Disaggregated、Fixed-SP Scheduling（SP=8/16），比 TTFT/TBT 的 P50/P99 与最大可持续负载；②TTFT 分布分析（临界请求率下的累计 TTFT CDF）；③吞吐分析（TTFT 约束下的吞吐）；④消融：improvement rate 分析（不同固定 rate vs 动态 rate）、chunking 分析（CDSP vs single-chunk 调度）；⑤调度器开销分析（Algorithm 1 在 SP≤128 时平均/最大延迟 ≤86.8µs，端到端 ≤93.79µs/32.90µs for 8B/70B）；⑥cache transfer 开销（CDSP balancing ≤1.8% 额外开销，CDSP handshake 0.6%-11.8%）；⑦simulator 精度（性能模型误差 ≤7.64%/6.35%，模拟器误差平均 6.9%/2.5%）。主要结果：相对 SOTA baseline，TTFT 最高降低 4.35×（P99，LLaMA3-70B），median TBT 最高降低 40.1%，最大请求容量提升最高 45%（20%-45%），吞吐提升 1.24-3.38×/1.15-1.81×（8B/70B），P50 TTFT 降低 1.64-2.78×/2.86-4.17×。
- 硬件平台是什么，配置是什么。
  - A100 GPU 集群：每个节点 8× NVIDIA-A100-SXM4-80GB（NVLink 互联）、128 CPU 核、2TB 主机内存、8× 200Gbps InfiniBand NIC。LLaMA3-8B 部署在 4 节点（32 GPU），LLaMA3-70B 部署在 8 节点（64 GPU）。P/D 比例为 1:1；8B 模型 prefill TP=1、decoding TP=8，70B 模型全 TP=4。
- 开源Serving框架是什么。修改了什么。
  - 基于开源 Serving 框架 **vLLM**（控制面）与 **PyTorch + Triton-distributed**（推理后端），并复用部分 vLLM 组件；前端用 FastAPI；跨进程通信用 Ray。总实现 ~17.5K 行 C++/Python。修改/新增：①扩展 vLLM 调度器三个接口——initialize_schedule（初始化延迟模型与 improvement rate）、update_schedule（HTTP POST {service_url}/update 更新调度元数据：improvement_rate_mapping、sp_size_candidates、improvement_rate_update_period）、cdsp_schedule（对到达的 prefill 请求调用 Algorithm 1 生成 CDSP 执行计划，构造 per-instance 元数据转发给 local manager）；CDSP 调度器本体用 C++ 写（消除调度延迟），global manager 用 Python + Ray。②初始化分布式集群时（initialize_model_parallel）显式配置 prefill 统一实例池的 SP 大小建立 ring attention communicator，decoding 实例指定 DP 大小。③prefill 计算：扩展 Flash Attention 支持 zigzag ring attention（历史 token），用 NVSHMEM 降低 ring 通信开销；decoding 计算：Flash Decoding + CUDAGraph（消除 kernel launch 开销）。④KV cache balancing 与 prefill-decoding cache 传输用 NCCL（v2.26+ 支持并发 communicator 执行），预留专用 buffer 与 CUDA stream 提升带宽利用率。⑤decoding 调度器扩展 Llumnix 的 "virtual usage"：把正在 cache transfer 的请求的 KV cache 槽视为虚拟占用，新请求路由到 freeness rate（可用槽/活跃 batch 大小）最高的实例。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未提供 Tetris 的开源代码或链接。其 arXiv 预印本（"Optimizing Long-context LLM Serving via Fine-grained Sequence Parallelism"，arXiv:2511.06247）已被撤回以符合所属机构内部发表政策，暂无公开 GitHub 仓库（截至检索时无法确认；注意与同名 ZhaoxuanWu/Tetris 的投机解码仓库区分）。可复用组件均开源：vLLM（https://github.com/vllm-project/vllm）、LoongServe（https://github.com/LoongServe/LoongServe，SOSP 2024 的 ESP baseline）、NCCL（https://github.com/NVIDIA/nccl）、Triton-distributed、Ray。vLLM 使用方式：在线 Serving 框架，接收 HTTP 请求→scheduler 决定执行计划→执行器在 GPU 上跑 prefill/decode；Tetris 在其 scheduler 上叠加 CDSP 执行计划求解。
  - 使用例子（一个 128k 长 prompt 请求到达 4 节点 A100 集群，vLLM/Tetris 框架输入到硬件执行全过程）：
    ```
    # 输入：HTTP 请求（FastAPI 前端收，127k token prompt + 已知长度分布），Poisson 流到达
    # 1) 前端（FastAPI）：POST 请求 → 解析为 prefill 请求 → 交给 CDSP scheduler
    # 2) CDSP scheduler（C++，控制面 global manager）：
    #    - 输入 L=127k、A=∅、S={1,2,4,8}、P=32 个 prefill 实例（各带排队延迟 T_i）
    #    - Algorithm 2：按当前 improvement_rate（30s 刷新，来自离线 simulator 的 rate→最优 rate 映射）
    #      用延迟模型 T_s=a_s+b_s·L+c_s·(C·L)+d_s·L² 估计各 SP 的 TTFT=T_queue+max{T_i}，
    #      只在 TTFT 增益超过阈值时才扩 SP（防止过度扩张）
    #    - Algorithm 1 递归：枚举 (s_current,s_next) 对，Algorithm 3 用排队延迟差设 chunk 延迟预算、
    #      数值求解当前 chunk 长度（如 chunk0 用 SP=2 填 p2,p3 的空隙、chunk1 用 SP=4、chunk2 用 SP=8 跑主计算）
    #    - 递归比较各计划的估计 TTFT，选出最优 chunk 计划 → 构造 per-instance 元数据 → Ray 转发给 local managers
    # 3) 推理后端（PyTorch + Triton-distributed，prefill 实例）：
    #    - 每个 chunk 的 token 按 zigzag 交错到该 chunk 的实例组（扩展的 Flash Attention 做 ring attention）
    #    - 计算新 chunk 前，用 NCCL 把前序 chunk 的 KV cache 均匀重分布到当前实例组（cache balancing，
    #      复用 ring communicator 与下一层 prefill 计算跨层重叠，隐藏传输开销）
    #    - send manager 向 decoding 侧 receive manager 发 handshake 预留传输 backend（防 starvation），
    #      确认后用 NCCL/NVSHMEM 把各 chunk 的 KV cache 流式传到目标 decoding 实例
    # 4) decoding 实例（DP，TP=8，Flash Decoding + CUDAGraph）：
    #    - receive manager 收齐全部 chunk 的 KV cache 后通知 local scheduler
    #    - 用 iteration-level/continuous batching 把请求加入 decoding batch，
    #      每个实例作为一部分请求的 master 跑多请求分布式 decoding，逐 token 输出
    # 5) 输出：TTFT（首 token 延迟）与 TBT（token 间延迟）日志，P50/P99 指标
    ```
    作用：以 chunk 级细粒度 SP 分配同时优化 TTFT 与资源利用率——长请求用后段大 SP 压 prefill 延迟，前段小 SP 利用资源碎片提前开跑，配合负载感知的 improvement rate 控制 SP 扩张，避免 LoongServe 式"整请求大 SP"造成的实例空转，在 prefill 与 decoding 异构并行需求（小 TP prefill / 大 TP decode）下最大化集群吞吐并满足 SLO。

## Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现是 Triage，FTQC（容错量子计算）经典控制栈中的解码器调度中间件，把"如何把有限 M 个物理解码器动态分派到 N 个逻辑量子比特的 syndrome 解码任务"建模为带约束的动态实时调度问题，属于把有限资源池调度给持续到达任务流、满足实时截止时间（SLO 式）并最大化吞吐的系统级调度器（非 LLM Serving 框架，按最相似层次归入 Serving 调度）。核心调度单元是 slice S(t,p)：一个 d×d 逻辑 patch 在 t 时刻一个 d-round 同步测量周期内产生的 syndrome 数据块。离线阶段：Litinski 风格编译器（A game of surface codes）把高层程序编译为 LLI（Low-Level Instruction）流 → 静态分析器单趟构建带注释的 Timeline（每个 unit 存 layer index t、空间坐标 (r,c)、操作标签、6-bit 邻居 mask(t−1,t+1,↑,↓,←,→)、到最近关键同步点的 deadline（无则为∞）、可能的 causal cone 引用）。在线阶段：离散事件模拟器在 syndrome 到达和任务完成两个事件上触发调度器，把 PENDING slice 以无冲突 independent set 方式分派到 M 个物理解码器；同步失败时插入 idle syndrome layer（所有 t≥ℓ 的层右移一层），目标是最小化 idle layer 数 → 最小化逻辑错误率 LER。
  - Triage 是双模调度架构：①稳态模式（steady mode）——轻量启发式，统一优先级函数 P(V)=w_u·Urgency(V)+w_c·Cost-Efficiency(V)（Urgency=1/Deadline，Cost-Efficiency=1/(Degree+1)，w_u=w_c=0.5），并探索 FIFO/EDF/MDF 三种策略；②紧急模式（emergency mode）——Triage Trigger 在任一 PENDING slice 的 deadline 达到 τ_emergency（默认 4）时触发，用"预测性因果锥着色"（Algorithm 1：把 causal cone 的 PENDING slice 放入优先级队列，离散事件模拟推进时钟，贪心选独立集，MDF 排序，O(n log n)）预先计算关键操作 causal cone 的解码计划并按时间表派发；配套 ScopeCap（<100，巨型因果锥回退稳态模式）、增量重规划（新因果锥超出现有 scope 30% 且距上次重规划 ≥2 层才触发）、机会回填（backfilling：由紧急计划算 M_peak，用 M−M_peak−B_bf(t) 的预算派发非关键、因果无关任务）。
  - 实验比较什么：①与 baseline 对比（串行 sliding window [41]、time-parallel window [24]、SWIPER [26] 投机调度 SOTA），在 Bell4（Litinski compact layout，39 逻辑层、5 个 π/8 关键门）上扫解码器数量与相对解码速度 (τdec/τgen)，对比插入 idle layer 数与热力图/最优调度器地图（Triage 在资源受限下界前沿最优，SWIPER 在资源充裕区最优，左下角黑色区为 backlog 终止失败区）；②跨 15 个 FTQC benchmark（QASMBench 派生：T_injection、rotation_C+T、MSD15to1、bell4、mult15_CL/SL、adder28_CL/SL、adder64_SL、adder118_SL、secal1_SL、variational4_SL、qft4_SL、trotter4_SL、ising26_SL）在两种资源场景（并行富余 count=2×#LQs/speed=0.9；延迟富余 count=#LQs/speed=1.8）比 idle layers 与 LER；③随机解码延迟鲁棒性（log-normal 抖动，σ_base=0.3447/α_d=0.0041/α_p=15.03/p_ref=10^-3，按 pymatching 每 shot 延迟校准，LOO 验证 MAE 0.064）；④调度计算开销（Triage 每层 median 亚毫秒，紧急模式 O(n log n) 拟合 y=a·n log n, a=0.01513, R²=0.8056，Delay Ratio 0-0.20 灵敏度模拟，ScopeCap 在 0.06 处避免 backlog 失败）；⑤窗口缓冲大小（资源受限选小缓冲、资源充裕趋近 d/2）、超参数 (w_u, τ_emergency) 敏感性（性能对参数鲁棒）。主要结果：相比标准时间并行 baseline，平均 LER 降低 52.6%；在慢解码器区（τdec>τgen）仍可工作；总执行时间 T_total=N_total_layers×d×T_meas（d=21 时超导约 21μs/层，离子阱/中性原子 2.1-21ms/层）直接受益于 idle 层减少。
- 硬件平台是什么，配置是什么。
  - 无真实量子硬件或 FPGA 加速器，全部为软件仿真。仿真设备：Intel i9-14900K 处理器 + 188 GB RAM，Python 3.9。模拟的是经典控制流水线（编译器→静态分析→离散事件模拟）。解码器延迟建模：profiling pymatching 解码器在不同解码 volume 下的实测延迟，幂律拟合 t_decode=A·(volume)^α（α=1.17），单个 slice 的延迟由其窗口缓冲大小（约束图中未解析邻居数，即 degree）决定。Monte Carlo LER：用 Stim 模拟 d=9 旋转 surface code、circuit-level depolarizing noise p=3×10^-3，外推到 d=21，每点 ≥10^5 runs。平台时间假设：每个逻辑层需 d 轮 syndrome 测量，T_layer=d×T_meas。
- 开源Serving框架是什么。修改了什么。
  - 论文未基于任何开源 Serving/调度框架修改，作者自建调度仿真框架（Python 3.9 离散事件模拟器）。复用的开源组件：pymatching / Sparse Blossom [16]（MWPM 解码器，用于延迟 profiling 与 LER Monte Carlo）、Stim [42]（快速 stabilizer 电路模拟器）、QASMBench [43]（benchmark 来源）、Litinski 风格编译器 [38]（生成 LLI/tile 布局，指令集含 multi-patch measurement、patch rotation、idle）。Baseline 复现：sliding window [41]（一次处理一个 lattice surgery 操作涉及的 slice 块，不提前处理更晚时间的 slice）、time-parallel window [24]（跨时间维并行逻辑 patch，但不拆分多量子比特操作）、SWIPER [26]（复现其 successor-based 投机策略，10% misprediction rate + 10% speculation time，投机解码模块不计入解码器占用）。
  - 修改/新增：①新增 Triage 调度器本体——slice 约束图（最多 6 邻居：2 时间 + 4 空间）、slice 状态机（UNGENERATED→PENDING→OCCUPIED→ASSIGNED→COMPLETED）、Timeline 构造、稳态加权启发式、紧急模式 Algorithm 1 预测因果锥着色、Triage Trigger（τ_emergency 阈值 + ScopeCap<100 + 30% 扩幅 + 最小重规划间隔 2）、机会回填预算 M_usable(t)=max(0,min(M−M_peak−B_bf(t), F(t)−E(t)))；②新增离散事件模拟框架——syndrome 生成、idle layer 插入（把 t≥ℓ 的层右移并新建 idle 层）、backlog 终止条件（插入 idle 层数超过 benchmark 原始层数 10 倍则强制终止）、随机延迟 log-normal 抖动注入、调度延迟灵敏度模拟（Delay Ratio 0.00-0.20，紧急模式用 O(n log n) 拟合函数按实时 scope 大小计算延迟）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：Triage 论文代码截至检索（2026-08）未公开，arXiv:2605.04459（ISCA 2026 录用），未发现官方 GitHub 仓库，无法确认开源链接；作者属 HKUST(GZ) Wang Research Group / QuAIR Lab。可复用开源组件：pymatching（https://github.com/oscarhiggott/PyMatching）、Stim（https://github.com/quantumlib/Stim）、QASMBench（https://github.com/pnnl/QASMBench）。
  - 使用例子（一个含 T 门（非 Clifford）的 lattice surgery 计算在解码器池调度下的全过程，框架输入到性能输出）：
    ```
    # 输入：高层量子程序（如 mult15_SL，15 逻辑比特、586 层、252 个 T 门）+ 解码器池配置（M 个解码器，各 r_dec 倍 syndrome 生成速度）
    # 1) 离线（编译 + 静态分析）：Litinski 风格编译器生成 LLI 流（multi-patch measurement / patch rotation / idle）
    #    → 静态分析器构建 Timeline：每个 unit 记录 (layer t, 坐标 (r,c), 操作标签, 6-bit 邻居 mask, 到最近 T 门同步点的 deadline, causal cone 引用)
    # 2) 在线（量子硬件 + 调度器）：每个 t 产生一层 syndrome（每个逻辑 patch 一个 slice）→ 触发 Triage 调度器
    #    - 稳态：对所有 PENDING slice 按 P(V)=0.5·1/Deadline+0.5·1/(Degree+1) 排序，选冲突无关（independent set）且 ≤M_available 的 slice 分派给空闲解码器
    #    - T 门临近：某 slice deadline ≤ τ_emergency=4 → Triage Trigger 触发紧急模式
    #      * 用 BFS 从关键 slice 的时间/空间前驱反向展开 causal cone（只扩同层空间邻居与 t−1 时间前驱，COMPLETED 剪枝，LRU 缓存）
    #      * Algorithm 1 预测着色：把 cone 内 PENDING slice 按 t_start 入优先级队列，模拟时钟推事件，每步按 degree 升序贪心选独立集，
    #        记录 (t_sim, s) 到计划 P，更新邻居 t_start 与 degree，直到队列空 → 得到紧急计划（≤ScopeCap=100 切片）
    #      * 机会回填：M_peak 之后的空闲解码器按稳态启发式跑非关键任务
    #    - 若因果锥未在关键操作执行前解码完 → 插入 idle syndrome layer（所有 t≥ℓ 层右移），暂停推进；idle 超 10 倍原层数则判定 backlog 失败终止
    # 3) 解码器执行：pymatching/Sparse Blossom 在每 slice 的窗口缓冲（含邻居边界人工 syndrome）上跑 MWPM，延迟按幂律 t_decode=A·volume^1.17 计入
    # 4) 性能输出：idle layer 数（同步失败度量）→ 总层数 N_total_layers → 墙钟时间 T_total=N_total_layers×d×T_meas；
    #    LER：Stim Monte Carlo（d=9, p=3×10^-3, ≥10^5 runs）模拟 windowed lattice surgery 后按层聚合，外推到 d=21
    ```
    作用：在 M<N 共享资源模型下把有限的解码器动态分配给 N 个逻辑比特的持续 syndrome 流——稳态启发式保证平均吞吐与 backlog 可控，紧急模式在非 Clifford 同步点前用最大并行度预解码因果锥、避免 Pauli frame 同步失败导致的 idle stall 与 LER 上升，机会回填回收空闲解码器吞吐，从而在慢/稀缺解码器资源下维持低 idle 层数与低 LER（平均比时间并行 baseline 低 52.6% LER）。

## Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现是系统表征研究（无新增 Serving 代码）：以 vLLM v1 推理引擎（PagedAttention，block size B=16，默认 FCFS 调度）为被测 Serving 系统，通过调节 max_num_batched_tokens 与 max_num_seqs 两个调度参数，系统刻画 reasoning（长 CoT）负载下 Serving 调度的瓶颈与并行策略选择。属于"Serving 调度"最接近层次：论文不改代码、而是量化 vLLM 调度器在 KV 容量压力下的行为（并发-容量权衡、preemption、chunked prefill、admission control），并给出 KV-aware 调度与 DP/TP/PP 并行策略选择指南。论文明确将 kernel 级优化抽象掉，只分析系统级容量与调度动力学。
  - 实验比较什么：①并发-容量权衡：DeepSeek-8B 在单 H200 上把 max_num_seqs 从 1K 扫到 10K（10K 输入序列 batch），对比 TTFT/TPOT/E2E/Waiting-Running 时长与吞吐、HBM 带宽利用、KV 占用时间线，发现 E2E 凸曲线与 ≈2K 并发最优甜点（Observation 1/2）；②DP 扩展：固定 8×H200、DP=8，batch size 从 500 扫到 5000，对比聚合吞吐、HBM 带宽、KV 利用率与 E2E（61s→165s 亚线性增长），证明 DP 无法池化内存（Observation 3）；③DP 从 1→8 GPU 扩展的"stranded capacity"与带宽 sawtooth（40%–85% 振荡）现象（Observation 4）；④DP vs TP vs PP vs 混合并行：8B/14B/32B（batch 2K）下 DP 4.9× vs TP 6.15×（32B），14B 最优 DP=8（332s）、32B 最优 DP=4+TP=2（484s）混合策略（Observation 5）；⑤frontier 模型：Llama-405B 密集模型 TP=8（986s）优于 PP=8（7537s，7.6× 慢），DeepSeek-R1-671B 稀疏 MoE 模型 PP=4+TP=2（1663s）优于 TP=8（2047s）（Observation 6）；⑥8B/70B/671B 参数扩展：吞吐亚线性下降（9× 参数→5–6× 吞吐下降），HBM 利用率 8B≈85% vs 671B≈50–60%（带宽-计算反转）（Observation 7）；⑦prefill vs decode 资源发散：prefill compute-bound（SM 占用高、HBM 带宽 ≈30%），decode bandwidth-bound（HBM 带宽 ≈85%）；⑧KV 缩放与"reasoning cliff"：Llama-405B 在 batch 4K/5K 时 KV 在 prefill 阶段即耗尽（Observation 8/9）。
- 硬件平台是什么，配置是什么。
  - 单节点 8× NVIDIA H200 Tensor Core GPU（SXM5）：每卡 141 GB HBM3e、峰值带宽 4.8 TB/s、FP16/BF16 峰值 1979 TFLOPS；第四代 NVLink + NVSwitch 互连，每 GPU 双向 900 GB/s（TP 的 all-reduce 用）；主机双 Intel Xeon Platinum 8558P + 2 TB DDR5 系统内存。所有实验在单个 NVLink 8-GPU 节点内进行（论文视为现代推理的基本扩展单元，TP 限制在 NVLink 域内，更大部署用 DP 复制节点行为）。
- 开源Serving框架是什么。修改了什么。
  - 开源 Serving 框架：vLLM v1（https://github.com/vllm-project/vllm），启用 PagedAttention（KV 按块管理消除内部碎片），block size B=16；调度策略默认 FCFS，但调节 max_num_batched_tokens 与 max_num_seqs 以刻画并发上限。论文未修改 vLLM 源码，只做参数扫描与遥测；复用 LMCache/Mooncake 等 prefix cache 讨论（非实验主体）。论文自身未开源（arXiv:2605.19775，ISCA'26 Industry Track，无代码链接）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文无 GitHub/复现工件（arXiv 页无代码链接，无法确认）；vLLM 开源；数据集 Meta Natural Reasoning（arXiv:2502.13124，NaturalReasoning，1.15M 多跳推理样本，profiling 抽样 100k：77% prompt 为 50–150 token，45% 响应 >5000 token，43.04% 含 >5000 reasoning token）；模型开源：DeepSeek-R1-Distill-Llama/Qwen 变体（Llama-8B、Qwen-14B、Qwen-32B、Llama-70B，GQA，KV 262 KB/token@32B、328 KB/token@70B）、Llama-3.1-405B（密集 GQA，≈1.05 MB/token）、DeepSeek-R1-671B（MoE 激活 ≈37B，MLA 低秩 latent 压缩 KV）。
  - 使用例子（一个含 100 token prompt、将生成 10k reasoning token 的请求，进入 vLLM 跑 DeepSeek-8B 的 Serving 框架输入到硬件执行全过程）：
    ```
    # 输入：HTTP 推理请求（prompt ≈100 token，来自 Natural Reasoning 数据集的推理题）+ 引擎配置（max_num_seqs=2K、max_num_batched_tokens 受限）
    # 1) 请求准入：vLLM 调度器（FCFS）把请求放入 Running 队列并为其分配 PagedAttention KV 块（block size=16，HBM 上按块表管理）
    # 2) Prefill（compute-bound，决定 TTFT）：对 prompt 全部 token 并行执行矩阵乘（GEMM），H200 tensor core 高占用、HBM 带宽仅 ≈30%；
    #    batch 内先到的其他请求也加入；KV 块按 token 写入（KV 占用上升）
    # 3) Decode（bandwidth-bound，决定 TPOT）：逐 token 自回归，每步从 HBM 读全部权重 + 活跃 KV cache（PagedAttention 按块取）；
    #    算术强度塌缩、HBM 带宽饱和 ≈85%；2K 并发下 TPOT≈0.08–0.48s 区间
    # 4) KV 容量压力：10k reasoning token 输出使每请求 KV 累积，聚合 KV 占用逼近 100% → 触发 vLLM 调度器 preemption
    #    （请求降级到 Waiting 队列 / swap 到 CPU）；恢复时 prefix cache 命中失败 → 全量 prefill 重算惩罚（观测到的 E2E 尾部延迟尖峰）
    # 5) 并行扩展路径：单卡瓶颈 → DP=8 复制模型分请求（每卡独立 HBM，625 请求/卡，仍各自撞容量墙）；
    #    32B+ 转 TP（权重分片 64GB→8GB/卡，释放 133GB/卡给 KV）；671B MoE 用 PP=4+TP=2 混合
    # 6) 遥测输出：nvidia-smi 采 HBM 带宽利用、vLLM 指标采 TTFT/TPOT/吞吐/KV 占用/请求状态机（Waiting/Running）→ 判定 Capacity Trap
    ```
    作用：以 vLLM 作为被测系统，量化 reasoning 负载下"并发提高占用率 vs KV 耗尽引发抢占"的根本矛盾，论证 KV-aware 并发上限、TP 容量释放、混合并行与 prefill/decode 解耦等 Serving 层设计原则（论文核心贡献为决策框架而非代码）。
