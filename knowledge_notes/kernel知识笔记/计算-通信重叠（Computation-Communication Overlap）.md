## 计算-通信重叠（Computation-Communication Overlap）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
计算-通信重叠是分布式训练/推理中的核心优化：让通信（如 All-to-All dispatch/combine、AllReduce）与计算（如专家 GEMM、attention）在同一时间窗口内并行执行，用计算隐藏通信延迟，从而把吞吐推近"通信零暴露"的理想水平。对 MoE 专家并行尤其关键——forward pass 平均 47% 的执行时间花在 All-to-All dispatch/combine 的 device-to-device 数据交换上，瓶颈从计算转向通信。MoE-Hub 论文按粒度把现有重叠方法分为两类并指出其共性缺陷：粗粒度（FasterMoE/Tutel 张量切片流水）在计算图级 pipeline，但因 MoE 动态路由导致通信量/专家负载每轮变化，产生 pipeline bubble；细粒度（Comet/CCFuser 等 kernel 融合、tile/指令级调度）重叠效果好，但需大量硬件专属的软件编排（同步、内存屏障、SM 分组），开销大、可移植性差。二者合计的调度开销+暴露通信占 MoE 层时间 >24%，即使最优实现也达不到理想层性能。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# baseline 细粒度重叠（Comet 风格）：专用 SM 做通信 + 软件就绪轮询
# 阶段1：地址解析 kernel（dispatch 前必须完成）
allreduce(index_expert)                  # 跨 GPU 同步确定 per-token 目标地址
shuffle_and_layout(tokens, addr_table)   # 重排 token、CPU/GPU 计算偏移
# 阶段2：dispatch 与 expert GEMM 重叠
for tile in tokens:
    if SM_group_A: send(tile, dst_gpu, addr)     # 专用通信 SM 子集
    if SM_group_B: while !poll(ready[tile]):;    # 消费者轮询等数据
                   gemm1(tile); gemm2(tile)      # 计算 SM 子集
# 阶段3：combine 反向重叠 + 轮询
```
Annotations：baseline 在"发数据前"必须先做地址解析（同步+shuffle，无法与计算重叠）；细粒度重叠虽让 dispatch 与 GEMM 并行，但消费者靠原子轮询 semaphore 检测就绪，busy-wait 的 warp 占用带宽与算力；且一旦某 token 的地址/顺序未就绪，GEMM tile 无法开始，暴露通信。

```
# MoE-Hub：硬件接管控制平面后的重叠（routing→GEMM1 窗口）
# 生产者（routing kernel）
for token, (e, gpu) in route_result:
    st.rowsp(MallocID[e], RowID=token, RowOffset)   # 立即发出，无需地址
# hub 硬件（透明）
RPM:  合并/整形 → 调度传输（round-robin + 最小 RowID 优先）
AAU:  到达即分配地址（RAT/APT，BaseAddr+LocalRowID*RowSize+RowOffset）
DAM:  写应答→Dependency Table→TB 计数器→Ready→派发 TB
# 消费者（专家 GEMM，CUTLASS）
for tb in ready_tbs:  gemm1(tb); act; gemm2(tb)    # 数据一就绪即执行
```
Annotations：生产者拿到路由结果立即发 st.rowsp（无地址解析、无同步）；RPM 保证整行连续到达；AAU 按到达密集打包；DAM 用写应答自动触发 TB 派发，无轮询。routing→GEMM1 窗口内通信被计算完全隐藏。消融：RPM（MH-PKT）平均 1.13×、DAM（MH-DEP）平均 1.14×、全设计达理想 MoE 层 96.8%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现：粗粒度——算子级张量切片流水（FasterMoE、Tutel 的图级 overlap）；细粒度——kernel 融合（CCFuser 用 inter-GPU shared memory 把 All-to-All 与 GEMM 融进一个 kernel、FlashDMoE 单 kernel）、tile 级调度+专用 SM（Comet）、GPU-resident 地址同步（Primus-Turbo）；工业库——DeepEP 的 NVLink/RDMA token 分组路由重排。使用上，软件方案要求开发者手写数百至数千行调度/通信代码（Table I）并做硬件专属调优。MoE-Hub 的做法：把重叠能力下沉为硬件原语——软件只发 st.rowsp + 用 rowspMalloc 注册区域，RPM/AAU/DAM 在 hub 透明完成"整形、寻址、就绪信号"，实现无缝透明重叠且调度代码 0 行、通信指令 <10 条。MoE-Hub 论文还指出重叠优化与计算侧 MoE 优化（并行策略、专家不均衡缓解、expert 复制/动态放置、TP+EP 混合并行）正交可叠加。

RoCC 补充视角（ISCA'26，ROP 硬件的细粒度计算-通信重叠）：RoCC 把重叠的执行引擎从 SM 换成 GPU 的 ROP 硬件——warp 完成自己 GEMM tile 后立即发 rocc_allreduce 触发 ROP 做 CC、继续算下一 tile，实现 warp 级（tile 级）细粒度重叠，SM 无需为 CC 让出算力（对比 oracle 软件重叠 20% SM 专做 CC 导致 GEMM 指数级减速：80% SM 时 GEMM 慢 20%）。实测：RoCC 平均重叠率 83.4%（未重叠部分为首段 GEMM 与末段 CC），并发 CC 仅使 GEMM 慢 6.25%，相对 oracle 软件重叠（GEMM 与 CC 完美重叠、SM 分区）平均高 23%，相对顺序 baseline 平均 51%；端到端（Astra-Sim+Chakra）44%，32-256 GPU 13%-21%。核心是门铃同步 + ROP 异步执行，避免软件方案的 cache/NoC 污染与 busy-wait 轮询。
涉及论文标题：
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication
