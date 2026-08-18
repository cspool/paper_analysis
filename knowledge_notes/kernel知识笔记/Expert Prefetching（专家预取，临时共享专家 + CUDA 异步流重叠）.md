## Expert Prefetching（专家预取，临时共享专家 + CUDA 异步流重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Prefetching 是在 CPU-GPU 异构 MoE 推理中，把将被激活的专家权重从 CPU 主存提前经 PCIe 加载到 GPU 并与计算重叠、以隐藏专家取数延迟的运行时机制。STEP 中 profiling（Qwen3-30B-A3B、A100、INT8）显示专家取数占 MoE 推理执行时间约 88%，远高于 gating/计算/聚合合计的 12%，因此隐藏 expert-fetch 延迟是核心优化。STEP 的预取策略：①把窗口内票选的 top-c 临时共享专家在计算开始前整体预取并常驻 GPU（结构 j+c shared + k−c routed 后每步动态加载从 k 降到 k−c）；②预取实现为独立 CUDA stream 上的异步数据传输 kernel（cudaMemcpyAsync H2D），在专家计算 kernel 之前发起，借助 CUDA 非抢占式 kernel 执行让传输与计算并发；③每个解码步序列的最后一个预取 kernel 后插入 CUDA event 记录完成，CPU 用非阻塞查询（cudaEventQuery）保证数据可用同时避免同步阻塞。预取质量由命中率（Prefetch Hit Rate）衡量：CNN/DM 85.5–98.8%、LongBench 72.1–95.6%（Table II-IV），窗口自适应在准确率 >75% 时保持积极预取、<40% 时减窗甚至停用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 每层 decode step（Mixtral top-2 → STEP 后 1 shared + 1 routed 为例）
# stream 0（计算流）                    # stream 1（预取流）
launch(gating_kernel)               
                                        launch(cudaMemcpyAsync, H2D,  # 临时 shared 权重
                                               elected_expert_w, 0)   # 已常驻则跳过（hit）
topk_idx = topk(gate_out, k-c)
                                        launch(cudaMemcpyAsync, H2D,   # 非驻留 routed 权重
                                               routed_expert_w)
launch(expert_gemm, stream=0)        # 与 stream1 传输并发（非抢占 kernel 执行）
...（每步最后一个预取 kernel 后）
                                        cudaEventRecord(ev, stream=1)
cudaEventQuery(ev)  # CPU 非阻塞轮询，数据就绪后继续，避免 blocking 同步
```
Annotations：stream 0/1=两条 CUDA 流（默认流 + 独立预取流），H2D=host-to-device 异步拷贝，cudaEventRecord/Query=记录/查询事件（非阻塞同步原语），hit=权重已在 GPU 显存（命中时不发起传输）。重叠条件：预取 kernel 先于计算 kernel 入队、流间无依赖（非抢占 kernel 执行允许并发），事件提供"就绪查询"而非"等待"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：pinned memory（cudaHostAlloc/cudaMallocHost）保证 H2D 异步传输可用；cudaMemcpyAsync + 独立 stream + cudaEvent 组合；预取目标来自窗口投票选举（临时共享）或 gating 预测。STEP 部署在 HuggingFace Transformers 推理路径上（batch=1 实时推理），与 EP 正交（每 EP group 独立预取，peer GPU HBM 可作二级缓存）。评估方法：以 Cached Expert Ratio（CER 25%/50%/75%）控制显存约束，测 TTFT（prefill）/TPOT（decode）与命中率；与 MoE-Infinity（activation-aware 预取）、HybriMoE（CPU-GPU 调度+缓存）、AdapMoE、DAOP、APTMoE、MoE-Lightning、llama.cpp 对比，decode 平均几何加速 1.54×–2.22×。Fig.14c 定量：命中率 >75% 时增大预取数显著降延迟，<40% 时过度预取浪费带宽——这是自适应窗口阈值（th_s/th_f）的 kernel 层依据。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
