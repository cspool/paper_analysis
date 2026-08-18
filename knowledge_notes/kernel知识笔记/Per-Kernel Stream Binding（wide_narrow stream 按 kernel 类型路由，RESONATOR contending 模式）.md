## Per-Kernel Stream Binding（wide/narrow stream 按 kernel 类型路由，RESONATOR contending 模式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-Kernel Stream Binding 是 RESONATOR Intra-GPU Sharing Engine 在 contending 场景（encoder 与 compute-bound prefill chunk 共跑）下的 kernel 级调度机制：不做静态 SM 分区，而是让两个任务都看到完整 SM 池，在 kernel 粒度控制资源使用。每个任务 T∈{enc,llm} 预创建两条 CUDA 流——wide stream s_T^wide（SMCTRL.SetQuota 1.0，可用全部 SM）与 narrow stream s_T^narrow（SMCTRL.SetQuota q_narrow，0<q_narrow<1，只跑窄 SM 子集）。每个到达 kernel k 查 profile 表 P 的 TYPE(k)∈{comp,mem}：compute-bound kernel 路由到 wide stream（占满 SM 填计算空洞），memory-bound/低占用 kernel 路由到 narrow stream（限制在小区间、不阻塞全局计算）。这样 encoder 与 prefill 的重 kernel 以 time-space sharing 填满 GPU，轻/带宽型 kernel 不干扰——对应论文"单一全局流或单一静态 SM 分区无法利用 kernel 级空洞"的洞察。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 1（RESONATOR）的 per-kernel 路由伪代码：
```
Input: 到达 kernel k，任务 T∈{enc,llm}
State: kernel profile 表 P，SM manager SMCTRL
Streams: 每任务 wide 流 s_T^wide、narrow 流 s_T^narrow（预创建）
InitContendingMode:
  for T in {enc,llm}:
    SMCTRL.SetQuota(s_T^wide, 1.0)        # wide 流可占全部 SM
    SMCTRL.SetQuota(s_T^narrow, q_narrow) # narrow 流只占 q_narrow 比例 SM
DispatchKernel(k, T):
  τ ← P.TYPE(k)                            # τ∈{comp,mem}
  if τ = comp: s ← s_T^wide                # compute-bound 用全 SM
  else:        s ← s_T^narrow              # memory-bound/低占用 用窄 SM
  LaunchOnStream(k, s)
```
Annotations：kernel 类型由离线 profile 表给出（每 kernel 典型 SM 用量与 HBM 带宽），运行期只做查表+选流；wide/narrow 流与 SM 配额预创建，dispatch 只加一次元数据查找与流选择；因 launch 流运行期才定，contending 路径用 eager 执行而非 CUDA Graph 重放（对 compute-heavy encoder+prefill 场景 CPU launch 开销相对 kernel 时间可忽略）；narrow 流绑 SM 子集依赖 green-ctx/libsmctrl 机制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖"把 CUDA 流绑定到指定 SM 子集"的运行时支持：RESONATOR 用 green-ctx 或 libsmctrl（[26]，Bakita & Anderson RTAS'23 的硬件计算分区）——libsmctrl 通过修改 CUDA stream 内部 metadata（GPC 配置掩码）让 GigaThread Engine 只把 grid 分到 mask 内 SM，mask 更新开销 ~4us（微秒级 SM 重分区），比 GreenContext 的 context 切换便宜。使用场景：一切"两个 compute-heavy 负载共享一卡、靠 kernel 特性互补"的 GPU 服务；RESONATOR 用它实现 encoder+prefill 共存（Figure 11 消融：Stream-based Sharing 相对静态 SM Partitioning 在 TTFT 上最高 6.5× 提升、平均 1.6×，因为 compute-bound 阶段 co-scheduling 更有效）。局限：仅在 contending（compute-bound chunk）场景启用，complementary decode 路径仍用 SM 分区以保护 decode 尾延迟。

涉及论文标题：
- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
