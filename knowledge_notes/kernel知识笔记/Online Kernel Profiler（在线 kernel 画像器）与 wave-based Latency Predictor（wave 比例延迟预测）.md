## Online Kernel Profiler（在线 kernel 画像器）与 wave-based Latency Predictor（wave 比例延迟预测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Online Kernel Profiler 是 PowerWeave（ISCA'26）的组件：在 serving 运行过程中透明地逐 kernel 记录执行延迟，构建"kernel 身份 → 频率-延迟行为"的在线画像，为后续 DVFS 频率选择提供依据。它运行在 PowerWeave Interposer（透明拦截 CUDA driver API 的 Rust 层）的专用后台线程中。kernel 身份 key = 函数句柄 + grid/block 维度 + 共享内存大小 + CUDA stream——把"同函数不同序列长度/batch 的实例"区分为独立条目，避免输入相关行为（如 attention 内核随序列长度变化的访存模式）被平均进一条曲线。
- Latency Predictor：对新出现的 kernel 配置（同 kernel 家族不同 grid/thread-block 配置），用 wave 比例从已画像实例泛化延迟：l = waves × (l_old / waves_old)，其中 waves = 总 launched blocks /（每 SM 可驻留 block 数 × 分配给该 kernel 的 SM 数）。直觉：每个 SM 并行驻留若干 block，一个 kernel 的 block 需要多"波"串行执行完，waves 与延迟成正比。论文实测平均误预测 3.9%（prefill 4.55µs vs 平均 118.75µs；decode 0.84µs vs 16µs），对 SLO 安全足够。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 运转流程（伪代码级）：
```
// Interposer 在每次 kernel launch 时：
key = (funcHandle, gridDim, blockDim, sharedMemSize, stream)
startEvt.record(stream); kernel.launch(); endEvt.record(stream)
// 后台线程异步查询事件耗时，避免阻塞关键路径
if 首次见到 key:
    baseline 阶段: 以最大频率执行，记录 latency[key][f_max]
    profiling 阶段: 把 key 分配到多个频率点(1965MHz..915MHz, 12步)执行
                    latency[key][f] = measured  # 每频率点在不同请求中完成
else if key 未见过的变体(同家族不同配置):
    l_pred = waves(key) * latency[donor][f_max] / waves(donor)
    # waves = launchedBlocks / (blocksPerSM * numSMs)
# 进入 operating 阶段后每 kernel 完成后对照预测延迟，
# 偏离 > profiling-threshold(5%) 则重启 profiling
```
- 例子（Llama-3.1-8B decode，148 SM 中 decode 域占 37 TPC=74 SM，blocksPerSM=8）：某 attention kernel 启动 1024 个 block → waves=1024/(8×74)≈1.73；若已画像的 donor 配置 waves_old=1、l_old=16µs，则预测 l≈27.7µs。跨序列长度/batch 的新配置无需重新全量扫描即可得到延迟估计，profiling 开销被摊到 ≈150 个请求（12 频率点×2 轮），单个请求不感知完整成本。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：kernel launch 拦截（LD_PRELOAD 式/CUDA driver API interpose）+ CUDA event 对异步计时（不阻塞关键路径）+ 后台线程执行 profiler/predictor/controller。kernel 执行通常数百 µs 到数 ms，profiling 对单个 kernel 选择性监控、影响最小。新 kernel 无 donor 时保守用最大频率直到其运行时间占比被评估（论文：这类 kernel 平均占 1.9% 运行时间，低于 5% 重画像阈值，实验从未触发重画像）。在线画像使系统无需离线 profiling 即可适应：不同 batch/序列长度/模型的频率-延迟行为被持续学习，权重（kernel 占应用运行时间比例）在线更新以跟踪 prefill/decode 比例变化。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management
