论文标题：TetriServe: Efficiently Serving Mixed DiT Workloads

    开源仓库确认：
        - 状态：已找到
        - 链接：https://github.com/DiT-Serving/TetriServe
        - 说明：论文首页脚注明确写出 “TetriServe is available at https://github.com/DiT-Serving/TetriServe”。本次检查时该 GitHub 仓库可访问，仓库 README 描述其目标为面向 DiT serving 的 dynamic GPU allocation 与 SLO-aware scheduling，并给出 FLUX.1-dev、Stable Diffusion 3、H100/A40 等复现实验入口。因此可判定为论文官方开源仓库。

    1、论文工作：
        - 论文要解决的核心问题：面向在线 Diffusion Transformer（DiT）图像生成服务，在共享 GPU 池中处理混合分辨率、混合 deadline 的请求时，如何提高 SLO Attainment Ratio（SAR）。DiT 推理由多个去噪 step 组成，每个 step 都在完整 latent token 序列上执行，2048×2048 乃至 4096×4096 请求的单 GPU 延迟很高；但模型本身通常能放入单张 H100，因此瓶颈不是模型容量，而是如何在 compute-bound 的多 step 推理中为不同请求选择合适的 sequence parallelism（SP）度数。
        - 论文的主要贡献：第一，将 DiT serving 形式化为 step-level GPU scheduling 问题，并证明该问题在离线特例下也是 NP-hard。第二，提出 TetriServe，用 deadline-aware round-based scheduler 在固定长度 round 内动态决定每个请求本轮使用多少 GPU、执行多少 step。第三，结合 profiling cost model、DP request packing、GPU placement preservation、elastic scale-up 和 selective continuous batching，提高混合 DiT workload 的 SAR。第四，在 FLUX.1-dev 与 SD3 上、8×H100 和 4×A40 平台中对比 xDiT fixed-SP baseline，最高报告 32% SAR 提升。
        - 论文所处背景：DiT 已成为 Stable Diffusion 3、FLUX.1-dev 等高质量图像生成模型的核心架构。与 LLM serving 不同，DiT 推理无 KV cache，状态主要是 latent，中间 step 的运行时间稳定且可 profile；同时 DiT serving 的输入分辨率通常来自有限集合，如 256、512、1024、2048，但不同分辨率的 token 数、TFLOPs 和并行扩展效率差异很大。这使得传统 LLM serving 的 prefill/decode、KV cache 管理或模型切分思路不能直接解决问题。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：论文主要 baseline 是 xDiT 的固定 sequence parallelism 配置，即每个请求从开始到结束都使用同一个 SP 度数，例如 SP=1/2/4/8。低 SP 适合小分辨率请求，因为通信开销低、不会过度切分小 tensor，但会让大分辨率请求执行太慢，并在多 GPU 服务器上留下空闲资源。高 SP 能加速大分辨率请求，但对 256×256、512×512 这类小请求会引入过高 all-to-all / ring communication、kernel launch 和低 occupancy 开销，还会造成 head-of-line blocking。xDiT 还被论文描述为 non-preemptive：请求一旦以固定 SP 启动，就持有对应 GPU 直到完成，无法在 step 之间按 deadline 重新塑形资源。
        - 论文的设计方法：TetriServe 使用 step-level sequence parallelism，把“一个请求用多少 GPU”从请求级固定配置改为 step/round 级调度决策。系统把连续时间离散成固定长度 round；在每个 round 开始时，先通过离线 profile 得到的 per-step latency lookup table，为每个请求计算满足 deadline 所需的最小 GPU-hour 分配候选；再把本轮请求选择问题转化为 group-knapsack 形式，用动态规划选择哪些请求以哪种 GPU allocation 执行，使下一轮开始时“不会确定变 late”的请求数量最大化。
        - 方法如何对冲 Baseline 缺陷：对小分辨率或 deadline 较宽松的请求，TetriServe 倾向使用较少 GPU，避免 fixed high-SP 的通信和资源浪费；对大分辨率或 deadline 紧迫的请求，则临时提高 SP 缩短关键路径。round 边界提供了轻量级 preemption 点，使请求可以在不同 step/round 之间改变并行度。DP packing 显式面向 deadline，而不是只按吞吐或先来先服务排序，因此能减少 fixed-SP 策略在混合 workload 中的跨分辨率失衡。
        - 关键 trade-off：round 越短，调度越灵活，新请求 admission delay 越低，但控制面调用和并行重配置更频繁；round 越长，调度开销更低，但 non-preemptible 区间变长，可能错过 deadline。step granularity 也有类似 trade-off：论文实验显示过细的 1-step 控制会带来较高 overhead，过粗的 10-step 控制又不够灵活，中等粒度（如 5 step）更稳健。另一个 trade-off 是 GPU-hour 最小化与利用率之间的平衡：deadline-aware allocation 先避免过度分配，elastic scale-up 再把剩余 idle GPU 分给确实能受益的请求。

    3、论文实现：
        - Baseline 如何实现：xDiT baseline 使用固定 SP 度数运行 DiT 推理，请求全程保持同一个 GPU 数；论文评估 SP=1、2、4、8。RSSP（Resolution-Specific SP）是更强的静态 baseline，通过 offline profiling 为每个分辨率选固定最优 SP：论文中给出的配置是 256×256 和 512×512 使用 SP=1，1024×1024 使用 SP=2，2048×2048 使用 SP=8。RSSP 能感知分辨率，但仍不感知单请求 deadline，也不能在请求执行中动态调整。
        - 新设计如何实现：TetriServe 共 5,033 行 Python/C++。它复用 xDiT 的 sequence parallelism engine，复用 vLLM 的异步逻辑，以及 MuxServe、SGLang 的进程启动组件；scheduler 的核心 decision loop 用 C++ 实现并通过轻量绑定暴露，以达到毫秒级控制面延迟。系统组件包括 Request Tracker、Scheduler、Execution Engine 和 Latent Manager。实现上还加入 VAE decoder sequential execution，以避免高分辨率 batch 解码时的峰值 activation memory；NCCL process group 采用部分常用 group 预热、其他 group 按需 warmup，避免一次性初始化所有 group 导致 HBM 占用过高；latent transfer 用 Future-like abstraction 异步传递中间 latent 和元数据。
        - 实验 / 实现平台：论文在两个 GPU 集群上评估。第一类节点为 8×NVIDIA H100-80GB HBM3，NVLink 4.0，GPU 间带宽 900 GB/s；第二类节点为 4×NVIDIA A40-48GB，其中 GPU 成对通过 NVLink 连接，并经 PCIe 4.0 与 host 连接。软件环境基于 NVIDIA NGC container、CUDA 12.5、NCCL 2.22.3、PyTorch 2.4.0、xDiT git-hash 8f4b9d30。模型选择 FLUX.1-dev 和 Stable Diffusion 3 Medium，prompt 从 DiffusionDB 采样 300 条，默认 Poisson 到达，12 req/min。
        - 关键实验设置与指标：主指标是 SLO Attainment Ratio（完成 deadline 内请求的比例），并用 latency CDF 观察尾延迟。分辨率混合包括 Uniform（256、512、1024、2048 等量）和 Skewed（按 latent length 指数加权，更偏向大分辨率）。基础 SLO 设定为 256×256: 1.5s、512×512: 2.0s、1024×1024: 3.0s、2048×2048: 5.0s，并扫 SLO Scale 1.0× 到 1.5×。论文报告 TetriServe 在 Uniform 和 Skewed workload 中均超过固定 SP 与 RSSP；在 FLUX/H100 上，平均相对最佳 fixed strategy 提升约 10%（Uniform）和 15%（Skewed），紧 SLO 下最大例子达到 28% 或 32% SAR 提升。消融显示 placement preservation 与 elastic scale-up 都重要，latent transfer overhead 低于 per-step latency 的 0.05%。

    4、pipeline/kernel 解析：
        - 新 pipeline/kernel 是什么：论文没有提出新的 CUDA kernel，而是提出一个新的 DiT serving runtime path：deadline-aware round-based step-level SP scheduling pipeline。核心不是改变 DiT layer 内部算子，而是在 diffusion step 边界动态改变请求的 GPU allocation，并用 DP packing 在每个 round 内选择请求集合。可把它理解为“以 diffusion step 为调度单位、以 round 为控制面周期、以 SP degree 为资源旋钮”的 serving pipeline。
        - 新 pipeline/kernel 的执行流例子：一个请求到达后，Request Tracker 记录分辨率、deadline 和剩余 diffusion steps。进入下一个 scheduling round 时，Scheduler 查询 cost model，知道该分辨率在 SP=1/2/4/8 下每个 step 的耗时和 GPU-hour。若请求还有 50 个 step 且 deadline 较紧，scheduler 会计算若本轮使用 1/2/4/8 GPU 分别能推进多少 step，以及不推进会不会在下一轮成为 definitely late。随后 DP Round Scheduler 在所有 pending requests 的候选 option 中选择一组不超过总 GPU 数的 plan，例如让一个中等请求本轮用 2 GPU 跑 15 step，让一个小请求用 1 GPU 跑 20 step。Execution Engine 把这些 step 分派给对应 GPU workers；workers 运行 xDiT sequence parallelism 计算，Latent Manager 保存和异步转移中间 latent。round 结束后，请求状态更新；若同一请求下一轮继续执行，placement preservation 尽量让它留在同一组 GPU 上，减少 state transfer 和 group remapping stall；如果仍有空闲 GPU，elastic scale-up 会把额外 GPU 临时分给有收益的请求。所有 diffusion steps 完成后，VAE decoder 逐请求解码最终图像并返回给用户。

