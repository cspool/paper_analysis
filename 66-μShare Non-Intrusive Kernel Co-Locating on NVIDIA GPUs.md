论文标题：μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

GPU运行时的设计，运行时将kernel启动划分为half-plus block，允许SM并行资源需求不同的kernel，提高资源使用率。

开源仓库确认：

- 状态：未找到明确开源仓库
- 链接：N/A
- 说明：本地 PDF、HPCA 2026 官方页面和 DBLP 记录确认了论文与 DOI 信息；HPCA 页面给出论文摘要、作者和报告时间，DBLP 给出 HPCA 2026 论文记录与 DOI/开放访问入口，但未给出代码或 artifact 仓库链接。基于标题、μShare/uShare/mushare 和 GitHub 关键词检索，未发现可判定为作者官方发布的仓库。
- 参考页面：https://2026.hpca-conf.org/details/hpca-2026-main-conference/10/-Share-Non-Intrusive-Kernel-Co-Locating-on-NVIDIA-GPUs
- 参考页面：https://dblp.org/rec/conf/hpca/HuangDZZWLWCTCZLL26

1、论文工作：

- 论文要解决的核心问题：NVIDIA GPU 的硬件 block scheduler 不理解不同 kernel 的微架构资源需求，常把同一个 kernel 的多个 block 堆叠放进同一个 SM，形成 stacked co-location。由于同一 kernel 的 block 往往消耗相同类型的 SM 内资源，例如 Tensor Core、LD/ST、INT32、FP32、FP64 或 SFU，这会让某一种资源很忙，而其他资源闲置。论文把这种现象概括为 “1 more, 5 less”：一个硬件资源利用率高，其余五类低。论文举例说明，常见矩阵乘法 kernel 的 Tensor Core 利用率可达 88.52%，但其他五类资源平均只有 5.45%。
- 论文的主要贡献：μShare 提出一种不修改 NVIDIA 闭源 scheduler、不修改用户 kernel 代码的 intra-SM kernel co-location 系统。它用 half-plus blocksize shaping 让同一 kernel 的大 block 无法继续堆叠在同一 SM 内，从而被迫分散到不同 SM；再用 time-shifted launching 把资源互补的小 block 延迟或重新发射到合适窗口，与 half-plus block 共享同一个 SM。系统还包含 kernel profiler、LD_PRELOAD kernel interceptor、block shaper 和基于 SLO 的 batch manager。
- 论文所处背景：现有 GPU 共享机制多在 inter-SM 层面工作，例如 MIG、MPS、CU masking 或模型级容量控制，它们能隔离或共享 GPU，但难以提高单个 SM 内 FP32/FP64/INT32/LDST/SFU/Tensor 等细粒度资源的同时利用率。更激进的方案通常依赖 kernel fusion 或硬件 scheduler 改造；前者需要读写或融合用户 kernel，跨用户场景不可行，后者受 NVIDIA GPU 闭源硬件限制，只能在模拟器或定制硬件中验证。

2、相对 Baseline 解决的问题与设计方法：

- Baseline 的具体问题：INFless 主要根据模型资源容量做共置，仍会出现 stacked co-location，不能显式利用 SM 内不同低级硬件单元的互补性。Orion 通过 launch time 控制计算密集与内存密集 kernel 的共置，但策略保守，并且同样没有打破同类 block 在 SM 内堆叠的问题。Tacker 这类 kernel fusion 能做 intra-SM 共置，但需要侵入式融合 kernel，论文只把它作为局部对比，并指出它不能直接管理多模型共置时的 SLO。
- 论文的设计方法：μShare 先离线 profiling 每个 kernel 的九元组资源画像，包括 FP32、FP64、INT32、LDST、SFU、Tensor 利用率，共享内存、寄存器使用量和 launch time。在线阶段用 LD_PRELOAD 截获 CUDA launch 函数，把 kernel 分成需要立即加速的集合 X 和可调整 relaunch time 的集合 Y。对 X 中 slack 紧张的可修改 kernel，设置 half-plus blocksize；对 Y 中 kernel 保持默认 blocksize，但只在它与当前执行 kernel 的六类资源利用率相加不超过 100%，且 shared memory/register 足够时发射，否则等待 beta 微秒后重新检查。
- 方法如何对冲 Baseline 缺陷：half-plus blocksize 的关键作用是改变 scheduler 可见的 block 粒度，而不是改变 scheduler 本身。以 A40 为例，每个 SM 可容纳 1536 threads，half-plus 最小设置为 800 threads；这样一个 SM 里不能同时放两个同一 kernel 的 800-thread block，避免 identical blocks 继续堆叠。SM 剩余线程空间则可容纳另一个默认小 block，使资源互补的 kernel 在同一 SM 内并行。对于 cuDNN/cuBLAS 等 blocksize 隐藏或修改后会出错的 unmodifiable kernels，μShare 不改 blocksize，而是通过 time-shifted launch 把它们安排到更互补的窗口。
- 关键 trade-off：μShare 接受了离线 profiling 成本和在线拦截控制复杂度；它的收益依赖可修改 kernel 的比例，论文统计 10 个模型的 6802 次 kernel 执行中，modifiable kernels 占 51.63%，unmodifiable kernels 占 48.37%。它还在 throughput 与 SLO 之间显式折中：更激进的 batch 增长和 block shaping 提高吞吐，但可能提高 SLO violation；更保守的参数能把 SLO violation 降低到 baseline 级别附近，但牺牲部分吞吐。

3、论文实现：

- Baseline 如何实现：论文以 INFless 和 Orion 作为主要 state-of-the-art 对比。INFless profile 模型资源容量需求，并在 GPU 容量允许范围内共置模型；Orion 基于 kernel 的计算/内存资源特征控制 launch time，限制干扰；Tacker 作为 intra-SM kernel fusion 对比，只在其提供的 ResNet50 和 BERT fused models 上比较，并关闭 μShare 的 latency management 以保持公平。论文还比较 CUDA Graph，指出在 large-batch 或 co-location 场景中，kernel 执行时间主导，launch overhead 优化带来的收益有限。
- 新设计如何实现：μShare 以 PyTorch 2.2.0 为 inference framework，把系统组件编译为 .so，通过 LD_PRELOAD 加载进 PyTorch 进程。kernel interceptor 用 dlopen/dlsym 获取 libcudart、libcublas、libcudnn 等动态库中的 launch 函数地址和参数；block shaper 用 shm_open 创建共享内存，并通过 mmap 读写 kernel_process 上传的 blocksize 等参数，再把修改后的参数交回原始 CUDA 函数继续执行。为了支持自定义 blocksize，论文把 PyTorch 的 C10_LAUNCH_BOUNDS(blocksize) 限制设置到 CUDA 上限 1024。
- 实验 / 实现平台：实验部署在 8 台服务器，每台包含 Intel Xeon Gold 6338 CPU、251GB 内存，以及 NVIDIA A40 或 A800 GPU。A40 配置为 84 个 SM、44.784GB 显存、每 SM 1536 threads、102400 bytes shared memory、65536 registers，CUDA 11.8；A800 配置为 108 个 SM、80GB 显存、每 SM 2048 threads、167936 bytes shared memory、65536 registers，CUDA 12.1。工作负载来自 MLPerf 和 PyTorch benchmark 的 10 个模型，包括 Llama2-7b、GPT-2、BERT、ResNet50、MobileNet v2、Swin Transformer、Vision Transformer、Yolostiny、ResNet101 和 EfficientNet B7；大多数模型 SLO 设为 200ms，Llama2-7b 设为 400ms，输入/输出长度固定 10 tokens。
- 关键实验设置与指标：论文使用 INFless 的 Azure inference traces，并按 GPU 数量缩放；每个模型运行 4 个 replica，共 40 个 replica 分布到 8 个 GPU。A40 上 μShare 相比 INFless 和 Orion 提升 system throughput 26.90%-54.09%，normalized throughput 为 58.91，高于 INFless 的 46.42 和 Orion 的 38.23；峰值 throughput 从 INFless 的 1722 QPS 提升到 μShare 的 3046 QPS。A40 上平均 SLO violation 为 3.35%，高于 INFless 的 2.05% 和 Orion 的 1.12%，但参数 v7 可把 violation 降到 0.84%，同时仍保持 19.28%-44.83% throughput 提升。低级硬件平均利用率方面，μShare、INFless、Orion 分别为 15.10%、10.90%、9.37%，对应 38.53%-61.15% 提升。A800 上 μShare normalized throughput 为 99.39，高于 INFless 的 85.35 和 Orion 的 65.26，提升 16.45%-52.29%，平均 SLO violation 为 3.40%。

4、pipeline/kernel 解析：

- 新 pipeline/kernel 是什么：论文没有提出新的 CUDA kernel 算子本身，而是提出了一个非侵入式 kernel co-location pipeline：offline profiler 生成 kernel 资源画像，online interceptor 截获 launch，shaper 决定 half-plus/1-3-plus blocksize 或 time-shifted relaunch，batch manager 根据 SLO 反馈调整请求 batch size。这个 pipeline 的核心是让闭源 NVIDIA scheduler 在不暴露内部接口的情况下，通过 launch 参数和 launch timing 间接得到更好的 block 分散效果。
- 新 pipeline/kernel 的执行流例子：以两个推理 kernel 为例，一个 late kernel 的 launch slack 较小，被划入集合 X。μShare 拦截它的 cudaLaunchKernel 后，在 A40 上把 blocksize 改成 800 threads，也就是略大于 1536/2。由于一个 SM 无法再同时容纳两个这样的 block，该 kernel 的 blocks 会分散到多个 SM，每个 SM 留下不足 736 threads 的剩余空间。随后另一个默认小 blocksize 的 normal kernel 被划入集合 Y，shaper 查询 profiler 中两者的 FP32/FP64/INT32/LDST/SFU/Tensor 利用率、shared memory 和 register 需求；如果资源相加不过载，就立即 relaunch，否则等待 beta=10 微秒后重试。这样，一个 SM 内可能同时运行 half-plus block 和资源互补的小 block，使 Tensor、LDST、INT32 等资源更均衡地被占用。
- A800 上的变体：A800 每 SM 有 2048 threads，CUDA 单个 block 上限仍为 1024，因此 half-plus 无法阻止两个大 block 被放入同一 SM。论文改用 1/3-plus shaping，最小 blocksize 为 704 threads，使同一 SM 最多放两个大 block，剩余空间小于三分之一，再让小 block 与其共置。该策略仍能制造 scattered co-location，但论文也指出它比 A40 half-plus 的效果略弱，因为同类资源 block 的 SM 内占比上限从 1/2 放宽到 2/3。
- 局限与适用范围：μShare 更适合以 CUDA kernels 执行、且存在不同资源类型互补的多模型推理或混合科学计算场景。若 workload 中可修改 blocksize 的 kernel 很少，收益会退化为较普通的 kernel-level resource-coupled launch control；若 kernel 修改 blocksize 会导致语义错误，系统只能依赖 time-shifted launch。论文报告 online 控制开销很小，单 kernel 平均控制开销约 60.35ns，CPU 开销约单核 6.85%，但离线 profiling 需要 105-393 秒，Llama2-7b 由于 kernel 数更多需要 7160 秒。
