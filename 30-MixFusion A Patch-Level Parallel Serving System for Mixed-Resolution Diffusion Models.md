论文标题：MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

diffusion Serving。

本地条目说明：
    - 本地编号：paper_2026 第 30 篇
    - 本地 PDF：paper_2026/30-MixFusion A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models.pdf
    - 本地文本抽取：paper_2026/30-MixFusion A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models.txt
    - 发表信息：PPoPP 2026，DOI：https://doi.org/10.1145/3774934.3786420

原文与开源仓库确认：
    - 原文状态：已找到本地 PDF 全文，正文包含 Abstract、Background、Challenges and Motivations、Patched Inference with Batching、Exploiting Patch-Level Locality、SLO-Aware Scheduler、Implementation、Experiment 等完整章节。
    - 开源状态：已找到明确开源仓库
    - 仓库链接：https://github.com/desenSunUBW/mixfusion
    - 说明：论文摘要页直接声明代码可用；GitHub 仓库名为 desenSunUBW/MixFusion，README 写明基于 NVIDIA H100 GPU，包含 `distrifuser`、`exp`、`scripts`、`sduss`、`tests` 等目录，并给出 Docker / conda 环境、模型权重、`scripts/paper/run_all.sh` 复现实验脚本和 Apache-2.0 license。本文未在本地运行仓库，因此只确认公开可访问与 README 复现说明，不确认 artifact 可完全复现所有论文结果。

1、论文工作：
    - 论文要解决的核心问题：
      MixFusion 解决的是 text-to-image diffusion model serving 在真实线上场景中遇到的 mixed-resolution 请求并行效率问题。用户请求的图片分辨率常常不同，例如 512x512、768x768、1024x1024 同时到达；diffusion pipeline 又由多步 denoising 组成，且不像 LLM decoding 那样可以靠 KV cache 自然支持变长序列。因此分辨率差异会沿着整条 diffusion pipeline 传播成不同 tensor shape，使普通 batching 很难把这些请求合到一个高效 batch 里。论文给出的例子显示，在 H100 上同时生成 512x512、768x768、1024x1024 三个 SDXL 请求，混合 batch 可以 9.5s 完成，而顺序执行需要 17.8s，这说明并行机会存在，但 baseline 系统不能稳定利用。
    - 瓶颈来源：
      主要瓶颈来自 GPU 利用率不足和调度难度，而不是单个算子本身完全不可优化。不同分辨率导致 batch 形状不一致，使请求只能顺序执行或只能在相同分辨率内合批，GPU parallelism 被浪费。进一步地，patch 化以后又引入三个新瓶颈：第一，patch 之间存在 self-attention 和 convolution 的跨 patch 上下文依赖，朴素切图会损坏图像质量；第二，cache reuse 若按整图或固定 block 策略进行，无法适应不同分辨率下 skipped blocks 的变化；第三，在 SLO serving 中，不同分辨率组合的 latency 变化很大，离线枚举所有组合做 profiling 会出现组合爆炸。
    - 论文的主要贡献：
      论文提出 MixFusion，一个面向混合分辨率 diffusion serving 的 patch-level parallel serving system。它的核心贡献包括：1）把不同分辨率图片切成统一大小的 patch，用 patch 作为细粒度 batch 单元，从而把 resolution diversity 转换成可并行处理的 patch workload；2）提出 Compressed Sparse Patch（CSP）格式和 Patch Edge Stitcher，在保持 patch 并行的同时处理位置元数据、self-attention 重组和 convolution 边界；3）提出 patch-level cache reuse，让每个 block、每个 denoising step 可以按 patch 粒度决定复用或重算；4）提出 SLO-aware scheduler 和轻量 MLP Throughput Analyzer，在混合分辨率 batch 中预测 latency 并选择尽量不违反 SLO 的任务组合。
    - 论文所处背景：
      目标场景是在线 T2I diffusion model serving，模型覆盖 U-Net based diffusion 和 DiT based diffusion，实验中使用 Stable Diffusion XL（SDXL）和 Stable Diffusion 3（SD3）。论文关注的不是离线单请求加速，而是多用户请求流下的 SLO satisfaction、goodput 和 GPU 利用率。它把已有 diffusion 优化分成两类相关 baseline：一类是 patch parallelism，如 Distrifusion，把单个图片切 patch 分给多 GPU 以降低单请求 latency；另一类是 caching，如 block caching，通过跨 denoising step 复用中间结果降低计算量。MixFusion 的出发点是：这些方法对单请求或固定配置有效，但没有把 patch 作为 mixed-resolution batching 的基础调度单位。

2、相对 Baseline 解决的问题与设计方法：
    - Baseline 的具体问题：
      NIRVANA 这类 diffusion serving 系统可以通过 ORCA 式 batching 增大 batch size，但本质仍受分辨率 shape 对齐限制，难以把不同分辨率请求拆成均匀计算单元。Distrifusion 这类 patch parallel system 主要目标是把一张图拆给多 GPU，patch 数通常由 GPU 数量决定；它不支持把不同请求、不同分辨率的 patch 统一合批，而且其 stale cross-GPU context 近似会带来质量损失。Block caching 或其他 diffusion cache 方法通常依赖离线 profile 的固定 skipped block pattern，但论文通过 512、768、1024 三种分辨率的实验说明，不同分辨率下可跳过 block 的分布差异明显，固定 cache 策略无法适应 mixed-resolution serving。调度 baseline 若依靠离线 latency profiling，则需要枚举 batch 内请求数和分辨率组合；当 GPU 支持更多并发请求、分辨率种类增加时，搜索空间快速膨胀。
    - 论文的设计方法：
      MixFusion 的设计可以概括为“patch 化统一计算粒度 + patch-aware data structure + patch-level cache + SLO-aware admission”。系统先把图片沿 height 和 width 切分，patch size 选为当前 batch 中所有分辨率在对应维度上的 greatest common divisor，使不同分辨率请求能被分解成相同形状的 patch。然后使用 CSP 格式记录 patch 到 request、resolution、offset 的关系，使 patch 在 batch 中可以像压缩稀疏结构一样被高效定位和重组。对 diffusion operator，pixel-wise 的 Linear、FeedForward、Cross Attention 可以直接按 patch 批处理；Self-Attention 需要先按分辨率把 patch 重组成完整图片再执行 batched attention；Convolution 需要 Patch Edge Stitcher 处理跨 patch 边界。接着，cache manager 在每个 block 前用 predictor 决定哪些 patch 可复用，未复用 patch 重算，复用 patch 由上一 step cache 填充。最后，scheduler 用 MLP latency predictor 估计候选 batch 的执行时间，在紧急程度和吞吐收益之间选择请求组合。
    - 方法如何对冲 Baseline 缺陷：
      对 mixed-resolution batching 问题，MixFusion 不要求整张图片 shape 一致，而是把图片拆成 shape 统一的 patch，显著增加 batch 内可并行 work item 数。对 patch 边界问题，它没有采用简单 ghost zone 或 stale context，而是在 convolution 前后的 GroupNorm 中融合边界搬运，用最新相邻 patch 数据补齐上下文；论文的 Table 2 显示 Patch Edge Stitcher 相比 replicate/ghost zone 有更高 PSNR 和 SSIM。对 cache 不适配分辨率的问题，它把 cache 粒度从整图或固定 block 降到 patch，让同一 batch 中部分 patch 可以复用、部分 patch 可以重算，避免“只有所有 patch 都满足条件才跳过整块”的保守策略。对组合爆炸调度问题，它不用全量离线 profile，而是训练 MLP Throughput Analyzer，以每种分辨率任务数、正在处理的分辨率数和总 patch 数作为输入预测 batch latency；论文报告预测误差小于 3.7%。
    - 关键 trade-off：
      MixFusion 接受了更复杂的 runtime 数据结构、边界处理、cache 管理和调度逻辑，以换取更高 parallel efficiency 与 SLO satisfaction。patch 越小，可合批性越强，但 patch split、边界 stitch、metadata 和 cache 操作开销越大；patch 越大，开销较小但不同分辨率之间的统一性下降。Patch Edge Stitcher 能减小质量损失，但对 U-Net convolution 仍会产生近似误差；论文 Table 4 中 SD3 因没有 convolution 可以达到 100% pixel-wise accuracy，而 SDXL 的 PSNR/SSIM 随 patch size 增大而提高。patch-level cache 的收益也依赖 cache predictor 和相似度阈值，阈值过大可能影响质量，阈值过小可能减少跳过计算的机会。SLO 调度依赖 MLP latency model，虽然论文实验中误差较低，但模型是在特定硬件和分辨率组合上训练的，仓库 README 也说明 predictor 基于 H100 数据，因此跨硬件迁移需要重新采集数据或重新训练。

3、论文实现：
    - Baseline 如何实现：
      论文将 MixFusion 与 NIRVANA、Distrifusion 和 Mixed-Cache 对比。NIRVANA 被视为 state-of-the-art T2I diffusion serving system，并加入 ORCA 以增强 batch size；Distrifusion 是 distributed parallel inference engine for diffusion，论文只在多 GPU 上评估；Mixed-Cache 是作者构造的变体，保留 patch batching 和 cache，但把 MixFusion 的 SLO-aware scheduler 替换成 FCFS scheduler，用于隔离调度算法收益。所有 scheduling 方法由于显存限制设置最大 batch size 为 12。质量评估中还使用原始模型输出作为参考，比较 CLIP、FID、PSNR 和 SSIM。
    - 新设计如何实现：
      MixFusion 使用 Python 与 C++/CUDA 实现，共约 12.5K 行代码，基于 PyTorch，并遵循 vLLM 的部分系统设计原则。Stable Diffusion 被移植进框架，并拆成 Preparation、Denoising、Postprocessing 三个阶段，以便 baseline 和 MixFusion 共享更灵活的实现。系统集成 xformers 来加速 baseline 和 MixFusion。Patch Edge Stitcher 融合进 GroupNorm kernel：每个 GPU thread block 负责一个 patch 的 normalization，同时检查当前 patch 的边界像素是否被相邻 patch 需要；被需要的边界暂存到 shared memory，normalization 结束后再写回目标 patch 的 global memory，从而把边界搬运和归一化重叠。cache predictor 使用 GPU 上的 cuML Random Forest Classifier，训练数据来自 1K inference requests 中各 block 和 timestep 的 input-output similarity（MSE）；Throughput Analyzer 使用 Scikit-learn 训练 MLP，并放在 CPU 上运行以隐藏调度开销。
    - 实验 / 实现平台：
      单机实验平台为一台配备 NVIDIA H100 80GB GPU 和 AMD EPYC 9534 64-core CPU 的服务器，软件栈包括 Ubuntu 18.04、CUDA 12.3 和 PyTorch 2.2.2。模型使用 SDXL 和 SD3，默认 denoising steps 为 50，默认分辨率为 512x512、768x768、1024x1024，记为 Low、Medium、High。除非特别说明，实验使用 float16 precision。多 GPU scalability 实验扩展到单节点 2、4、8 张 H100；除 Distrifusion 外，其他方法使用 data parallelism 做负载均衡，即新请求到达时分派给 workload 最低的 GPU。
    - 关键实验设置与指标：
      workload 使用 COCO 和 DiffusionDB，各采样 5K text-image pairs 用于质量评估；请求流按 Poisson distribution 生成，三种分辨率默认均匀贡献 workload。SLO 设置遵循 Clockwork convention，为每种分辨率单独执行 latency 的 5x。主要服务指标是 SLO satisfaction 和 goodput；图像质量指标包括 CLIP、FID、PSNR、SSIM。端到端实验显示，相比 NIRVANA，MixFusion 平均提高 30.1% SLO satisfaction，并保持超过 90% SLO；在达到 90% SLO 时，MixFusion 相比 NIRVANA 获得 5.33x goodput，相比 Mixed-Cache 获得 1.06x goodput。质量上，Table 3 显示 MixFusion 的 CLIP 和 FID 与原始模型相近；Table 4 显示 MixFusion 的 patch stitch 质量明显高于 Distrifusion，尤其 SD3 因无 convolution 近似误差而达到 PSNR inf、SSIM 1.0。
    - 消融与参数证据：
      消融部分把额外机制拆成 Patched Batching 和 MixFusion 完整系统。Figure 17 显示 patch-based batching 降低 latency，而 cache management 进一步减少计算；Figure 18 说明 patch size 越大通常 throughput 越高，原因是 splitting overhead 更少，因此默认 patch size 选择 batch 内所有分辨率的最大公约数。Figure 19 在关闭 caching 和 scheduling 后比较 MixFusion 与 Distrifusion，显示 MixFusion 在 8 GPU 上 throughput 随 batch size 增大而提升，而 Distrifusion 在 SD3 上会因通信开销随 batch size 增大而吞吐下降。Figure 20 显示 patch-level cache reuse 相比 whole-image caching 在 SDXL 和 SD3 上都更有效。MLP 参数实验显示，SDXL 使用 (32, 32, 16) 足够，SD3 使用 (64, 64, 16) 足够；Cache Predictor 的随机森林在 SDXL 上可用 50 trees、max depth 5，SD3 上需要 100 trees、max depth 20 获得更高准确率。

4、pipeline/kernel 解析：
    - 新 pipeline/kernel 是什么：
      论文引入的核心不是单个通用 GPU kernel，而是一条 patch-level mixed-resolution serving pipeline。它由四个关键路径组成：1）Scalable Patch Partitioning and Operator Design，把不同分辨率请求转换成统一 patch batch，并用 CSP 维护 patch metadata；2）Patch Edge Stitcher，把 convolution 所需的跨 patch 边界搬运融合到 GroupNorm kernel；3）Patch-Level Cache Reuse，在每个 block、每个 denoising step 前按 patch 粒度做 reuse/recompute 决策；4）SLO-aware scheduler，用 latency prediction 决定哪些请求或 patch batch 应进入当前执行队列。最接近 kernel 级新机制的是 Patch Edge Stitcher 的 fused GroupNorm kernel；最接近 runtime pipeline 的新机制是 patch-based batching + patch-aware caching + SLO scheduling 的组合路径。
    - patch-based inference 的执行流程：
      一个请求到来后，系统先记录 prompt、目标分辨率和 SLO deadline。调度器在 waiting queue 中观察不同分辨率请求，选择一组候选请求进入 batch。对这一组请求，系统计算 height 和 width 上的 patch size，通常取所有分辨率维度的 greatest common divisor。然后把每张 latent image 切成 patch，并按分辨率重排；CSP 记录每个 request 的 RequestOffset、resolution offset、patch index 和邻接 patch 元数据。这样，原本 shape 不同的 512x512、768x768、1024x1024 请求会变成多个 shape 相同的 patch，进入统一 batch 执行。
    - operator 如何流过 patch pipeline：
      对 Linear、FeedForward、Cross Attention 等 pixel-wise operator，patch 可以直接作为普通 batch item 处理，因为每个 pixel/token 的计算不依赖同图其他 patch。对 Self-Attention，单个 query token 需要与同一图片内所有 key/value token 交互，朴素 patch attention 会形成跨 patch Cartesian product，kernel 实现复杂；因此 MixFusion 利用 CSP 按 resolution 将 patch 重组成完整图片，然后执行 batched self-attention。对 Convolution，patch 边界上的像素需要邻居 patch 的上下文；MixFusion 在 split 阶段预先记录每个 patch 的上、下、左、右邻居，缺失邻居用 0 padding，再由 Patch Edge Stitcher 在 GroupNorm 中搬运边界数据，使后续 convolution 能看到正确邻域。
    - Patch Edge Stitcher kernel 路径例子：
      假设一张图被切成 P0、P1、P2、P3 四个 patch，P0 位于左上角。P0 的右边界会被 P1 需要，底边界会被 P2 需要；P3 则需要来自左侧和上侧 patch 的边界。朴素 stitching 会在 convolution 前额外 fetch、concat 边界，导致内存搬运抵消 patch batching 收益。MixFusion 的做法是让处理 P0 的 thread block 在 GroupNorm 时检查边界依赖，把 P0 右边界和底边界暂存到 shared memory，normalization 完成后定位目标 patch，把边界写到 P1、P2 对应位置。这样边界搬运和 normalization 重叠，不需要额外 synchronization，也避免 ghost zone 使用复制边界带来的明显图像割裂。
    - patch-level cache 路径例子：
      在第 t 个 denoising step 的某个 block 前，cache predictor 比较当前 patch 输入与上一 step cache 中的对应 patch，输出一个 mask，标记哪些 patch 可复用、哪些 patch 需要重算。未被 mask 的 patch 进入 block 计算；被 mask 的 patch 使用上一 step 的 cached output 填充，以保持 cross-patch operator 所需的 shape 和上下文一致。block 完成后，系统再次用 mask 将被复用区域替换为 cache 中的值，并更新该 block 的 input/output cache。为了避免每个 patch 单独 query/delete/update 的开销，cache system 将 patch IDs 批量提交，按 Common Set、New Set、Expired Set 处理：Common Set 决定是否替换已有 cache，New Set 插入新 patch，Expired Set 删除已完成请求的 patch cache。
    - SLO-aware scheduling 路径例子：
      假设当前 waiting queue 里有多个 512、768、1024 请求，每个请求有不同 arrival time 和 SLO deadline。调度器先计算每个请求的 slack score，分数越低代表越紧急。随后 Throughput Analyzer 输入当前 batch 中各分辨率任务数、正在处理的分辨率数和总 patch 数，预测加入候选请求后的 latency。调度器在两种倾向之间切换：如果最紧急请求 slack 仍较宽松，就选择能最大化当前 batch throughput 的候选；如果请求接近超时，就优先 admission 该请求以避免 starvation。若某请求即使立即执行也会超时，则根据论文算法将其丢弃；若加入候选后会导致已有 active task 超时，则停止继续扩 batch。最终输出的是一个满足 SLO 约束、同时尽量提高 goodput 的 action queue。
    - 一个具体请求如何流过系统：
      以 4 个请求为例：R1 为 512x512，R2 为 768x768，R3 为 1024x1024，R4 为 768x768。传统 system 可能只能把 R2 和 R4 合批，其余请求单独执行；MixFusion 会把四个 latent 按最大公约数 patch size 切开，得到多个统一 patch。Preparation 阶段生成 prompt embeddings；Denoising 每一步中，pixel-wise operators 直接处理 patch batch，self-attention 按分辨率重组后执行，convolution 经 Patch Edge Stitcher 补边界后执行。每个 block 前 cache predictor 决定哪些 patch 复用上一 step 输出，哪些 patch 重算。Denoising 完成后，CSP offset 将 patch 重组成各自分辨率的 latent，再由 decoder / postprocessing 输出对应图片。这个流程的关键变化是：batching 单位从“整张图请求”变成“可定位、可复用、可重组的 patch”，从而让 mixed-resolution 请求可以共享 GPU 执行窗口。
    - 与传统 pipeline/kernel 的区别：
      传统 diffusion serving pipeline 以完整请求为单位做 batch，只有当分辨率、shape 或 pipeline 参数接近一致时才容易提升 GPU 利用率。Distrifusion 以 patch 加速单张图或少量请求，但 patch 数与 GPU 拆分绑定，且 context 交换采用近似策略，不解决 mixed-resolution batching。MixFusion 则把 patch 作为 runtime scheduling 和 cache management 的基本单位，通过 CSP 保留 request-level 语义，通过 Patch Edge Stitcher 修复 convolution 上下文，通过 patch-level cache 避免整图粒度过粗的问题，通过 SLO-aware scheduler 避免盲目合批导致 deadline violation。它的核心创新不在于替换 diffusion 模型结构，而在于把 serving system 的执行粒度从 request-level 降到 patch-level，并围绕这个粒度补齐算子、缓存和调度机制。
