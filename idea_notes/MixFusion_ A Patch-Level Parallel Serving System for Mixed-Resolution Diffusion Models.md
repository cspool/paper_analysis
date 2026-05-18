## MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

- baseline方法是什么？
  **Baseline为NIRVANA + ORCA batching和Distrifusion。**
  全栈执行例子（以SDXL serving三个不同分辨率请求512×512/768×768/1024×1024为例）：
  - **算法pipeline**：NIRVANA使用approximate caching（offline profiling决定每步skipped blocks），对全图做cache reuse决策——block级skip基于固定pattern（每步预定义skip哪些block，不随分辨率变化调整）。Distrifusion将image切为固定数量patch分发到多GPU，各GPU独立执行denoising，异步AllGather交换stale cross-GPU context。
  - **Serving系统框架**：NIRVANA集成ORCA进行iteration-level selective batching——ORCA优先新到达请求以在strict deadline内完成更多请求，但当请求分辨率不同时tensor shape mismatch→无法batching→sequential execute→GPU underutilization（三个SDXL请求sequential 17.8s, concurrent batching 9.5s on H100）。Distrifusion在multi-GPU下每个GPU独立处理分配到的patch→通过异步通信overlap同步开销→但patch数固定=GPU数，无法将不同分辨率请求统一为同一patch size→不支持mixed-resolution batching。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：NIRVANA使用标准PyTorch diffusion pipeline kernel（无patch-level kernel）。Distrifusion使用异步AllGather进行cross-GPU boundary exchange，但stale context导致accuracy degradation（Figure 6, Distrifusion生成图像内容偏离original）。Naive stitching（fetch all required boundaries + concatenate with target patches）的overhead完全抵消patch parallelism收益（Figure 5, naive stitch latency > sequential）。
  - **硬件架构**：论文未明确说明。NIRVANA和Distrifusion均运行于NVIDIA H100 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **MixFusion通过CSP patch管理 + Patch Edge Stitcher + patch-level caching + SLO-aware scheduling四个层次协同解决baseline在混合分辨率场景下的缺陷。**
  全栈执行例子（同场景三个不同分辨率SDXL请求）：
  - **算法pipeline**：MixFusion将不同分辨率请求partition为uniform patch（patch size = 各resolution height/width维度的最大公约数）→pixel-wise operators（Linear, FeedForward, Cross Attention）对各patch独立→Self-Attention按resolution分组reconstruct全图后batched attention→Convolution通过Patch Edge Stitcher fuse入GroupNorm kernel处理跨patch依赖。Cache predictor（GPU端Random Forest per block）在每步每block前动态预测每个patch的reusability——非固定skip pattern，能适应resolution变化导致的skip block差异（Figure 7, 不同resolution的skipped blocks分布显著不同）。
  - **Serving系统框架**：CSP格式通过resolution reorder + offset compression将heterogeneous patches映射为统一批次→所有请求的patches在单batch内并行处理（3请求从17.8s sequential→9.5s batched）。SLO-Aware Scheduler用slack score量化紧急度，MLP Throughput Analyzer（200个resolution组合训练，error <3.7%）预测batch latency→schedulability test避免SLO violation→相比NIRVANA在90% SLO satisfaction下SDXL goodput提升5.33×，SD3 SLO satisfaction平均高30.1%。
  - **编译框架**：论文未明确说明。基于PyTorch + xformers。
  - **kernel调度**：Patch Edge Stitcher fused in GroupNorm kernel——每个TB normalized一个patch的同时将boundary pixels暂存shared memory→所有normalization完成后TB定位目标patch写回boundary→消除naive stitching的额外memory movement overhead。CSP O(1) patch定位→batch cache操作（Common/New/Expired三集合coalesce）→每个block的cache操作<2ms（SD3每step 40-50ms含24 blocks→per-block cache overhead ~1-2ms）。
  - **硬件架构**：论文未明确说明。运行于NVIDIA H100-80GB GPU。

  **对应解决Baseline缺陷的映射：**
  **(1) 混合分辨率无法batching → CSP + uniform patch partitioning**：NIRVANA因tensor shape mismatch只能sequential处理不同分辨率请求。MixFusion将image沿height/width均切为uniform patch→所有patch shape相同→单batch并行→GPU utilization显著提升（Figure 19, throughput随batch size线性增长，3→12 batch size）。
  **(2) 跨patch context exchange开销大 → Patch Edge Stitcher fused in GroupNorm**：Naive stitching的boundary fetch+concat开销完全抵消并行收益（Figure 5）。Patch Edge Stitcher将stitching fuse入GroupNorm→TB shared memory暂存boundary→overlap stitching与normalization→无额外synchronization开销→PSNR 28.82/SSIM 0.88 (4 patches, Table 4)远高于naive replicate (PSNR 9.54/SSIM 0.45)。
  **(3) 固定cache pattern不适应resolution变化 → patch-level dynamic cache reuse**：NIRVANA的offline fixed skip blocks在混合分辨率下失效（Figure 7不同resolution skipped blocks分布不同）。MixFusion的per-patch per-block Random Forest Cache Predictor在线比较input与cached data→动态决定每个patch是否重算→patch-level caching比whole-image caching consistently更高computation savings（Figure 20）。
  **(4) 混合分辨率latency难以预测 → MLP Throughput Analyzer**：传统offline profiling无法应对explosive combination of resolutions（M并发请求×N分辨率→∑C(N-1, i+N-1)个组合）。MixFusion用200个resolution组合训练MLP→在线预测batch latency（error <3.7%）→SLO-Aware Scheduler做schedulability test→避免SLO violation同时最大化goodput。
  **(5) Distrifusion accuracy degradation → up-to-date cross-patch context**：Distrifusion使用async AllGather的stale KV cache导致内容偏离original（Figure 6）。MixFusion在单GPU内通过CSP同步所有patch→Convolution通过Patch Edge Stitcher使用real-time boundary data→PSNR/SSIM显著高于Distrifusion（Table 4, SDXL PSNR 28.82 vs 10.96, SSIM 0.88 vs 0.49）。

  Trade-off：(1) Patch Edge Stitcher only covers GroupNorm+Conv fusion——对DiT（无Conv, SD3）模型无需stitching但受益于batching；(2) Cache Predictor (Random Forest on GPU)增加per-block prediction overhead——需控制在<2ms per block，否则cache收益被抵消；(3) CSP格式假设请求在GPU内完成整个denoising pipeline——不支持跨请求preemption；(4) SLO scale为3×时NIRVANA因ORCA的priority scheduling在SD3上略优于Mixed-Cache——此时SLO constraint极紧限制了batching收益空间。
