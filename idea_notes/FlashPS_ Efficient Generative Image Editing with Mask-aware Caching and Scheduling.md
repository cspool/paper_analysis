## FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling

- baseline方法是什么？
  **Baseline**: Diffusers——标准扩散模型serving系统，使用static batching和mask-agnostic全图生成。FISEdit利用mask sparsity做图像编辑加速，但仅适配SD2.1、不支持不同mask ratio请求batching。TeaCache复用denoising中间activation跳过计算，面向通用生成任务，不能利用mask精确区分应保持不变区域。
  **Baseline全栈执行例子**（以Diffusers处理InstructPix2Pix编辑请求为例）：
  - 算法层：整个latent的所有spatial tokens通过DiT transformer全量计算attention和feed-forward，mask区域和unmask区域无差别处理。一个512×512 latent（约4096 tokens for SDXL）需要full attention over all token pairs。
  - 系统层：static batching——请求到达后被组为一组，等整批所有请求完成全部denoising steps后才一起输出。请求数负载均衡（request-count），不考虑不同请求mask ratio差异。
  - 编译框架层：HuggingFace Diffusers pipeline + PyTorch + FlashAttn，编译/框架层无mask-aware支持。
  - kernel层：标准attention kernel（FlashAttn）和FFN kernel，对全量tokens执行计算，无mask-aware sparsity。
  - 硬件层：NVIDIA A10/H800 GPU，标准HBM访存，无cache loading stream。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashPS方法**：(1) Mask-aware cached activation——按mask划分tokens，unmasked区域直接复用缓存的block输出activation Y，仅对masked tokens做主要计算；(2) DP bubble-free block selection——对每个transformer block比较cache loading+masked计算 vs 全量计算的完成时间，O(N) DP决定每个block是否使用cache；(3) Denoising-step continuous batching + disaggregated preprocessing；(4) Mask-aware load balancing。
  **FlashPS全栈执行例子**（同一InstructPix2Pix请求，mask ratio=0.11）：
  - 算法层：输入latent按mask划分为masked tokens（~450 tokens）和unmasked tokens（~3650 tokens）。对每个transformer block，若DP决定使用cache：unmasked tokens的output activation Y从cache直接读取（已在模板预计算时存储），masked tokens在attention中作为query仅与所有tokens的K/V计算，但unmasked tokens的最终Y直接从cache填入。若DP决定跳过cache（因cache loading延迟大于全量计算）：该block对所有tokens执行标准full attention+FFN。block输出合并masked计算结果与cached unmasked activations后送入下一block。
  - 系统层：请求通过mask-aware scheduler路由——scheduler根据mask ratio=0.11用线性模型估算该worker computation latency（低，因计算量与m成正比）和cache loading latency（与unmasked tokens数成正比），选pipeline latency最低的worker。进入worker队列→preprocessing进程将image/mask编码为latent。GPU主进程在下一denoising step边界将请求加入running batch。请求完成50 steps denoising后立即退出batch→postprocessing进程解码为输出图像→返回客户端。新请求在step边界加入而不等待整批完成。
  - 编译框架层：HuggingFace Diffusers + PyTorch，修改attention operator（只对masked tokens执行attention计算，unmasked位置从cache直接注入）。保留FlashAttn作为attention backend。
  - kernel层：CUDA stream-based async cache loading——cache load stream从host memory异步加载cached Y到GPU HBM，computation stream并行对masked tokens执行FlashAttn attention + FFN。kernel级merge操作将masked计算结果与cached unmasked activations拼接回完整Y。
  - 硬件层：NVIDIA A10/H800 GPU。Cache存储在host memory（PCIe加载）或分布式存储。CPU preprocessing/postprocessing进程与GPU denoising主进程并行运行。

  **设计如何解决Baseline缺陷**：
  - 缺陷1（全图重复计算）：mask-aware cached activation将计算量降为与mask ratio m成正比，理论speedup≈1/m。生产trace m=0.11时理论可加速约9×，实际因cache loading开销mask ratio 0.2下SDXL加速2.2×、Flux加速1.9×。缓存Y而非K/V减少cache footprint约2×但对质量影响可控（SSIM 0.99）。
  - 缺陷2（static batching排队延迟）：denoising-step级continuous batching + disaggregated预处理将P95延迟相比static batching降低35%，相比LLM-style naive continuous batching降40%。
  - 缺陷3（忽略mask ratio差异）：mask-aware load balance相比request-count/token-count负载均衡将高流量下tail latency降低26%。
  - 缺陷4（FISEdit的模型支持局限）：FlashPS在SD2.1/SDXL/Flux三类模型上均验证，相比FISEdit最高加速4×。

  **关键trade-off**：FlashPS以host memory/storage容纳GiB级template activation cache换取GPU HBM容量压力降低，但引入PCIe/storage→HBM加载开销，需CUDA stream pipeline重叠隐藏。收益依赖mask区域小、模板复用高、编辑确实保持unmasked region不变；对style transfer等全局改变任务收益下降。系统增加的scheduler/cache engine/异步batch/跨进程通信开销为毫秒级。
