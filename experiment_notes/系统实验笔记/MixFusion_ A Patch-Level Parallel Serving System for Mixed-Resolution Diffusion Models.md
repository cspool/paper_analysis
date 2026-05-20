## MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

- 属于Serving调度的实现是什么？实验比较什么？
  提出MixFusion，一个面向混合分辨率扩散模型serving的patch级并行serving系统。核心实现包含四个模块：(1) Compressed Sparse Patch (CSP) 格式：受CSR格式启发，将不同分辨率请求的图像partition为均匀patch，通过resolution reordering + offset-based compression实现高效patch定位；(2) Patch-Tailored Diffusion Operators：Self-Attention模块按resolution分组后reconstruct为全图再batch执行；Convolution通过记录每个patch的邻接patch依赖实现跨patch边界stitching；(3) Patch-Level Cache Manager：在每个block每步执行前通过Random Forest Cache Predictor（GPU端cuML）动态预测各patch是否需要重算，将cache query/delete/update/insert操作合并为batch执行（Common Set/New Set/Expired Set三集合分类）；(4) SLO-Aware Scheduler：以slack score（(DDL-C-P)/SA）量化请求紧急度→最紧急或最大吞吐提升的请求被加入active batch→MLP Throughput Analyzer（CPU端Scikit-learn, 3层MLP）预测batch latency→schedulability test阻止SLO violation。实验比较SLO Satisfaction、Goodput对比NIRVANA+ORCA（state-of-the-art diffusion serving）、Distrifusion（分布式patch并行）和Mixed-Cache（MixFusion变体用FCFS调度），以及CLIP/FID quality；ablation包括scalability（2/4/8 GPU）、workflow distribution（单分辨率主导）、SLO scale（3×/5×/10× baseline latency）。

- 硬件平台是什么，配置是什么。
  单台服务器：NVIDIA H100-80GB GPU + AMD EPYC 9534 64-core CPU。软件栈：Ubuntu 18.04, CUDA 12.3, PyTorch 2.2.2。多GPU scalability评估使用单节点内2/4/8 H100 GPU（数据并行，新请求dispatch到workload最低GPU）。

- 开源Serving框架是什么。修改了什么。
  基于PyTorch实现，遵循vLLM系统设计原则，总计12.5K行Python + C++/CUDA代码。核心修改/新增：(1) Stable Diffusion pipeline重构为Preparation/Denoising/Postprocessing三阶段，支持跨可变步数的batch denoising；(2) 集成xformers加速attention；(3) CSP格式的patch管理（resolution reorder + RequestOffset/ResolutionOffset/RequestStart/RequestEnd arrays）；(4) Patch Edge Stitcher fused入GroupNorm kernel（Section 4.3）；(5) Patch-level caching系统（Random Forest Cache Predictor on GPU via cuML, batched cache operations）；(6) MLP Throughput Analyzer（CPU端Scikit-learn）用于在线latency prediction；(7) SLO-aware scheduling algorithm（Algorithm 1）。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/desenSunUBW/mixfusion。以SDXL serving为例使用流程：
  1. 部署：加载SDXL模型→MixFusion将pipeline分解为Preparation/Denoising/Postprocessing三阶段→Denoising中每个block前注入Cache Predictor→patch管理使用CSP格式
  2. 请求到达：多个不同分辨率请求（如512×512, 768×768, 1024×1024各一个）→Scheduler的MLP Throughput Analyzer预测合并后的batch latency→计算slack score→决定是否将新请求加入active batch
  3. Patch处理：patch size取各resolution在height/width维度的最大公约数→CSP格式通过RequestOffset记录每个请求的首patch偏移，通过RequestStart/RequestEnd记录每个patch所属请求
  4. Denoising执行：pixel-wise operators（Linear, FeedForward, Cross Attention）直接对各patch独立执行→Self-Attention时按resolution分组reconstruct全图后batched attention→Convolution时Patch Edge Stitcher fused in GroupNorm kernel处理跨patch边界
  5. Cache预测：每个block前Random Forest Cache Predictor比较当前输入与上步cache→生成mask标记reusable patches→block仅重算unmasked patches→masked patches用上步cache填充→batch cache query/update
  6. 完成所有denoising步后VAE decoder生成最终图像
  7. 在H100上以90% SLO satisfaction为目标：SDXL上MixFusion相比NIRVANA达到5.33× higher goodput，SD3上相比NIRVANA SLO satisfaction平均高30.1%

