## Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

- 属于Serving调度的实现是什么？实验比较什么？
  提出Difflow（原名ChituDiffusion），一个利用扩散模型请求中数据属性局部性(data property locality)的serving系统。核心Serving/调度实现包含：(1) dGraph Identification：编译时通过符号化数据属性传播(symbolic data property propagation)将扩散pipeline分解为dGraphs，每个dGraph内操作共享相同优化条件；(2) dTask Scheduling：运行时将异构请求按dGraph分解为细粒度dTasks，通过动态规划(DP, Algorithm 1)识别并批处理具有相同数据属性的dTasks——schedule对所有dEngines枚举property requirements，为每种requirement找到最大batch (GetLargestBatch)，递归组合剩余dTasks为batches，用基于线性回归的性能模型(OLS regression, R²=0.998)估计执行时间，选择最优batching plan；(3) Data-Aware Batching：针对不同raggedness ratio混合使用regular dEngines和ragged dEngines（第7.7节：raggedness 0%时uniform batching更优，100%时ragged batching达1.5×, 25%-50%混合batching额外提升10%）；(4) Asynchronous Property Inference：通过tensor fingerprint技术在无实际执行情况下推断dTask输出数据属性，使调度与执行重叠以隐藏scheduling overhead；(5) 调度窗口分析：调度开销低于dEngine runtime的10%，大窗口下GPU idle time <5%含cold start。实验比较throughput（req/s），对比PyTorch v2.1、PyTorch-Inductor v2.1、TensorRT v8.6、Stable Fast v1.0、Katz (per GPU)，覆盖5个UNet-based应用(refine/edit/video/venti/grande)和3个DiT-based应用(refine-mix/refine-dit/edit-dit)。

- 硬件平台是什么，配置是什么。
  两台服务器：(1) NVIDIA A100 40GB PCIe GPU；(2) NVIDIA H100 80GB PCIe GPU。UNet结构模型使用CUDA 12.1，DiT系列模型使用CUDA 12.8。开源release使用PyTorch 2.9。

- 开源Serving框架是什么。修改了什么。
  基于Diffusers[61]、Triton[59]、Stable Fast[1]、FlashAttention[21,22]组件实现(C++和Python)。核心修改/新增：(1) dGraph分解pipeline：符号化数据属性传播规则(Table 1, 覆盖elementwise/linear/convolution算子)将pipeline分片为共享优化条件的dGraphs；(2) dTask调度与DP batching：Algorithm 1的动态规划调度，融合dTasks with identical inputs，枚举dEngine batching plans；(3) 异步property inference：tensor fingerprint技术(Φ operat-commutative hash)推断dTask输出属性；(4) 性能模型：OLS regression on input-related metrics预测执行时间(R²=0.998)；(5) 四个ragged data-independent operation kernels (Triton+CUDA)支持ragged batching；(6) 支持UNet和DiT架构，通过DFG作为universal IR实现architecture-agnostic。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/thu-pacman/chitu/tree/Diffusion。使用流程：
  1. 部署：加载diffusion pipeline定义（如SDXL+Refiner或SD15+ControlNet）→ChituDiffusion编译器执行符号化属性分析→将pipeline分解为dGraphs→每个dGraph编译为多个dEngines（针对不同数据属性配置）
  2. Application characteristics配置：开发者提供pipeline输入的prior knowledge——哪些输入cross-request相同(fixed property)、哪些可能相同(symbolic variable)、哪些经常变化
  3. 请求到达：Scheduler aggregate scheduling window内请求→按dGraph分解为dTasks→Async property inference计算tensor fingerprints→DP scheduler枚举dEngines寻找最优batching plan→dTasks batches dispatch到matched dEngines执行
  4. 以edit应用(SDXL Turbo + ControlNet + 16 LoRA styles)为例：pipeline被分解为3个dGraphs——dGraph1(输入依赖的U-Net层)、dGraph2(输入独立的U-Net层, 16 styles共享)、dGraph3(后处理)。dGraph2的input-independent计算被识别为loop-invariant→multi-value compile-time caching。运行时仅有2个unique dTasks for dGraph1和3个for dGraph2，通过data-aware batching exploit shared inputs实现1.71× speedup。
  5. 在A100上refine应用达2.1× speedup, H100上达2.2×；DiT refine-mix上相对Stable Fast达3.0×（因Stable Fast无法compile HunyuanImage refiner pipeline）。

Difflow的作用：通过利用扩散模型请求中数据属性的局部性（相同prompt的冗余计算、相同shape的批处理、loop-invariant tensors），将pipeline分解→编译优化→运行时动态调度三个层次协同，在不牺牲等价性的前提下最大化throughput。相比仅支持uniform batching的现有框架，Difflow通过dGraph重组+多版本dEngine编译+DP调度实现对异构数据属性请求的高效批处理。

