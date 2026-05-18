## Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

- baseline方法是什么？
  Baseline方法分为两类：(1) **Chunked-prefill**（SGLang/SARATHI-Serve）：将prefill拆分为chunks，每个chunk与一次decode迭代融合执行，通过capping token budget（prefill chunk新token数 + decode batch大小）来保证decode SLO。执行例子：请求到达→prefill被拆成多个chunk→每个chunk与decode iteration一同发射到GPU→所有SMs同时执行chunk prefill+decode→chunk prefill需读取之前所有chunk产生的KV cache→decode迭代在每次融合执行后产生新token。全栈路径：算法层无特殊优化→Serving框架在SGLang中通过Flashinfer将prefill attention和decode attention融合为单一kernel→GPU所有SMs统一执行融合kernel→KV cache以PagedAttention方式管理。(2) **Disaggregated serving**（Splitwise/SGLang-PD静态、LoongServe动态）：将prefill和decode分配到不同GPU实例，各自有独立KV cache pool。执行例子：请求到达→prefill实例处理→KV cache迁移或recompute→decode实例迭代产生token。Splitwise GPU分配在初始化时固定；LoongServe根据序列长度动态伸缩GPU数量，但释放原始GPU上的KV cache，跨请求无法复用。

  Baseline的缺陷：
  - Chunked-prefill：过度小的token budget无法打满GPU导致利用率低，过度大的token budget导致TBT超过SLO。而且chunk prefill需重复读取KV cache，当reused context长时（如multi-turn场景>4K tokens），TBT显著膨胀甚至SLO violation。
  - Static disaggregation：GPU静态分配无法适应请求负载波动，decode/prefill实例一方繁忙时另一方空闲。分离的KV cache pool使有效cache容量减半，cache hit rate从36.6%降至4.2%。
  - Dynamic disaggregation (LoongServe)：为支持动态GPU伸缩而立即释放KV cache，跨请求无法复用，multi-turn场景需全量recompute KV cache。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**intra-GPU prefill-decode (PD) multiplexing**：在同一GPU内不同SM上空间复用prefill和decode。核心设计：(1) **空间分区替代时间融合**：decode阶段保留best-fit SMs（如60%）以满足TBT SLO，剩余SMs分配给prefill，避免了chunked-prefill的token budget dilemma。因为prefill和decode独立执行，prefill不再阻塞decode，decode SLO不再受prefill chunk大小影响。(2) **共享内存空间**：两个阶段在同一进程中共享GPU内存和KV cache pool，避免了disaggregation造成的cache pool缩减和hit rate下降。(3) **动态SM重配**：通过GreenContext以微秒级开销调整SM分区比例，适应请求负载和输入长度的动态变化。(4) **Layer-wise prefill执行**：将prefill按transformer layer切分为多个prefill layer，每层独立发射，消除了因prefill和decode延时差异导致的GPU气泡，同时支持长请求抢占短请求以保障TTFT SLO。(5) **Contention-tolerant worst-case估计**：通过solo-run predictor + contention guard提供decode的最坏延时估计，保守但安全地保障SLO。

  论文方法全栈执行例子：
  - 算法层：与baseline相同，标准的transformer attention+FFN，论文未修改模型算法。
  - Serving框架层：MuxWise dispatcher收到请求→estimator用solo-run predictor（公式1/2）预测prefill和decode延时→contention guard查表得到最大slowdown factor→dispatcher选择multiplexing plan（如decode 60% SMs，prefill 40% SMs）→multiplex engine将prefill拆为prefill layers（PLs），计算发射层数NPL=⌈(Td×NT)/TP⌉→发射decode graph到decode stream→异步发射NPL个prefill layers到prefill stream→query-based同步定期轮询CUDA events→prefill完成后合并入decode batch。
  - 编译框架层：论文未修改编译框架。但利用了CUDA Graph优化decode iteration（单graph launch<0.5ms），prefill使用piecewise CUDA graph（按layer切分）。
  - kernel调度层：通过GreenContext将CUDA streams绑定到特定SMs实现intra-process空间分区。SMs按16个为粒度分组（6种配置A100，7种H100），contention guard通过grid-sampling profiling（约7K样本对，12小时/模型-机器对）获得memory bandwidth竞争的最大slowdown（A100≤20%，H100≤30%）。Flashinfer提供融合的prefill+decode attention kernel。
  - 硬件架构层：使用NVIDIA A100/H100/H200 GPU，NVLink互联，论文未修改硬件。利用H100的Thread Block Cluster特性（要求16 SM粒度）。
