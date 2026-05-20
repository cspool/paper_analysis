## PiLLM: Resource-Efficient LLM Inference Using Workload Prediction

- 属于Serving调度的实现是什么？实验比较什么？
  提出PiLLM，一个基于workload prediction的LLM serving系统，核心实现包含两层调度：(1) Inter-GPU Elastic Dispatch：global scheduler用滑动窗口统计输入/输出长度分布，通过中心极限定理预测批级长度上界（输出长度预测为 μ_d + σ_d/√|B| * Φ⁻¹(1-ε)），再用离线校准系数将长度转为prefill/decode FLOPs和KV cache内存需求，动态决定prefill/decode实例数；请求分发器优先放入idle instance，否则选预计最早完成的active instance，无法满足deadline时进入spike reaction快速激活新实例；(2) Intra-GPU Batch-Aware KV Cache Scheduler：不为每个请求预留worst-case剩余输出长度，而是在batch级别更新共享KV cache预算，利用大数定律使批级预测方差随batch size增大而下降，在保持低eviction的同时允许更高显存overcommit；KV cache组织为token slot linked list而非固定block。实验比较：(a) Inter-GPU：GPU Saving Factor和SLO Satisfaction Rate，对比utilization-based autoscaling和fixed maximum allocation两类baseline；(b) Intra-GPU：KV cache memory utilization、eviction rate、average batch size，对比vLLM greedy、PastFuture和SGLang。PiLLM在不同workload上实现1.62x-3.06x平均GPU节省，prefill SLO满足率≥97.9%，decode SLO满足率100%；显存利用率78.93%-96.05%，eviction rate 0.01%-0.53%（vLLM eviction可达68.39%）。

- 硬件平台是什么，配置是什么。
  8×NVIDIA H800 GPU（NVLink all-to-all互连），2×Intel Xeon 6448Y CPU，1TB DDR内存。软件栈：PyTorch 2.1、CUDA 12.1。

- 开源Serving框架是什么。修改了什么。
  基于LightLLM扩展实现。选择LightLLM的原因是它已有token-level KV cache管理，适合做批级KV cache sharing和disaggregated prefill/decode。核心修改：(1) API layer添加输入长度采集；(2) 新增global scheduling layer，包含input predictor、inter-GPU manager、request dispatcher和output predictor；(3) Execution layer包含prefill instances和decode instances，每个instance内部有intra-GPU manager、batch memory pool、running batch和waiting queue；(4) KV cache从固定per-request block改为token slot linked list，请求从共享池分配新token空间；(5) 使用chunk-based workspace、AOT compilation和CUDA graph降低动态内存分配与JIT抖动。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：论文未明确说明开源仓库（EuroSys 2026）。作者Ruihao Gong主页（https://xhplus.github.io/publication/）未为PiLLM标出Code链接。以LLaMA-3.1 8B在8×H800上的disaggregated prefill/decode serving为例：
  1. 部署：启动PiLLM的API layer、global scheduler（含predictor + inter-GPU manager + dispatcher）、多个prefill instances和decode instances。LightLLM加载LLaMA-3.1 8B权重，按disaggregated模式配置。
  2. 请求到达：API layer接收请求并采集输入长度→global scheduler将长度统计写入滑动窗口。
  3. 预测阶段：global scheduler用当前窗口的输入长度统计和decode instance定期回传的输出长度统计，预测这批请求的平均输入/输出长度上界。用离线校准系数将长度转换为prefill FLOPs、decode FLOPs和KV cache需求。
  4. Inter-GPU调度：根据预测FLOPs与目标阶段延迟计算prefill/decode所需实例数。若idle instance足够直接分发；否则dispatcher选预计最早完成且满足deadline的active instance；若无合适实例则进入spike reaction组成最大可行batch并快速激活新实例。
  5. Intra-GPU执行：prefill instance处理输入token，KV状态逐层转移到decode instance。decode instance自回归生成，batch scheduler将batch的预测剩余token作为共享内存预算——某些请求生成少于预测值释放余量，另一些请求使用更多token slot，只要batch总量不越界即可维持高显存利用率并避免频繁eviction。
  6. 反馈：decode instance将完成请求的输出长度周期性回传给predictor更新统计窗口。
  7. 效果：在BurstGPT workload上PiLLM实现3.06x GPU节省，prefill SLO满足率99.2%，decode SLO满足率100%。单GPU显存利用率95.84%，eviction rate 0.01%。

