## PiLLM: Resource-Efficient LLM Inference Using Workload Prediction

- baseline方法是什么？
  Baseline分为两层：(1) 跨GPU层面：KServe/云服务常用autoscaling依赖GPU utilization等硬件指标，但LLM中长输入/长输出可能在利用率未明显变化时显著拉长TTFT/TPOT，导致扩容滞后；固定最大资源分配能保SLO但代价是长期GPU空转。(2) 单GPU层面：vLLM的greedy resource utilization策略追求即时显存利用率但可能带来高eviction（可达68.39%）；PastFuture和SGLang等更保守策略能压低eviction但显存利用率和batch size偏低。

  Baseline全栈执行例子（以vLLM greedy + utilization-based autoscaling为例）：
  - 算法层：LLaMA-3.1 8B，标准prefill+decode pipeline，KV cache按paged attention分块为per-request固定block。
  - 系统框架/Serving层：vLLM continuous batching，greedy per-request KV cache reservation（每个请求预留max output length的KV空间），跨GPU通过KServe/Kubernetes HPA根据GPU utilization触发autoscaling。
  - 编译框架层：论文未明确说明。
  - kernel调度层：LightLLM token-level KV cache管理，标准attention kernel。
  - 硬件架构层：8×NVIDIA H800 GPU（NVLink），无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出PiLLM，一个predictable inference系统。核心方法分两部分：(1) 用滑动窗口统计+中心极限定理预测批级输入/输出长度分布，再转为FLOPs和KV cache内存需求；(2) 基于该predictor，实现跨GPU的elastic dispatch和单GPU的batch-aware KV cache scheduling。

  **缺陷1：Utilization-based autoscaling依赖硬件指标间接反映负载，长请求导致扩容滞后**
  → PiLLM直接估算LLM真实工作量（prefill/decode FLOPs），而非等待GPU utilization间接反映负载。滑动窗口维护输入/输出长度均值方差，对批级平均长度构造带error bound的预测值，用离线校准系数转FLOPs后计算所需prefill/decode实例数。能更快应对长请求造成的计算峰值。

  **缺陷2：固定最大分配过度预留GPU，长期空转浪费**
  → PiLLM用管理窗口内的预测工作量削减不必要实例。spike reaction兜底：当无法满足deadline时尝试组成最大可行batch并快速激活新实例。不同workload上实现1.62x-3.06x GPU节省，prefill SLO满足率≥97.9%。

  **缺陷3：Per-request KV cache reservation（vLLM greedy）显存利用率高但eviction可高达68.39%；保守策略（PastFuture/SGLang）eviction低但显存利用率低**
  → PiLLM的batch-aware KV cache scheduler利用大数定律：单个请求输出长度难以预测，但多个请求的平均行为方差随batch size增大而下降。不对每个请求预留worst-case剩余长度，而是在batch级别更新共享KV cache预算，以可控error bound允许更高显存overcommit。结果显存利用率78.93%-96.05%，eviction rate仅0.01%-0.53%。

  **缺陷4：固定per-request KV cache block分配不灵活，无法池化共享**
  → PiLLM将KV cache组织为token slot linked list而非固定block。请求需要新token空间时从共享池分配，释放后的slot回池。配合批级memory pool，某些请求生成少于预测值释放余量供其他请求使用。

  论文方法全栈执行例子（以PiLLM disaggregated prefill/decode + LLaMA-3.1 8B为例）：
  - 算法层：LLaMA-3.1 8B，标准prefill+decode pipeline。新增控制面的统计采集、批级资源预测、跨GPU实例数管理和批级KV cache memory pool。预测公式：输出长度上界 μ_d + σ_d/√|B| * Φ⁻¹(1-ε)，离线校准系数转FLOPs/memory。
  - 系统框架/Serving层：基于LightLLM扩展，三层架构——API layer（输入长度采集）、global scheduling layer（predictor + inter-GPU manager + dispatcher）、execution layer（prefill/decode instances + intra-GPU manager + batch memory pool）。Disaggregated prefill/decode，prefill compute-bound独立缩放，decode memory-bound独立缩放。
  - 编译框架层：chunk-based workspace、AOT compilation和CUDA graph降低动态内存分配与JIT抖动。
  - kernel调度层：LightLLM token-level KV cache（token slot linked list），无定制kernel。
  - 硬件架构层：8×NVIDIA H800 GPU（NVLink），无定制硬件。
