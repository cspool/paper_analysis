论文标题：PiLLM: Resource-Efficient LLM Inference Using Workload Prediction

SS-面向单GPU的token level资源管理机制，更激进分配内存同时减少OOM。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：本地 PDF 只说明 PiLLM 基于 LightLLM 扩展实现，并未给出 PiLLM 官方 GitHub、artifact appendix 或复现实验包。外部核对中，EuroSys 2026 论文列表和作者 Ruihao Gong 主页均能确认该论文条目，但作者主页没有为 PiLLM 标出 Code 链接；只在同页其他论文条目中出现 Code 链接。因此截至本次分析，不能确认 PiLLM 已公开官方实现。相关公开页面：https://2026.eurosys.org/papers.html；https://xhplus.github.io/publication/

    1、论文工作：
        - 论文要解决的核心问题：LLM 在线推理的 GPU 资源浪费来自两层不确定性：跨 GPU 层面，请求数虽然可能有日周期，但真实计算量由输入长度、输出长度和注意力复杂度共同决定，长请求或复杂请求会让基于 GPU utilization 的 autoscaling 反应滞后或长期过量预留；单 GPU 层面，KV cache 会随 decode 增长，现有 batching scheduler 因无法准确估计未来输出长度，只能在保守预留和激进 overcommit 之间取舍，前者浪费显存，后者导致 OOM / eviction。
        - 论文的主要贡献：论文提出 PiLLM，即 Predictable inference for LLMs。核心贡献不是预测单个请求，而是用滑动窗口统计和中心极限定理预测一批请求的输入 / 输出长度分布，再把批级长度估计转成 FLOPs 与 KV cache 内存需求。基于这个 predictor，PiLLM 提出两个互补机制：第一，跨 GPU 的 elastic dispatch / instance manager，根据预测的 prefill 与 decode 计算需求动态决定实例数，并用 spike reaction 处理突发负载；第二，单 GPU 内的 batch-aware KV cache scheduler，用批级剩余输出长度估计提升内存利用率并降低 eviction。
        - 论文所处背景：该工作属于 LLM serving、云端 GPU 资源管理、SLO-aware scheduling 和 KV cache 管理方向，面向 CoT、reasoning framework、AI agent、多轮对话和长文档处理等在线推理负载。论文采用 disaggregated prefill/decode 范式，因为 prefill 偏 compute-bound、decode 偏 memory-bound，二者适合独立估算和独立缩放。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：KServe / 云服务常见 autoscaling 依赖 GPU utilization 等硬件指标，但 LLM 中一次长输入或长输出可能在利用率尚未明显变化时显著拉长 TTFT / TPOT，导致资源扩容滞后。固定最大资源分配能保 SLO，但代价是长期空转。单 GPU batching 方面，vLLM 式贪心策略追求即时显存利用率，可能带来高 eviction；PastFuture 和 SGLang 等更保守策略能压低 eviction，但显存利用率和 batch size 偏低。共同根因是它们缺少轻量、实时、批级的长度和资源需求预测。
        - 论文的设计方法：PiLLM 用滑动窗口维护输入长度和输出长度的均值、方差，并对批级平均长度构造带 error bound 的预测值，例如输出长度预测为 μ_d + σ_d / sqrt(|B|) * Φ^{-1}(1 - ε)。随后用离线校准的系数把长度转换为 prefill FLOPs、decode FLOPs、prefill KV memory 和 decode KV memory。跨 GPU 管理中，PiLLM 根据预测 FLOPs 与目标阶段延迟计算 prefill / decode 所需实例数；请求分发器优先把请求放入 idle instance，否则选择预计最早完成的 active instance，并在无法满足 deadline 时进入 spike reaction。单 GPU 管理中，PiLLM 不再为每个请求预留 worst-case 剩余长度，而是在 batch 级别更新共享 KV cache 预算。
        - 方法如何对冲 Baseline 缺陷：相对 utilization-based autoscaling，PiLLM 直接估算 LLM 真实工作量，而不是等待硬件利用率间接反映负载，因此能更快应对由长请求造成的计算峰值。相对固定最大分配，它用管理窗口内的预测工作量削减不必要实例。相对 per-request KV cache reservation，batch 级预测利用了大数定律：单个请求输出长度难以预测，但多个请求的平均行为方差随 batch size 增大而下降，因此能在保持低 eviction 的同时允许更高显存 overcommit。相对 vLLM 贪心 batching，PiLLM 的 error bound 和批级 memory pool 给激进利用率加了风险控制。
        - 关键 trade-off：PiLLM 用可控的少量 SLO 风险换取 GPU 节省和显存利用率提升。跨 GPU 调度依赖历史窗口统计，当 workload distribution 快速漂移、场景混合比例突变或长尾 outlier 过多时，预测会变差，因此论文加入 spike reaction 兜底。单 GPU 调度依赖 batch 足够大才能显著降低方差；低并发、小 batch 场景中收益会下降。系统还引入集中式 dispatcher / controller，论文在结论中明确指出它在数千 GPU 超大集群中可能成为扩展瓶颈。

    3、论文实现：
        - Baseline 如何实现：跨 GPU baseline 包括两类：基于指标的动态扩缩容，即根据 utilization metrics 调整 GPU count；以及固定最大资源分配，代表生产中为保证 SLO 而保守 overprovision 的方案。单 GPU batching baseline 包括 vLLM 的 greedy resource utilization 策略、PastFuture 的 SLA / 非 eviction 优先策略、SGLang 的 rate-based 策略。论文强调 PiLLM 和 baseline 使用相同底层 GPU kernel 实现，以便主要比较 inter-GPU 和 intra-GPU scheduling 策略，而不是 kernel 差异。
        - 新设计如何实现：PiLLM 基于 LightLLM 扩展实现，选择 LightLLM 的原因是它已有 token-level KV cache 管理，适合做批级 KV cache sharing 和 disaggregated prefill/decode。系统分三层：API layer 收请求并采集输入长度；global scheduling layer 包含 input predictor、inter-GPU manager、request dispatcher 和 output predictor；execution layer 包含 prefill instances 与 decode instances，每个 instance 内部有 intra-GPU manager、batch memory pool、running batch 和等待队列。KV cache 被组织为 token slot linked list，而不是给每个请求固定 block；请求需要新 token 空间时从共享池分配。为提高执行时间可预测性，附录说明 PiLLM 使用 chunk-based workspace、AOT compilation 和 CUDA graph 降低动态内存分配与 JIT 抖动。
        - 实验 / 实现平台：实验平台为 8 张 NVIDIA H800 GPU，NVLink all-to-all 互连，2 颗 Intel Xeon 6448Y CPU，1TB DDR 内存。软件栈包括 PyTorch 2.1、CUDA 12.1，以及 LightLLM 的定制扩展。模型为 LLaMA-3.1 8B。数据集包括 BurstGPT、MoonCake，以及从生产模式派生并去除内容和个人信息的 Conversation、Document、Assistant，覆盖短对话、长文档、多轮上下文、异构长度和长尾输出等负载。
        - 关键实验设置与指标：跨 GPU 主要指标是 GPU Saving Factor 和 SLO Satisfaction Rate；单 GPU 主要指标是 KV cache memory utilization、eviction rate、average batch size。SLO 不是统一固定 deadline，而是先在最大资源 baseline 上测每个请求的参考执行时间，再将 PiLLM 的目标设为该请求 baseline execution time 的 1.2 倍，以适配从几百 token 到十万 token 级输入的巨大差异。总体结果中，PiLLM 在不同 workload 上实现 1.62x 到 3.06x 平均 GPU 节省，prefill SLO 满足率不低于 97.9%，decode SLO 满足率为 100%。单 GPU 层面，PiLLM 的显存利用率约 78.93% 到 96.05%，eviction rate 约 0.01% 到 0.53%；vLLM 显存利用率接近但 eviction 可高达 68.39%，PastFuture / SGLang eviction 低但显存利用率明显更低。

    4、pipeline/kernel解析：
        - 新pipeline/kernel是什么：论文没有提出新的 attention kernel 或 GPU 微内核；最接近的新执行路径是 PiLLM 的 prediction-aware disaggregated prefill/decode serving pipeline。它由控制面的统计采集、批级资源预测、跨 GPU 实例数管理、请求分发与 spike reaction，以及数据面的 LightLLM prefill / decode instances 和批级 KV cache memory pool 组成。关键变化是把“未来 workload”显式建模为批级长度分布，再把该预测同时用于两类资源控制：跨 GPU 的实例数量和单 GPU 的 KV cache overcommit 程度。
        - 新pipeline/kernel的执行流例子：假设一个新时间窗口内到达一批请求。API layer 先拿到每个请求输入长度，并把长度统计写入滑动窗口；global scheduler 用当前窗口的输入长度统计和 decode instance 定期回传的输出长度统计，预测这批请求的平均输入 / 输出长度上界。随后它把预测长度转换为 prefill FLOPs、decode FLOPs 和 KV cache 需求，计算 prefill pool 与 decode pool 各需要多少实例。如果现有 idle instance 足够，请求直接被分发；如果没有 idle instance，dispatcher 选择预计最早完成且仍能满足 request deadline 的 active instance；若没有合适实例，请求进入 spike queue，spike reaction 尝试组成最大可行 batch，并在预测利用率超过阈值时快速激活新 instance。进入 execution layer 后，请求先在 prefill instance 处理输入 token，KV 状态可逐层转移到预先分配的 decode instance；decode instance 继续自回归生成，并把完成请求的输出长度周期性回传给 predictor。单个 GPU 内，batch scheduler 不为每个请求保留最大剩余输出长度，而是把 batch 的预测剩余 token 作为共享内存预算：某些请求生成少于预测值释放余量，另一些请求可使用更多 token slot，只要 batch 总量不越界，就能维持高显存利用率并避免频繁 eviction。
