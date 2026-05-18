## The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

- baseline方法是什么？
  Baseline是conventional single-turn LLM serving（以ShareGPT chatbot workload为代表），即每个用户请求对应一次LLM inference，prefill+decode后返回结果。这是目前LLM serving系统（如vLLM continuous batching）优化的主要目标workload。

  全栈执行例子（以ShareGPT + Llama-3.1-8B-Instruct + vLLM on A100为例）：
  - 算法层：单次LLM inference，无需外部tool interaction、无迭代reasoning loop。模型接收prompt→prefill一次→decode生成response tokens→返回。
  - 系统框架/Serving层：请求到达vLLM→FCFS scheduler→continuous batching将多个请求的decode step合并→prefix caching可选但收益有限（因为单次inference prefix共享少）。ShareGPT latency集中分布在低位（95th percentile 9.7s），throughput可达6.4 QPS。GPU时间分配：prefill约占4.7%、decode约占74.1%、GPU无idle时段（无外部tool等待）。
  - 编译框架层：论文未明确说明（使用PyTorch 2.6 + CUDA 12.8默认编译路径）。
  - kernel调度层：论文未明确说明（使用vLLM默认CUDA kernel，无定制kernel）。
  - 硬件架构层：NVIDIA A100 40GB GPU，无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文不是提出新系统优化，而是构建系统级表征方法论（AgentBench框架），通过量化agent workload vs static LLM serving的成本差距，揭示baseline中"被忽略"的系统瓶颈，为未来scheduler、cache、routing设计提供目标。

  **缺陷1：Static LLM serving假设请求是一次或少次模型前向，无法反映agent的多轮迭代LLM call + tool call模式**
  → 论文测量：tool-augmented agent平均LLM calls是CoT的9.2×，LATS平均71.0次LLM calls/request。单请求内LLM与tool因数据依赖难以并行（LLMCompiler的DAG规划仅实现18.2% overlap）。这导致单请求延迟分布宽且带重尾（ReAct HotpotQA 95th percentile 20.7s vs ShareGPT 9.7s）。

  **缺陷2：Baseline serving optimization假设GPU持续执行LLM inference，不处理tool等待造成的GPU idle**
  → 论文测量：GPU runtime中tool latency可导致最多54.5% idle time（HotpotQA/MATH因Wikipedia API和Wolfram Alpha API在CPU/外部执行）。论文论证需要inter-request parallelism来填补这些idle gap：ReAct sequential execution仅0.10 QPS，concurrent execution提升到2.6 QPS（25×）。但即便如此，agent serving throughput仍远低于ShareGPT（2.6/1.2 QPS vs 6.4 QPS）。

  **缺陷3：Baseline KV cache优化针对短context、单次inference的静态LLM，不处理agent长interaction history导致的KV cache膨胀**
  → 论文测量：tool-augmented agent的KV cache memory/request平均是CoT的3.0×、最坏5.4×。但prefix caching在agentic workload中收益显著：prefill latency降低60.1%，end-to-end LLM latency降低15.7%，LATS memory requirement降低64.8%，serving throughput提升5.62×（vs ShareGPT仅1.03×）。prefix caching在serving场景下使KV cache平均/最大memory usage分别降低51.7%/63.5%。

  **缺陷4：Baseline不考虑test-time scaling的成本-收益递减，将模型推理视为固定成本**
  → 论文测量：test-time scaling存在sharply diminishing returns。Reflexion从16.9s→25.6s仅获4% accuracy gain，从56.0s→325.5s仅获同等marginal gain（31× cost）。sequential scaling峰值资源需求低但延迟长；parallel scaling可降低延迟但增加瞬时GPU memory和serving contention。8B模型配合LATS parallel scaling可接近70B性能但energy更低。8B ShareGPT 0.32 Wh/query vs 8B Reflexion 41.53 Wh (130.9×) vs 70B Reflexion 348.41 Wh (136.5×)。

  **缺陷5：Baseline无datacenter-level energy/power意识**
  → 论文projection：71.4M queries/day下70B Reflexion接近1.0 GW datacenter power；Google Search级13.7B queries/day下70B Reflexion达198.9 GW（接近美国电网平均负荷的40%）。

  论文方法全栈执行例子（以HotpotQA + ReAct agent + Llama-3.1-8B-Instruct + vLLM on A100为例）：
  - 算法层：ReAct agent workflow：LLM产生thought/action（如Wikipedia search）→tool执行→observation回写context→下一轮LLM。单请求平均多次LLM call（agent calls平均9.2× CoT），token组成包含instruction + few-shot + user query + LLM history + Tool history + output。
  - 系统框架/Serving层：Agent server entrypoint→ReAct worker→vLLM backend。prefix caching在multi-round LLM calls间复用shared prefix KV cache，prefill latency降低60.1%。多worker并发通过continuous batching填补单worker的tool-idle GPU时间。GPU execution breakdown：prefill 4.7%、decode 74.1%、tool-idle最高54.5%。
  - 编译框架层：论文未明确说明（PyTorch 2.6 + CUDA 12.8默认编译）。
  - kernel调度层：论文未明确说明（使用vLLM默认kernel）。
  - 硬件架构层：GCP A100 40GB GPU（8B单卡、70B 8卡），GPU utilization用DCGM测量。论文强调control-flow serialization、long-context KV cache pressure和idle-period underutilization是dynamic reasoning workload的特征，不是特定GPU microarchitecture独有。
