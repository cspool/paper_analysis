## AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

- baseline方法是什么？
  Baseline是现有LLM serving系统的continuous batching策略（vLLM、TensorRT-LLM、Sarathi-Serve），将同一batch中所有请求同质化处理，iteration-level scheduling导致batch内各请求per-token latency近似一致。
==continuous batch、iteration-level==
```
	## step结尾补充batch
	for each decoding_iteration:
	    batch = running_requests
	    if waiting_requests can be admitted:
	        prefill(new_requests)
	        batch.add(new_requests)
	
	    run_one_decode_step(batch)
	
	    remove_finished_requests(batch)
	
	## iteration = step
	for iteration t:
	    batch = select_requests()
	    for layer in 1..N:
	        run_layer(layer, batch)
	    for request in batch:
	        emit_one_token(request)
```
- 当strict SLO请求（如coding copilot TPOT < 50ms）与relaxed SLO请求（如summarization TPOT < 150ms）混部时，要么缩小batch牺牲吞吐满足严格请求，要么保持大batch导致latency violation。
==不同SLO请求放到1个batch。==
- vLLM + Priority等优先级策略通过限制batch或抢占非紧急请求照顾严格请求，但损害整体SLO attainment。
==抢占机制补丁==
- vLLM-Spec和SpecInfer虽使用speculative decoding，但策略静态（固定speculation length/width），缺少per-request SLO aware allocation，不能随请求分布和系统负载动态调节。
==不同SLO相同batch，使用相同draft length，不适合多SLO的动态负载。==

  全栈执行例子（以vLLM + Llama3.1-70B + coding copilot(50ms SLO) + chatbot(100ms SLO) + summarization(150ms SLO) 混部为例）：
  - 算法层：所有请求统一使用自回归decode（每轮出1 token）或固定speculation length speculative decoding（每轮尝试出n个token）。同一batch中三个请求sync barrier等待后一起进入下一decoding step。
  - 系统框架/Serving层：vLLM continuous batching + FCFS scheduler + PagedAttention。batch composition由到达顺序决定，非SLO-aware。coding copilot因strict SLO可能在batch中因其他请求的decode时间积累而violate TPOT。
  - 编译框架层：论文未明确说明（使用PyTorch + CUDA默认编译路径）。
  - kernel调度层：vLLM默认CUDA kernel（FlashInfer/PagedAttention），无per-request token-level调度。
  - 硬件架构层：NVIDIA A100 80GB GPU，无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文将multi-SLO serving形式化为带budget约束的token tree构造问题，提出SLO-customized speculative decoding的speculate-select-verify三阶段pipeline。

  **缺陷1：Baseline continuous batching将batch内请求同质化，所有请求在一次decoding iteration中同步前进相同步数，无法为strict SLO请求提供更快的decoding速度**
  → AdaServe允许同一batch中不同请求在同一次LLM verification中接受不同数量的draft token：strict SLO请求在SLO-customized selection中优先获得足够高概率节点（保证一次verification可能接受多个token），relaxed SLO请求仅获少量节点（不浪费验证budget）。打破continuous batching的"所有请求同速前进"限制。

  **缺陷2：Baseline speculative decoding策略静态（固定speculation length/width），不随per-request SLO需求和系统负载动态调节**
  → AdaServe根据每个请求的当前latency、已生成token数、TPOT SLO阈值和全局活跃请求数动态计算per-iteration SLO推进目标。同时根据活跃请求数动态调节speculation depth d和beam width w：高负载时减小d/w降低speculation overhead，低负载时增大d/w提升吞吐。

  **缺陷3：Baseline优先级策略通过限制batch或抢占照顾strict请求，损害整体SLO attainment和goodput**
  → AdaServe的SLO-customized + throughput-optimized两阶段token allocation：先用budget满足所有请求SLO需求（SLO优先），剩余budget再按全局path probability最大化总吞吐。严格请求不通过牺牲其他请求满足，而是通过speculation depth/width和token tree节点分配的精细粒度实现。

  **缺陷4：Draft model和target LLM colocate时speculation overhead高，尤其在动态调节depth/width时kernel launch overhead显著**
  → AdaServe用CUDA Graph优化draft model decoding：从第二个speculation step到第d步，若活跃请求数相同则复用预捕获CUDA graph，消除重复kernel launch overhead。实验显示CPU selection overhead仅占总serving time的0.41%/0.31%。

  论文方法全栈执行例子（以AdaServe + Llama3.1-70B + coding copilot/chatbot/summarization 混部为例）：
  - 算法层：每轮decoding iteration：
	  - (1) draft model beam search生成candidate token tree（每请求d步×w beam），记录draft logits近似path probability；
	  - (2) SLO-customized selection：coding copilot（TPOT SLO最紧）优先获高prob节点，累计≥3 expected accepted tokens达SLO目标；chatbot获中等节点（≥1 token）；summarization仅获最低需求节点；
	  - (3) throughput-optimized selection：剩余budget全局最高prob节点分配；
	  - (4) target LLM tree-based verification并行验证所有请求selected trees。
	==每个request生成beam，batch请求构造draft tree。==
  - 系统框架/Serving层：FlexFlow Serve + SLO-customized scheduler。
	  - Request manager维护per-request latency/token/SLO状态。
	  - FlashInfer batched prefill kernel改造用于speculation+verification。
	  - CUDA Graph复用减少draft decoding kernel launch开销。
	  - Dynamic d/w根据活跃请求数调节。
  - 编译框架层：论文未明确说明。
  - kernel调度层：
	  - FlashInfer batched prefill kernel改造用于tree-based parallel verification；
	  - CUDA Graph预捕获draft model的固定shape decoding steps。
  - 硬件架构层：4×NVIDIA A100 80GB GPU（NVLink），无定制硬件。
