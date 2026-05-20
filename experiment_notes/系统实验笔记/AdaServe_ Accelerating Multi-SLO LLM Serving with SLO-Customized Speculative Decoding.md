## AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

- 属于Serving调度的实现是什么？实验比较什么？
  提出AdaServe，一个支持multi-SLO的LLM serving系统，核心实现是将multi-SLO serving形式化为带budget约束的token tree构造问题，并用SLO-customized speculative decoding实现细粒度per-request token分配。核心Serving调度实现包含：(1) SLO-customized scheduler：维护active request pool，执行speculation→SLO-customized selection→throughput-optimized selection→verification四阶段pipeline；(2) speculate阶段：用小draft model对每个请求做d步beam search，每步扩展w个候选token，形成有限candidate token tree，记录由draft logits近似的path probability；(3) SLO-customized selection阶段：先计算每个请求本轮至少需接受的expected accepted tokens以追上TPOT SLO，然后在该请求候选树中按path probability从高到低加入节点直到累计概率满足SLO所需或达per-request token limit；(4) throughput-optimized selection阶段：满足各请求SLO推进后若还有剩余token budget，全局排序所有剩余候选节点按path probability选择最高节点补入token tree；(5) verification阶段：所有请求selected draft token trees提交target LLM做tree-based parallel verification，接受匹配目标分布的token并返回verified tokens。实验比较vLLM（PagedAttention + continuous batching）、Sarathi-Serve（chunked prefill + co-batching）、vLLM-Spec(n)（固定speculation length n=4/6/8）、SpecInfer（原生speculative decoding引擎），在multi-SLO混合workload（coding copilot + chatbot + summarization）下评估SLO attainment、goodput、violation reduction。

- 硬件平台是什么，配置是什么。
  主评测节点：4×NVIDIA A100 80GB GPU，NVLink互连；CPU AMD EPYC 7763 (64 cores/128 threads)，256GB DRAM。Llama3.1-70B-Instruct使用4-way tensor parallelism + 4×A100 80G；Qwen2.5-32B-Instruct使用2-way tensor parallelism + 2×A100 80G。draft model与target LLM colocate在其中一张GPU上（Llama-3.2-1B-Instruct、Qwen2.5-0.5B-Instruct）。Artifact Appendix推荐复现配置：8×A100-SXM4-40GB或AWS p4de.24xlarge，CUDA 12.4，Docker + NVIDIA container runtime。

- 开源Serving框架是什么。修改了什么。
  基于FlexFlow Serve构建。核心修改：(1) 新增SLO-customized scheduler模块，实现speculation、SLO-customized selection、throughput-optimized selection和verification调度逻辑；(2) 将FlashInfer batched prefill kernel改造用于speculation steps和LLM tree-based verification；(3) 对draft model decoding使用CUDA Graph优化：从第二个speculation step到第d步，若活跃请求数相同则复用预捕获CUDA graph减少kernel launch overhead；(4) Request manager维护per-request状态（当前latency、已生成token数、TPOT SLO阈值），供scheduler决策使用。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：https://github.com/zikun-li/AdaServe-Artifact-Evaluation（artifact evaluation repo），Zenodo DOI: 10.5281/zenodo.17052619。以Llama3.1-70B-Instruct + coding copilot/chatbot/summarization混合workload为例说明调度流程：
  1. 部署：4×A100 80G，4-way tensor parallelism部署Llama3.1-70B-Instruct + Llama-3.2-1B-Instruct draft model colocate on GPU。FlexFlow Serve启动，加载SLO-customized scheduler。
  2. 请求到达：coding copilot（HumanEval, TPOT SLO 1.2×baseline latency）、chatbot（Alpaca, 50ms/token）、summarization（CNN/DailyMail, 150ms/token）三类请求按真实trace缩放后的到达率进入request pool。
  3. 每轮decoding iteration：scheduler从request pool取出活跃请求→计算每个请求本轮的SLO推进目标（expected accepted tokens ≥ 已消耗latency / TPOT_SLO - 已生成token数）→execution engine跑d步beam search生成candidate token tree（每步w个beam）→记录draft logits近似的path probability。
  4. SLO-customized selection：coding copilot（SLO最紧）优先获得高概率节点，累计概率达SLO所需；summarization（SLO最宽）仅获少量节点，节省budget。
  5. Throughput-optimized selection：若budget有剩余，全局选最高path probability节点分配给各请求，最大化总expected accepted tokens。
  6. Verification：所有请求selected token trees提交target LLM做tree-based verification→LLM并行验证→接受匹配token→返回corrected tokens。
  7. 结果：在不同RPS下AdaServe SLO attainment提升可达2.1×/1.6×，未满足SLO请求数减少4.3×/3.2×，goodput提升1.9×/1.7×。CPU selection overhead仅占总serving time的0.41%/0.31%。

