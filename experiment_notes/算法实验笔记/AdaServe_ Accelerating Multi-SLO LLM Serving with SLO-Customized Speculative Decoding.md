## AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

- 属于算法pipeline的实现是什么？实验比较什么？
  提出SLO-customized speculative decoding的speculate-select-verify pipeline：将multi-SLO serving形式化为带token budget约束的token tree构造问题，在每次decoding iteration中为不同SLO请求分配不同数量的speculation token tree节点，使严格SLO请求获得更多验证token、宽松SLO请求节省budget给其他请求。核心算法分三阶段：(1) Speculation phase：用小draft model对每个请求做d步beam search（每步width=w），从root（每个请求最后生成的token）开始逐步扩展候选child tokens，记录由draft logits近似的path probability；(2) SLO-customized selection：贪心为每个请求选择最高path probability节点直到累计expected accepted tokens满足其TPOT SLO约束，严格请求优先、宽松请求节省；(3) Throughput-optimized selection：满足所有SLO后剩余budget按全局path probability排序分配给各请求，最大化总expected accepted tokens。系统根据活跃请求数动态调节speculation depth d和beam width w。实验比较vLLM（continuous batching）、Sarathi-Serve（chunked prefill）、vLLM-Spec(n)（固定speculation length n=4/6/8）、SpecInfer（静态speculative decoding），评估不同RPS、严格请求比例、SLO scale下的SLO attainment和goodput。

- 硬件平台是什么，配置是什么。
  4×NVIDIA A100 80GB GPU (NVLink)，AMD EPYC 7763 (64 cores/128 threads)，256GB DRAM。Llama3.1-70B-Instruct (4-way TP) + Llama-3.2-1B-Instruct draft model；Qwen2.5-32B-Instruct (2-way TP) + Qwen2.5-0.5B-Instruct draft model。CUDA 12.4。

- 模型是什么。数据集和bench分别是什么。
  Target LLM: Llama3.1-70B-Instruct（4-way tensor parallelism）、Qwen2.5-32B-Instruct（2-way tensor parallelism）。Draft model: Llama-3.2-1B-Instruct、Qwen2.5-0.5B-Instruct（不做任务特定finetune）。Workload数据集：coding copilot (HumanEval, SLO=1.2×baseline latency)、chatbot (Alpaca, SLO=50ms/token)、summarization (CNN/DailyMail, SLO=150ms/token)。请求到达率：源自真实trace经截断与缩放形成不同RPS。指标：SLO attainment（满足TPOT SLO的请求比例）、goodput（成功满足SLO的请求产生的tokens/s）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://github.com/zikun-li/AdaServe-Artifact-Evaluation（artifact evaluation），Zenodo DOI: 10.5281/zenodo.17052619。
  SLO-customized speculative decoding算法pipeline（以batch含coding copilot + summarization两请求为例）：
  1. 初始化：scheduler取活跃请求→计算每请求本轮SLO推进目标：expected_accepted_tokens ≥ latency_elapsed / TPOT_SLO - tokens_generated。coding copilot（SLO紧）需求高→假设需≥3 expected accepted tokens；summarization（SLO宽）需求低→假设需≥1。
  2. Speculation phase：draft model对coding copilot做d步beam search（d=4, w=3）→生成候选tree节点（root→12个candidate nodes），每个节点记录draft logit作为path probability近似。summarization同样生成候选tree。
  3. SLO-customized selection：先处理coding copilot（需求更高）→在候选tree中按path prob从高到低选节点：node_A(0.8)→node_B(0.6)→node_C(0.45)，累计1.85 expected tokens，加入SLO tree。再处理summarization→选node_X(0.7)，累计0.7→已达≥1需求，停止。
  4. Throughput-optimized selection：若token budget=6，SLO阶段已用4个节点→剩余2个budget。全局排序所有请求未选候选节点→选最高path prob两节点补入各自tree。
  5. Verification：所有请求selected token trees提交target LLM做tree-based parallel verification→LLM并行验证共享前缀的多条speculative path→接受匹配token，拒绝错误分支返回correction token。
  6. 状态更新：accepted tokens写回request pool→若coding copilot接受3个token（SLO达成）→下一轮目标降低；若summarization仅接受0个（概率较低）→但SLO宽仍可容忍。结果：同一batch不同请求在一次大模型验证中前进不同token数，打破continuous batching的统一per-token latency。
  7. 关键trade-off：path probability用draft logits近似而非真实分布→效果依赖draft model与target一致性；per-request token limit防止某请求吞过多低概率budget→极端紧SLO可能只能被尽量推进而非完全满足。
