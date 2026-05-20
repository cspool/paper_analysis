## Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

- 属于Serving调度的实现是什么？实验比较什么？
  提出Laser，一个面向multi-SLO LLM serving的layer-level scheduling系统，将调度粒度从iteration降为layer。核心实现包含：(1) layer-level chunked prefill：prefill Scheduler在每个transformer layer边界评估新请求的TTFT slack，决定是否抢占当前chunk并优先执行latency-critical请求，或动态合并relaxed请求到当前chunk以提升GPU利用率；(2) layer-level decode batching：decode Planner为每个请求生成execution plan（L: 每iteration执行的层数，O: 调度offset），通过模块化latency model预测per-iteration latency，贪心调整relaxed请求的层数以最大化batch容量；(3) inter-instance request dispatching：Global Controller对prefill侧选择slack最大的实例，对decode侧按SLO-homogeneous group管理实例并按TBT increment最小原则分配请求；(4) intermediate state cache与fused CUDA kernel：GPU上维护layer级中间状态缓存（prefill 16384 tokens，decode 2048 tokens），fused kernel合并state caching/retrieval。实验比较SLO attainment和goodput（90% SLO attainment下的throughput），对比Sarathi-Serve（prefill-decode aggregation + chunked prefill + EDF）和DistServe（prefill-decode disaggregation + iteration-level scheduling + EDF）。

- 硬件平台是什么，配置是什么。
  4台物理主机，每台4×NVIDIA A100 80GB GPU，主机间100 Gbps LAN，机内GPU NVLink互联。模型：Qwen2.5-14B (1-way TP)、Qwen2.5-32B (2-way TP)、LLaMA-3-70B (4-way TP)。

- 开源Serving框架是什么。修改了什么。
  基于vLLM + Ray构建。主要修改：(1) 新增layer-level chunked prefill：prefill Executor支持在layer边界保存/恢复intermediate state（hidden states），Scheduler实现EDF队列+slack-based抢占决策+动态chunk合并；(2) 新增layer-level decode batching：decode Planner实现latency analysis（分段线性model stateless module + 线性model stateful attention）+ execution plan construction（贪心减少relaxed请求层数，offset平衡同group负载），Executor支持按L/O plan在layer边界切换请求；(3) 新增Global Controller：prefill instance selection（slack-aware）、decode group-based assignment（SLO-homogeneous groups + decentralized performance evaluation）、instance group management（根据arrival rate动态调整group大小）；(4) intermediate cache：GPU memory内维护layer-level intermediate state缓存（类似KV cache），state manager索引active requests，fused CUDA kernel合并caching/retrieval；(5) KV cache migration按layer粒度异步进行，与prefill computation overlap；(6) 新增offline profiling流程：系统初始化时测量token count和context length变化下的module latency，拟合latency model参数，profiling在2秒内完成。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确说明代码开源地址。Laser使用流程：
  1. 系统初始化：Global Controller离线profiling各serving instance的layer-level module latency（stateless模块用分段线性函数，self-attention用线性函数），拟合latency model系数，建立instance group。
  2. 请求处理：(a) 新请求到达Global Controller → prefill侧选择TTFT slack最大的prefill实例；(b) prefill instance的Scheduler计算新请求slack，若当前chunk剩余iteration时间危及新请求SLO则在layer边界抢占，判断是否合并后共同执行；若无需抢占且队列为空则动态合并；否则EDF入队；(c) prefill完成后KV cache按layer粒度异步迁移到decode instance。
  3. Decode阶段：(a) Global Controller按group-based算法分配decode请求到SLO-homogeneous group内TBT increment最小的实例；(b) decode Planner触发execution plan更新（仅request arrival/departure或latency接近最严格SLO时），latency analysis估计per-iteration latency → 若超标则选SLO最relaxed的请求减少其L并选最优offset → 若低于目标则尝试恢复strict请求全层执行；(c) Executor按plan执行decode computation，在layer边界切换请求。
  4. 在4×A100 80GB × 4 host集群上，Qwen-14B+ShareGPT/HumanEval/LongBench混合workload下Laser相比DistServe和Sarathi-Serve分别提升goodput 43.4%和1.67×；99% attainment目标下提升最高1.85×；relaxed请求占比高时改进从19.4%增至>86%；tight 0.8× SLO下goodput gain最高6.25× vs Sarathi-Serve。
  Laser的作用：将LLM serving的调度粒度从完整的模型forward pass（iteration）细化到单个transformer layer，使系统能在layer边界进行抢占、合并和差异化执行。prefill侧避免长prompt的head-of-line blocking和低效小chunk；decode侧允许relaxed请求部分执行layer以释放batch容量给更多relaxed请求。适用于共享foundation model、多应用多SLO、prefill-decode disaggregation的production serving环境。

