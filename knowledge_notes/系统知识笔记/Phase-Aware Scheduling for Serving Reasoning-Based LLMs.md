## Phase-Aware Scheduling for Serving Reasoning-Based LLMs

术语是什么？

Phase-aware scheduling是PASCAL提出的面向reasoning-based LLM的两级调度架构，核心思想是区分reasoning phase（hidden CoT token生成）和answering phase（user-visible token生成），对不同phase应用不同调度策略。Reasoning phase latency高度sensitive to interruption（直接影响TTFT），需最小化blocking/preemption；answering phase是threshold-sensitive（只需满足TTFAT和TPOT目标），可以容忍controlled preemption。

从系统架构角度拆解术语：

PASCAL的phase-aware调度包含两级：

**Instance-level scheduler**：
- Algorithm 1（reasoning请求）：选择SLO-compliant instances（token pacer状态为TRUE）中KV cache footprint mi最小的instance。理由：mi小→少量active requests→accommodate新请求无interference + 短attention执行时间→利于meeting TTFT。
- Algorithm 2（answering请求）：选择reasoning请求数ri最少的instance。理由：reasoning在high-priority queue优先分配GPU memory→ri少意味着更多free memory给answering。若无SLO-compliant instance，选ri+ai最小的instance（ai=尚未消耗第一个time quantum的answering请求数）。

**Intra-instance scheduler**：
- High-priority queue：存放reasoning phase请求，RR调度（token quantum=500），优先分配GPU memory。
- Low-priority queue：存放answering phase请求，RR调度 + token pacer平滑输出速率。
- 每queue内部RR保证fairness；跨queue的hierarchical priority保证reasoning总是preempt answering。

3. **Phase Transition & Adaptive Migration**：在检测到phase boundary token时，Algorithm 2决定是否迁移请求。Adaptive migration override：若当前instance GPU memory充足但目标instance已满→保留在当前instance，避免不必要的KV cache transfer和stall。

4. **Conditional Demotion**：当single reasoning请求KV cache超过5000 tokens时demote到low-priority queue，作为answering处理。防止单个超长reasoning请求垄断GPU memory。

术语一般如何实现？如何使用？

PASCAL实现为cluster-level simulator（基于vLLM v0.6.1 profiling data），模拟8×H100 96GB instances经100 Gbps fabric互联。Simulator建模：(1) Instance monitor采集token pacer状态和queue occupancy；(2) Instance-level scheduler执行Algorithm 1/2做placement决策；(3) Intra-instance scheduler管理high/low priority queues的RR调度和token pacing；(4) KV cache transfer模拟（phase transition时的跨instance迁移）。验证用真实hardware（H100+Intel Xeon Platinum 8558），MAPE 1.62%。

PASCAL与FCFS（vLLM默认）和RR对比：tail TTFT降低up to 72%（vs FCFS, Arena-Hard），SLO violation rate consistently更低或comparable，throughput差异<3%。Phase-aware scheduling与disaggregated inference（如DistServe, Splitwise）正交：disaggregated inference分离prefill/decode stage，PASCAL在decoding stage内部区分reasoning/answering phase。

涉及论文标题：
- PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models

---
