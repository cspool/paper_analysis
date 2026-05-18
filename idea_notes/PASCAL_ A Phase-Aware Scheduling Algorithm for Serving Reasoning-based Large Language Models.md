## PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models

- baseline方法是什么？
  Baseline是vLLM默认的FCFS（First-Come-First-Served）scheduler和Round-Robin（RR）time-sharing scheduler，两者均不区分reasoning phase和answering phase。

  FCFS全栈执行例子（以DeepSeek-R1-Distill-Qwen-32B + vLLM 0.6.1 + H100 96GB, memory-constrained 50% KV cache capacity为例）：
  - 算法层：Reasoning-based LLM inference（CoT decoding）。模型生成reasoning tokens（hidden, r1-rN）→ 遇到`<\think>`标记 → 生成answering tokens（user-visible, t1-tM）。FCFS不区分两阶段。
  - 系统框架/Serving层（FCFS）：请求按到达顺序进入batch。当KV cache占满GPU memory时，新请求被block在队列中等待已运行请求完成→ Head-of-Line (HoL) blocking。即使短reasoning请求（128 tokens）也需等待长请求完成，latency up to 5.14× vs oracle。Reasoning phase被block→TTFT膨胀。Answering phase同样被block→TTFAT（Time-To-First-Answering-Token）延迟→SLO violation。
  - 系统框架/Serving层（RR）：RR分配固定token quantum（500 tokens），轮转调度所有请求。对short reasoning请求减轻HoL blocking（TTFT更低），但frequent preemption fragment execution→long reasoning请求（2048 tokens）latency up to 1.75× vs oracle。Answering phase通过preemption避免HoL blocking→SLO attainment较高。但RR不能enforce全局"reasoning-first"优先级→reasoning和answering在同一batch竞争。
  - 编译框架层：论文未明确说明（PyTorch 2.4.0 + CUDA 12.1默认路径）。
  - kernel调度层：论文未明确说明（vLLM默认kernel）。
  - 硬件架构层：NVIDIA H100 96 GB GPU + Intel Xeon Platinum 8558 CPU + 256 GB DDR5，PCIe 5.0。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出PASCAL，通过phase-aware两级调度架构（instance-level + intra-instance）解决reasoning-based LLM serving中baseline不区分reasoning/answering phase导致的TTFT膨胀和SLO violation问题。

  **缺陷1：FCFS/RR不区分reasoning phase和answering phase → reasoning latency（直接决定TTFT）被blocking或preemption不必要地延长**
  → **PASCAL方案**：Intra-instance hierarchical priority queue。High-priority queue存放reasoning请求，优先调度、优先分配GPU memory；Low-priority queue存放answering请求。原因：reasoning phase latency高度sensitive to interruption（影响TTFT），answering phase是threshold-sensitive（只需满足TTFAT≤0.25s + TPOT≤100ms即可）。RR用于high-priority queue保证short reasoning请求低TTFT；RR+token pacer用于low-priority queue保证answering phase SLO compliance。

  **缺陷2：FCFS/RR无跨instance感知 → 可能导致reasoning/answering请求分布不均，某些instance因memory压力导致answering phase SLO violation**
  → **PASCAL方案**：Instance-level scheduler双算法。Algorithm 1（reasoning请求）：选择SLO-compliant instances中KV cache footprint最小的instance，最小化新请求对现有请求干扰的同时减小attention执行时间。Algorithm 2（answering请求）：选择reasoning请求数ri最少的instance，避免answering被高优先级reasoning抢占memory。通过instance monitor持续采集token pacer状态和queue occupancy指导placement。

  **缺陷3：Baseline无phase transition处理 → reasoning完成后answering请求可能与同instance的reasoning请求竞争资源，导致answering phase stall**
  → **PASCAL方案**：Phase-aware request migration。在检测到phase transition token（如`<\think>`）时，根据Algorithm 2决定是否迁移到更合适的instance。Adaptive migration override：若当前instance GPU memory充足而目标instance已满，保留在当前instance避免不必要的KV cache transfer overhead（P99 transfer latency仅0.14-0.25 sec, negligible vs reasoning latency）。

  **缺陷4：RR的fairness-oriented priority导致长reasoning请求持续被preempt但短request answering phase却获得高优先级**
  → **PASCAL方案**：Conditional demotion。当单个reasoning请求KV cache超过5000 tokens时demote到low-priority queue，释放GPU memory给answering请求。Global "reasoning-first" priority确保所有实例一致：reasoning always preempts answering。

  PASCAL全栈执行例子（以DeepSeek-R1-Distill-Qwen-32B + AlpacaEval2.0 + 8-instance H100 cluster为例）：
  - 算法层：DeepSeek-R1-Distill-Qwen-32B CoT decoding，reasoning tokens → `<\think>` → answering tokens。PASCAL不修改模型或推理算法，仅通过调度区分两阶段。
  - 系统框架/Serving层：Request到达→Algorithm 1选instance（reasoning phase）→ high-priority queue → RR调度（token quantum=500）→ 检测`<\think>` → Algorithm 2选instance（answering phase）+ adaptive migration → low-priority queue → RR + token pacer。Tail TTFT vs FCFS: 减少up to 72%（Arena-Hard, 64.21 sec absolute reduction）；vs RR: 减少up to 33%（89.91 sec absolute reduction）。SLO violation rate consistently lower或comparable。Throughput差异<3%。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：8×H100 96GB GPU cluster，100 Gbps fabric互联。KV cache transfer: P99 0.14-0.25 sec（negligible vs tens-to-hundreds sec reasoning latency）。
