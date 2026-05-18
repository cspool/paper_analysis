## Phase Transition Detection and Request Migration in LLM Serving

术语是什么？

Phase transition detection是指在reasoning-based LLM serving中，通过监控生成的token来识别reasoning phase到answering phase的切换点（如DeepSeek-R1的`<\think>` token）。一旦检测到phase transition，PASCAL触发request migration决策：根据当前instance和目标instance的GPU memory状态，决定是否将请求迁移到更适合处理answering phase的instance。

从系统架构角度拆解术语：

Phase transition migration流程（PASCAL）：
1. **Detection**：Instance monitor持续检查每个请求的newly generated token。当检测到特殊token（如`<\think>`，标志着reasoning结束）→触发migration decision。
2. **Algorithm 2 decision**：Instance-level scheduler评估所有SLO-compliant instances（ti=TRUE），选择reasoning请求数ri最小的instance作为migration target。理由：ri少→answering受reasoning preempt的风险低。
3. **Adaptive migration override**：比较current instance与target instance的GPU memory availability。若current instance有足够memory处理answering phase而target instance已满→保留在current instance。这避免：(a) unnecessary KV cache transfer overhead；(b) 将answering请求迁移到已满instance导致stall。
4. **KV cache transfer**：若决定迁移→通过100 Gbps fabric传输KV cache到目标instance。P99 transfer latency: 0.14s (AlpacaEval2.0) - 0.25s (Arena-Hard)，相对reasoning latency（tens to hundreds of seconds）可忽略。
5. **Resume on target instance**：请求进入target instance的low-priority queue，开始answering phase。

术语一般如何实现？如何使用？

Phase transition detection依赖于模型specific的phase boundary token（如DeepSeek-R1的`<\think>`）。Migration的必要性来自于：(1) reasoning-first scheduling可能使reasoning请求集中在少数instance；(2) 当reasoning完成后，继续在reasoning-heavy instance上执行answering可能因memory不足stall；(3) migration允许answering请求转移到reasoning负载更轻的instance。PASCAL(NoMigration)变体禁用migration后，tail TTFT worsens，P99 blocking latency（从phase transition到first scheduling）达27.39s（vs PASCAL near zero），answering phase SLO violation rate markedly更高。

与disaggregated inference migration的区别：disaggregated inference（如Splitwise）的KV cache transfer发生在prefill→decode stage boundary，且可提前预测transfer时机；PASCAL的migration发生在decoding stage内部的phase boundary，且transfer时机不可提前预测（需等待模型emit `<\think>` token）。PASCAL发现KV cache transfer overhead（~40ms for 2048 tokens）相对reasoning latency（tens to hundreds of seconds）可忽略。

涉及论文标题：
- PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models
