## TTFAT（Time-To-First-Answering-Token）

术语是什么？

TTFAT是面向reasoning-based LLM serving的延迟SLO指标，定义为从reasoning phase完成到第一个answering token（user-visible token）生成之间的延迟。在PASCAL中，TTFAT是answering phase QoE的两个组成因素之一（另一个是steady-state TPOT），目标是保持TTFAT near-instantaneous（≤0.25s）以确保从reasoning到answering的即时过渡。

TTFAT区别于TTFT：TTFT衡量从请求到达到第一个user-visible token的总延迟（包含prefill + reasoning + transition），而TTFAT仅衡量phase transition point之后的短暂过渡延迟。

从系统架构角度拆解术语：

TTFAT在PASCAL调度流程中的作用：
1. 请求完成reasoning phase（生成`<\think>` token后）→ instance-level scheduler调用Algorithm 2选择answering instance。
2. 若adaptive migration决定迁移→ KV cache transfer（P99延迟0.14-0.25s，100 Gbps fabric）→ transfer完成→answering phase开始→TTFAT计时起点。
3. 若adaptive migration决定不迁移→推理在current instance继续→TTFAT近乎为零。
4. TTFAT ≤ target（0.25s）是answering phase SLO达标的前提条件之一；超过阈值→QoE < 0.95→SLO violation。

术语一般如何实现？如何使用？

TTFAT通过instance monitor检测phase transition token来标记计时起点，首个answering token生成为计时终点。PASCAL通过两个机制保证低TTFAT：(1) Algorithm 2选择reasoning负载最小的instance，减少answering请求被高优先级reasoning请求block的风险；(2) adaptive migration避免将请求迁移到已满instance导致stall。在FCFS baseline中，HoL blocking使TTFAT显著增加（reasoning完成后answering被block在长请求之后），导致高SLO violation rate。

涉及论文标题：
- PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models

---
