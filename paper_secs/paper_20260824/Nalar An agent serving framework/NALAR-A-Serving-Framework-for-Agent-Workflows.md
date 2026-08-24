# NALAR: A Serving Framework for Agent Workflows

Marco Laju *UT-Austin*

Donghyun Son *UT-Austin*

Saurabh Agarwal<sup>∗</sup> *UT-Austin*

Nitin Kedia *UT-Austin*

Myungjin Lee *Cisco-Research*

Jayanth Srinivasa *Cisco-Research*

Aditya Akella *UT-Austin*

## Abstract

LLM-driven agentic applications increasingly automate complex, multi-step tasks, but serving them efficiently remains challenging due to heterogeneous components, dynamic and model-driven control flow, long-running state, and unpredictable latencies. NALAR is a ground-up agent-serving framework that cleanly separates workflow specification from execution while providing the runtime visibility and control needed for robust performance. NALAR preserves full Python expressiveness, using lightweight auto-generated stubs that turn agent and tool invocations into futures carrying dependency and context metadata. A managed state layer decouples logical state from physical placement, enabling safe reuse, migration, and consistent retry behavior. A two-level control architecture combines global policy computation with local event-driven enforcement to support adaptive routing, scheduling, and resource management across evolving workflows. Together, these mechanisms allow NALAR to deliver scalable, efficient, and policy-driven serving of heterogeneous agentic applications without burdening developers with orchestration logic. Across three agentic workloads, NALAR cuts tail latency by 34–74%, achieves up to 2.9× speedups, sustains 80 RPS where baselines fail, and scales to 130K futures with sub-500 ms control overhead.

