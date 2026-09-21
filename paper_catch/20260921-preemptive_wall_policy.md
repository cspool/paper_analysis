# 20260921 下载清单：GPU 抢占 / 共置调度（preemptive_wall_policy）
## 用法：python3 scripts/paper_download.py --file paper_catch/20260921-preemptive_wall_policy.md --output papers_pdf/paper_preemptive_wall_policy --dry-run
## 规则：`#` 行为说明（下载器跳过）；论文一行一条，链接为首选来源
## 已去重（本地已有，未列入）：μShare（experiment_notes 66）、Autellix/Agentix（paper_secs/paper_20260824）、GPreempt（human_notes/多任务-动态调度风暴笔记）、XSched（human_notes/多任务-动态调度风暴笔记/多任务抢占Preemption）
## REEF 仅在 human_notes/GPU架构笔记/GPU Preemption for LC+BE 有摘要一节，非完整笔记，保留

## A. 生产级在线/离线共置与抢占
- [Valve: Production Online-Offline Inference Colocation with Jointly-Bounded Preemption Latency and Rate](https://arxiv.org/abs/2604.07874)
- [Hummingbird: SLO-Oriented GPU Preemption at Microsecond-scale](https://arxiv.org/abs/2601.04071)
- [FastServe: Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920)

## B. GPU 抢占机制（yield / reset / 命令队列）
- [REEF: Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences](https://www.usenix.org/conference/osdi22/presentation/han)
- [PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications](https://www.usenix.org/conference/osdi20/presentation/bai)

## C. 细粒度 GPU 共享 / 干扰感知
- [Orion: Interference-aware, Fine-grained GPU Sharing for ML Applications](https://dl.acm.org/doi/10.1145/3627703.3629578)
- [TGS: Transparent GPU Sharing in Container Clouds for Deep Learning Workloads](https://www.usenix.org/conference/nsdi23/presentation/wu)
