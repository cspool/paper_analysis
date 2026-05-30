## Elastic Training for Distributed Deep Learning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Elastic Training 是指分布式训练系统能够在节点数量动态变化（节点故障、preemption、或新增）的情况下继续训练，并充分利用所有当前可用节点的能力，而不是僵化地要求固定数量的节点或等待替换节点。弹性训练的核心要求是：(1) 节点减少时自动降级——利用剩余节点继续训练而无需等待替换；(2) 节点增加时自动扩展——将新节点纳入训练并提升吞吐；(3) 不损失训练正确性。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Lazarus 的弹性训练架构：

```
┌── Cluster State Management ───────────────────────────┐
│ Controller (CPU-only persistent node):                 │
│   - Maintains global view of alive nodes               │
│   - Agents send periodic heartbeats (TCP)              │
│   - Heartbeat timeout → mark node as failed            │
│   - New agent registration → mark node as available    │
└───────────────────────────────────────────────────────┘
                          ↓ (node membership change)
┌── Elastic Reconfiguration ───────────────────────────┐
│ Controller recomputes:                                 │
│   1. Expert allocation (Eq. 1) using all N' alive nodes│
│   2. MRO placement plan for N' nodes                   │
│   3. Greedy node mapping to minimize state migration   │
│                                                        │
│ Key: All N' nodes are fully utilized                    │
│   (no EP size × integer constraint)                    │
└───────────────────────────────────────────────────────┘
                          ↓
┌── State Migration ───────────────────────────────────┐
│ Batched NCCL send/recv:                                │
│   - Each node fetches missing expert states from       │
│     nodes that own them                                │
│   - Distributed among all owning nodes to balance load │
│                                                        │
│ Scale-up: Lazy reconfiguration (after current step)    │
│ Scale-down: Immediate reconfiguration (NCCL timeout)   │
└───────────────────────────────────────────────────────┘
                          ↓
┌── Training Resumes ──────────────────────────────────┐
│ All N' alive GPUs fully utilized                       │
│ Throughput ∝ number of alive nodes                     │
│ (No GPU idle, no wasted capacity)                      │
└───────────────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

早期弹性训练系统（TorchElastic）仅支持 data parallelism 的小模型。对于大模型，基于 pipeline parallelism 的系统（Oobleck, Bamboo, Parcae）利用 pipeline stage-device mapping 的灵活性实现弹性，但无法应用于使用 expert parallelism 的 MoE 模型。Lazarus 是首个专为 MoE 模型设计的弹性训练系统，通过 adaptive expert allocation（保证每个 expert 至少有 f 个 replicas）+ MRO placement（最大化恢复概率）+ efficient reconfiguration（最小化状态迁移）实现。Elastic training 的典型应用场景：spot/preemptible instances（cost savings up to 90%，preemption every 5-10 min）、大规模集群（Meta 报告 128K GPU 集群 MTTF 仅 14 分钟）。

涉及论文标题：
- Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement
