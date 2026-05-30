## FineEP All-to-All Communication Group Expansion

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FineEP All-to-All Communication Group Expansion 是 FineMoE 中为支持 token scheduling 而扩展 all-to-all 通信组大小的设计。传统 EP 中 all-to-all 在 EP group 内执行（大小为 EP_degree），token 只能在 EP group 内的 GPU 间 dispatch。FineEP 将通信组扩展为 FineEP group（大小为 d × EP_degree，其中 d ≤ DP_degree/EP_degree），允许 token 在更大的 GPU 集合间进行跨 EDP group 调度。通信组扩大 d 倍意味着：调度空间从 1 个 replica/expert 扩展到 d 个 replica/expert（EDP）；但同时通信量可能增加（intra-node → inter-node 转换）。

从kernel调度角度拆解术语：
通信组扩展对 all-to-all 的影响：
- 通信量：每 token 仍需发送到 1 个 replica（top-2 时 2 tokens × 1 dispatch），通信量不变（每个 token 只发给 1 个 replica）。
- 通信模式：d× group size 的 all-to-all（NCCL 或 DeepEP），大 group 的 collective 可能有更多并行度但也有更多同步开销。
- Inter-node 转换：当 d×EP_degree > GPUs per node 时，部分 intra-node 通信变为 inter-node（额外延迟）。但两种情况影响小：(1) d×EP_degree ≤ GPUs/node → 全部 intra-node；(2) EP_degree 极大（如 DeepSeek-V3 的 64）→ 几乎全是 inter-node。

术语一般如何实现？如何使用？
- 在 Megatron-LM 中通过修改 NCCL communicator 大小（从 EP group → FineEP group）实现。
- Locality-Aware Routing（Algorithm 1）配合减少实际跨 GPU 通信量。
- 支持两种 backend：NCCL（默认）+ DeepEP（高性能 all-to-all）。
- 注意事项：(a) FineEP + DeepEP 的数据格式不兼容会引入额外 pre-processing 开销；(b) Pipelining FineEP（Appendix A.2）可进一步隐藏通信延迟。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling
