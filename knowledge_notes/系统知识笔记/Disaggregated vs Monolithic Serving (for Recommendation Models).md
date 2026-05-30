## Disaggregated vs Monolithic Serving (for Recommendation Models)

术语是什么？
在推荐系统推理场景中，Monolithic serving（单体式服务）指模型的全部计算——包括data preprocessing和neural network computation——在同一加速器（如GPU/MTIA）上执行；Disaggregated serving（分离式服务）指将preprocessing operators分配在CPU server上执行、neural network computation在accelerator上执行，通过网络的cross-tier通信完成数据交换。KernelEvolve论文揭示了disaggregation的根本原因：当preprocessing operators缺少native accelerator实现时，模型无法在accelerator上monolithic部署，被迫采用disaggregated architecture——这不是性能优化选择，而是binary deployment constraint。

从系统架构角度拆解术语：
论文Table 2给出了两种serving paradigm的latency对比（production MTIA模型）：
- Paradigm 1 (Monolithic): Client → MTIA Tier直接通信，preprocessing client-side完成。P99 latency: 61ms。
- Paradigm 2 (Disaggregated): Client → CPU Tier (preprocessing) → MTIA Tier (neural network)。P99 latency: 97ms。
  其中 α (Client→CPU): 65ms P75, β (CPU→MTIA): 48ms P75, γ (Data Preproc): 7ms P75。
  额外网络开销 δ = α − β − γ ≈ 10-20ms，占sub-100ms latency budget的10-20%。
Disaggregation的代价不仅仅是网络延迟：(1) cross-node serialization/deserialization开销；(2) cascading failure modes——任一tier故障导致整体不可用；(3) operational complexity——需要synchronized deployments和version compatibility；(4) TCO增加——冗余CPU infrastructure。

KernelEvolve的核心价值主张：通过自动生成preprocessing operators的accelerator-native kernel实现，将disaggregated architecture转变为monolithic deployment，消除10-20ms pure network tax。这使kernel availability从"性能优化问题"升级为"架构决策问题"。

术语一般如何实现？如何使用？
Monolithic serving要求所有operators（包括200+ data preprocessing operators）都有对应hardware platform的优化实现。这在vendor library coverage不成熟的新硬件（如MTIA）上尤为困难——论文Table 5显示MTIA v2i缺少clamp.out、gather.out、sort.values_stable、_unique2等多个ATen ops。KernelEvolve通过automated kernel generation填补这些缺口，实现从"missing kernel → disaggregated serving → 10-20ms network overhead"到"generated kernel → monolithic accelerator deployment → zero network tax"的架构转变。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
