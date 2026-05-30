## Dispatch-Combine Pattern (MoE Expert Parallelism)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dispatch-Combine 是 Expert Parallelism (EP) 中 MoE 层 forward 的标准两阶段通信模式。由于 experts 分布在不同 GPU 上，tokens 需通过 All-to-All 发送到持有目标 expert 的 GPU（dispatch），计算完成后输出通过 All-to-All reverse 传回原始 GPU（combine）。EP size = P 时，每 GPU 持有 M=N/P 个 experts。

标准流程（Alg. 1, LLEP）：sort routing indices → index_select reorder tokens → All-to-All dispatch → native expert Grouped-GEMM → All-to-All combine → reverse_sort → reshape → sum over K。

LLEP 扩展（Alg. 4）：(a) λ 阈值判断：max(l)/mean(l) < λ 时回退标准 EP；(b) dispatch 目标扩展为 native ∪ foreign experts；(c) 新增 P2P 权重传输；(d) backward: foreign expert 梯度 P2P 回传原生 GPU 累加。

从系统架构角度：**Dispatch**: GPU p sends {B_i} tokens → receives {B̂_i} from all GPUs. **Combine**: GPU p sends expert outputs {Ĥ_i} back → receives output for its original tokens. 通信量 = 2×total_tokens×d_model×sizeof(dtype)。sort/index_select 为 memory-intensive 操作。

术语一般如何实现？如何使用？

标准实现依赖 NCCL All-to-All collective。高效变体：DeepEP (NVSHMEM kernel)、Triton-Distributed (compiler 层融合通信计算)、FUSCO (融合 permute+通信消除显式重排)。vLLM/SGLang 推理变体用 all-reduce 替代 all-to-all（activaion 全复制, expert 权重按 EP 分布）。

涉及论文标题：
- Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts
