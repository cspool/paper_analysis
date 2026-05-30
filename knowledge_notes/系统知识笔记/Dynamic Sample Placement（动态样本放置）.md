## Dynamic Sample Placement（动态样本放置）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Sample Placement 是 NetMoE 提出的核心优化技术：在 MoE 分布式训练的每个 MoE 层中，根据 gating network 的 routing 结果，动态调整训练样本在各 GPU 上的放置位置（SmpDev），使更多 tokens 的 All-to-All 通信走高速 intra-node 通道（NVLink, 400 GB/s）而非低速 inter-node 通道（InfiniBand, 100 GB/s）。与 Dynamic Expert Placement（调整 expert 参数在不同 GPU 上的分布）不同，动态样本放置的调整不涉及模型参数传输——样本位置调整与 All-to-All Gather 操作融合，零额外通信开销。其理论依据是 data locality：同一 training sample 中的 tokens 倾向于路由到相同 expert（Xue et al. 2024; Jiang et al. 2024），因此样本级别的重放置可以有效减少跨节点通信。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
NetMoE 中 Dynamic Sample Placement 的系统流程（Algorithm 1）：
1. **Gating**：每个 GPU 上的 tokens 经过 gating network，得到 routing 结果 `route ∈ N^{I×L×K}`。
2. **计算通信矩阵**：统计 `num_{i,e}`（sample i 发往 expert e 的 token 数），在 GPU 上计算后传至 CPU。
3. **Stage 1（跨节点优化）**：CPU 后台线程将"sample 到 node 的分配"建模为二分图最小权完美匹配——左侧 I 个 samples，右侧 N 个 nodes（每 node 容纳 I/N samples，通过复制 B=I/N 次实现）。边权重为 sample i 放在 node n 时的跨节点通信量（gather + 下一层 scatter）。KM 算法 O(I³) 求解，得到每个 sample 的目标 node。
4. **Stage 2（节点内优化）**：每个 node 独立求解第二个二分图匹配——将分配到该 node 的 I/N 个 samples 分配到 J/N 个 GPUs，最小化 intra-node 通信量。
5. **All-to-All Gather with optimized SmpDev**：tokens 不再返回原 GPU，而是按 Step 3-4 求解的新 SmpDev 发送到目标 GPU，完成 sample 放置调整。
6. **Expert Residual Inlining**：确保计算正确性（残差在 scatter 后、gather 前执行）。
7. **关键时序**：KM 求解在 CPU 后台线程执行，与 All-to-All scatter + expert computation 重叠，Table 4 显示 I/J=4 时 KM 求解 0.48ms << scatter+computation 7.13ms，零额外开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- NetMoE 在 PyTorch 上实现，自定义操作用 C++/CUDA 编写。
- 核心数据结构：`num_{i,e}` 矩阵（I×E）、通信代价矩阵 `c_{i,n}` 和 `c'_{i,j}`（通过 Eq. 8 计算）、优化后的 `SmpDev` 向量。
- KM 算法实现于 CPU（因 GPU 不擅长不规则的二分图匹配算法），通过后台线程与 GPU 计算/通信重叠。
- 兼容性：NetMoE 的 solver 接受 `ExpDev(·)`（expert placement）作为输入，因此可以与 Dynamic Expert Placement 技术叠加使用——先调整 expert placement，再基于新的 expert placement 求解最优 sample placement。
- 局限性：KM 算法 O(I³) 时间复杂度，I 受限于 gradient accumulation 下的 effective batch size（通常可接受范围内）。论文 I/J 最大测试到 24。

涉及论文标题：
- NetMoE: Accelerating MoE Training through Dynamic Sample Placement
