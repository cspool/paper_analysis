## Merged Expert Representation（合并专家表示）

术语是什么？
Merged Expert Representation 是 MoEsaic 在 expert deduplication 完成后的后续优化：将去重后共享相同底层 tensor 的 expert 在模型结构中合并为单一 nn.Parameter 对象。即使 multiple expert 共享同一 tensor（dedup 完成），vLLM 的 Independent Expert Representation 使它们仍由不同 nn.Parameter 对象表示，Triton kernel 会独立处理每个 expert。Merged representation 消除这种"逻辑分离、物理共享"的不一致，使不同 client 路由到相同 expert 的 token 在单次 kernel 调用中被自动批处理。

从系统架构角度拆解术语：
在 MoEsaic 的推理循环中，Merged Expert Representation 的作用：

1. **去重后状态**：去重后 expert A 在 client 1 和 client 2 中共享同一底层 tensor，但分别由 nn.Parameter_1 和 nn.Parameter_2 表示。Triton kernel 看到的是"两个 expert"，分别处理来自 client 1 的 token 和 client 2 的 token。
2. **合并操作**：MoEsaic 在模型初始化完成后（所有 expert 加载并去重），扫描所有共享同一 tensor 的 nn.Parameter 组，合并为单一 nn.Parameter。
3. **Gate Mapping 重映射**：每个 MoE 的 gate 输出 expert ID 列表 → 通过 gate mapping 表映射到 merged expert ID。例如 client 1 gate 输出的 expert_A → merged_expert_3，client 2 gate 输出的 expert_A → 同一个 merged_expert_3。
4. **自动批处理效果**：Triton kernel 执行时，merged_expert_3 收到来自 client 1 和 client 2 的所有 token 作为单个 batch 处理。批量越大，GPU SM 占用率越高——4 instances Mixtral-3x1B 全部 shared experts 时，per-expert batch size 从 ~10 增至 ~42（4×，batch_size=128 时）。

术语一般如何实现？如何使用？
- 在 vLLM 的模型结构中，通过修改 expert 的 `nn.Module` 层级结构实现：去重后的 expert 参数对象被替换为单一共享的参数对象引用，gate 层的输出索引被重映射。
- 关键收益来自 Triton kernel 的 batch efficiency——GPU 在相同 kernel launch overhead 下处理更多 token，SM 占用率提升。
- 论文实验（Figure 4）显示：4 instances Mixtral-3x1B, batch_size=128 时，全 unique experts 下 per-expert 平均 ~10 requests，全 shared experts 下 per-expert 平均 ~42 requests。NVIDIA Nsight 测量的 SM 占用率随共享比例提升而下降（相同计算量被更高效完成）。

涉及论文标题：
- MoEsaic: Shared Mixture of Experts
