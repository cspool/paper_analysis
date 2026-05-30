## Multi-tenant MoE Model Sharing（多租户MoE模型共享）

术语是什么？
Multi-tenant MoE Model Sharing 是 MoEsaic 提出的在单个 inference service 上同时托管多个 MoE model instance 的系统范式。传统的 multi-tenant serving 中每个 client 需部署独立 model instance（dedicated deployment），即使不同 instance 包含大量相同 experts。MoEsaic 通过 expert deduplication + merged expert representation + fused gate 实现共享部署：多个 client 的请求在同一 GPU 上执行，共享相同 experts 的显存和计算资源，同时通过独立 gate 和 model_id 路由保持 client 之间的请求隔离。

从系统架构角度拆解术语：
MoEsaic 的 multi-tenant 服务全流程：

1. **Model Registration**：client 通过 LoRA-like interface（类似 `model.add_experts_and_gates(new_experts, new_gates)`）向其 base MoE 添加新 experts 和 gates。无需系统重启（non-disruptive add/remove）。
2. **Shared Deployment**：多个 model instance 的 experts 经去重后共享 GPU 显存。非 MoE 层（如 attention）可保持 unique per-client。
3. **Request Routing**：每个请求携带 client 的 model_id。fused gate 按 model_id 选择对应的 gating network，输出被 gate mapping 映射到 merged expert。
4. **Batched Execution**：Merged expert representation 保证不同 client 路由到同一 expert 的 token 在同一 Triton kernel batch 中处理——实现跨 client 的自动批处理。
5. **Client Isolation**：gate mapping 确保每个 client 的 gating 语义不变（选出的 expert 集合与独立部署相同）。client 不能访问其他 client 的数据或模型参数（security isolation）。

关键设计原则：(1) Non-disruptive add/remove——可在无活跃推理时动态集成/移除 client，无需重启；(2) Independent client experience——每个 client 通过独立接口提交请求，内部共享对 client 透明；(3) Limited performance impact——所有 instance 共享后性能接近独立部署。

术语一般如何实现？如何使用？
- 类似 S-LoRA 的 multi-adapter sharing（共享 base model，各 adapter 独立），但扩展到 MoE expert 级别的共享（不要求共享 base model）。
- 实现参考：vLLM RFC #9203 的 xMoE interface——类似 `model.add_lora()` 的 `model.add_moe()` 接口。
- 适用场景：服务平台（如 HuggingFace Inference Endpoints、Together AI）需同时托管大量客户定制的 MoE 变体（从 off-the-shelf experts 组合而来）。
- 局限性：(1) 添加/删除 model instance 期间不能有活跃推理；(2) 仅共享 expert 参数，non-MoE 层各 client 仍需独立存储。

涉及论文标题：
- MoEsaic: Shared Mixture of Experts
