## Token-Level Dependency Tracking in Async MoE Serving（异步MoE推理中的Token级依赖追踪）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token-Level Dependency Tracking 是 AMoE 系统中确保异步乱序执行语义正确性的机制。在传统同步 EP 中，barrier all-to-all 保证了所有 token 在进入下一层时已完全处理完毕。在 AEP 中，tokens 可能乱序到达和乱序执行——同一请求的不同 token 可能分布在不同的 µ-queue 中，以不同的顺序被调度执行。因此需要为每个 token 维护 metadata 以追踪其在模型中的执行位置和依赖关系。

从系统架构角度拆解术语，给出术语在系统架构中运转流程的具体例子。
每个 token 携带的 metadata（Table 1）：
- **RequestID**：绑定 token 到具体的 serving request，确保输出 token 合并回正确的用户请求。由于 token 可能因异步 queuing 和执行而 shuffle，无法仅凭 global batch 中的位置推断 RequestID。
- **LayerID**：<block#> + <expert#> 或 <attn DP rank>，指示该 token 应该作为哪个 layer 的输入。Receptor 据此分流入 µ-queue。
- **Tensors[]**：GPU 上 input tensor 数据的引用（可多个，如 Top-K merge 时需要多路 input）。
- **Prefill_length**：用于 attention 计算的原始请求长度。
- **Topk_weights**：用于 Top-K token merge 的加权系数。

Top-K 依赖追踪：当 K > 1 时，每个 token 被复制 K 份分发到 K 个 expert。Receptor 维护一个 token pool，通过 <RequestID, LayerID> 元组查找——当所有 K 路 expert 输出都到达时（merge 完成），token 才被移入对应的 attention µ-queue。

术语一般如何实现？如何使用？
在 AMoE 中，metadata 在 CPU 上追踪（C++ struct），tensor 数据保持在 GPU memory 中以最小化 CPU-GPU 拷贝。Tensor 引用通过指针或 handle 关联，而非拷贝数据。Receptor 和 Dispatcher 在 POSIX backend threads 上执行，通过 pybind11 与 Python 端 Executor 交互。

涉及论文标题：
- Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony
