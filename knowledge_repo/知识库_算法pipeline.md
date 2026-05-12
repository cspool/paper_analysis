## Prefill Phase

术语解释：
Prefill phase 是 LLM 推理中处理输入上下文并生成首个 token 前状态的阶段，会为输入 token 构建 KV cache。论文将 prefill 描述为更偏 compute-intensive 的阶段，其计算需求随 input length 和 reused context length 增长，直接影响 TTFT。

术语关联术语的使用例子：
在 MuxWise 中，prefill 不再以完整 phase 或 chunk 为最小执行单位，而是被拆成 transformer layer 粒度的 prefill layers。调度器根据 decode SLO 将剩余 SM 分配给这些 layers，并可在长 prefill 与短 prefill 之间进行非递归抢占。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## Decode Phase

术语解释：
Decode phase 是 LLM 推理中逐 token 生成输出的迭代阶段，每一轮 decode iteration 通常访问已有 KV cache 并生成新的 key/value。论文将 decode 描述为更偏 memory-intensive 的阶段，并用 TBT 衡量每个输出 token 之间的延迟。

术语关联术语的使用例子：
MuxWise 优先为 decode 分配 just-enough SM 以满足 TBT SLO，再用剩余 SM 执行 prefill。由于 decode batch size 变化较有限，论文认为 decode 适合 CUDA Graph / graph-level scheduling 来降低 launch overhead。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## Reused Context

术语解释：
Reused context 指多轮对话、agent workflow 或共享 system prompt 中已经计算过并可复用的上下文 token。论文认为 reused context 是影响现代 LLM serving 的关键变量：它会增加 prefill 和 decode 对 KV cache 的访问，并改变 compute 与 memory 需求。

术语关联术语的使用例子：
Conversation 和 Tool&Agent workloads 中，上一轮输出会成为下一轮输入上下文，形成很长 reused context。Chunked-prefill 在这种情况下会让 prefill chunk 反复读取历史 KV cache，导致 TBT 随 reused context 增长而恶化；MuxWise 通过共享 KV cache pool 和 PD multiplexing 缓解该问题。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## Transformer Layer

术语解释：
Transformer layer 是 LLM 的基础网络层，通常包含 attention 和 FFN。论文没有修改 Transformer 数学结构，而是利用模型天然由多层 transformer blocks 组成这一事实，将 prefill phase 拆成 layer-wise execution 单元。

术语关联术语的使用例子：
MuxWise 的 bubble-less multiplex engine 按 transformer layer 发射 prefill layers，使调度器可以选择只运行若干层来对齐 decode iteration 时间窗口。这与 chunked-prefill 按 token chunk 切分 prefill 的方式不同。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
