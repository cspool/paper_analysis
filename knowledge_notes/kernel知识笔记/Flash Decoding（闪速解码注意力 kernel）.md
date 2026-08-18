## Flash Decoding（闪速解码注意力 kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Flash Decoding 是 FlashAttention 团队提出的长上下文 decoding 优化 kernel：把 attention 计算沿 KV 序列长度维度拆分（split）到多个 SM 并行执行，各 SM 算部分 KV 块的 partial attention（含 partial logsumexp），最后通过 log-sum-exp reduction 合并各 partial results 得到精确注意力输出。与 FlashAttention 的区别：FlashAttention 针对 training/prefill 的单 query-多 KV 并行（一个 thread block 处理一个 query block + 逐步滚动 KV），解码时 query 只有 1 个、KV 很长，一个 thread block 顺序滚完整个 KV 会浪费并行度；Flash Decoding 用多个 thread block 并行扫不同 KV 段，吞吐接近同时处理所有 KV 段。Tetris（ISCA'26）在 decoding 阶段采用 Flash Decoding 计算 attention，配合 CUDAGraph 消除 kernel launch 开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Flash Decoding（1 query、KV 长 L、split 到 M 个 thread block）
for m in 0..M-1:                    # 每个 block 并行处理一段 KV
    o_m, lse_m = scan(Q, K[m·L/M:(m+1)·L/M], V[...])   # 段内 FlashAttention 式滚动 + online softmax
# 合并：lse = logsumexp(lse_0..lse_{M-1}); O = Σ_m exp(lse_m - lse) * o_m
```
Annotations: 每 block 独立扫描一段 KV，无跨 block 通信；合并阶段按 online softmax 规则加权，保证与顺序扫描数值等价；M 增大摊平单 block 顺序扫描延迟。
在 Tetris 中与分布式 decoding ring 配合：每个 decoding 实例作为部分请求的 master，Flash Decoding 负责本实例内多请求 batch 的 attention；CUDAGraph 把 decode 的 kernel 序列（含 Flash Decoding）录制为图，逐 token 重放消除 launch 开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：FlashAttention 2.6+ 集成（decode kernel）、vLLM/SGLang/TensorRT-LLM 默认 decode attention、SeerAttention-R 的 Block Sparse Flash Decoding 变体（跳过无效 KV blocks）；Tetris 中作为 decoding 计算组件（A100 集群，配合 Flash Attention prefill）。使用：长上下文 decode 阶段（KV 长、batch 小）的标准 kernel；支持 GQA（多 query head 打包同一 thread block）与 PagedAttention 兼容。Web 证据：Flash Decoding 官方 blog（Dao et al.）与 FlashAttention 2.6 release notes。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
