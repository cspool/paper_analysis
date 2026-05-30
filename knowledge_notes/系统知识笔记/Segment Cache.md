## Segment Cache

术语解释
Flood 框架的 KV Cache 管理机制，替代 vLLM 的 PageAttention。在连续内存中分配 `[max_token_num, num_head, head_dim]` 形状的大 block，避免小 block 的计算效率损失。

术语是什么？
vLLM PageAttention 使用小 block（如 16）减少碎片但降低 GPU 计算效率。Segment Cache 为每个请求预分配连续 segment 获得更高效率。超长输出策略：(1) extend——扩展相邻空闲 segment；(2) append——追加新 segment；(3) wait——无可扩展时进入等待队列。原生支持 prefix caching。

术语一般如何实现？如何使用？
- 初始分配保守 size，按需扩展
- 适合 max_output_len 相对小的场景

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

---
