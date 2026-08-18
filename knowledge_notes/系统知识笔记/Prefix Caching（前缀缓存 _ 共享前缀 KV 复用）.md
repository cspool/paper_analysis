## Prefix Caching（前缀缓存 / 共享前缀 KV 复用）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
前缀缓存把多个请求共享的 prompt 前缀（系统提示、few-shot 示例、多轮历史、agent 工具定义）的 KV 只计算一次并共享，后续匹配请求直接复用，省 prefill 计算与 KV 显存。两种主流实现：SGLang RadixAttention（radix tree 按 token 序列索引，任意 token 边界匹配，LRU 逐出）与 vLLM Automatic Prefix Caching（块粒度哈希链：hash(前缀 tokens+块 tokens)→物理块，块边界对齐匹配）。收益：匹配场景 prefill/TTFT 降 2–10×、吞吐最高 ~5×。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
ConServe 的兼容设计（Discussion）：前缀缓存与 contiguity 不冲突——CUDA VMM 允许同一物理页映射到多个虚拟区域，把缓存前缀视为共享只读 KV 对象（前缀缓存元数据表：前缀标识哈希 → VMM backing handles 与页范围/偏移）；新请求照常预留自己的连续 VA slice，把共享前缀页 alias-map 到 slice 固定偏移区（tokens [0,prefix_len) → VA base+[0,prefix_bytes)），私有 KV 页紧随其后按需增长——不复制物理内存、不破坏 base+offset 算术，kernel 仍看到一条连续 slice。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：vLLM `enable_prefix_caching=True`、SGLang 默认开 RadixAttention、TRT-LLM `kv_cache_reuse=True`。局限：只复用前缀、分叉后不再共享（非前缀位置共享需 CacheBlend/SparseX 类方法）；vLLM 块哈希的块对齐问题（共享 15/16 token 也视为不同块）。Web 证据：vLLM APC 设计文档（https://docs.vllm.ai/en/latest/design/automatic_prefix_caching.html）与 RadixAttention 资料（SGLang 论文 arXiv:2312.07104）确认两种实现与逐出策略。

Tetris 补充视角（ISCA'26，CDSP 与前缀缓存的组合）：CDSP prefill 调度可无缝集成到支持前缀缓存的 prefill 实例池（同 Mooncake 思路）——请求到达时先找出持有缓存前缀的 prefill 实例，再按 SP 大小候选枚举不同 reuse ratio 配置，对每个配置调用 Algorithm 1 得对应最优 CDSP 执行策略，最后选使 TTFT 最小的策略。例：缓存前缀长 x 驻留 {P0,P1}，新 prompt 长 y，枚举 (1) 不复用 L=x+y、A=∅；(2) 复用一半 L=x/2+y、A=[(x/2,{P0})]；(3) 全复用 L=y、A=[(x,{P0,P1})]，比较三者 TTFT。为保证复用，每实例需持久化连续 KV 段（P0 存前 x/2、P1 存后 x/2）：引入辅助 KV cache buffer（类似 LoongServe），ring attention 时 buffer 转移各实例生成的 KV 分区，前缀缓存引擎按"所需存储区间∩buffer 内容"选择性持久化——因 prefill 逐层推进，buffer 只需单层 KV，额外内存开销可忽略。
涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
