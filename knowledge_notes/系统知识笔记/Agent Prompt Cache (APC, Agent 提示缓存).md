## Agent Prompt Cache (APC, Agent 提示缓存)

术语是什么？
Faster-MoA 中用于桥接前驱 agent 解码输出和后继 agent prefilling 输入的中间存储组件。每个 agent 拥有一个独立的 APC，存储其解码过程中流式产出的部分 text/token chunks。后继 agent 的 Shell Router 周期性 poll 依赖 agent 的 APC，将获取的 chunk 追加到已 prefilled 前缀后，发出增量 /prefill_only 更新。APC 实现了 agent 间数据流的异步解耦：前驱写 → APC → 后继读，使得 decode 和 prefill 的时间重叠成为可能。

从系统架构角度拆解：
APC 数据流：
```
Agent 1 (前驱) 解码:
  while decoding:
    next_token = decode_step()
    output_text += token
    // 写入 APC
    APC(agent_1).append_chunk(token)
    // 支持可配置 chunk size (如每 16 tokens 写一次)

Agent 3 (后继) Shell Router 轮询:
  while slot_1_incomplete:
    chunk = APC(agent_1).poll(offset=consumed)
    if chunk:
      // 追加到 PE 中已有 prefix 之后
      prefill_only(chunk, append_to_existing_kv=True)
      consumed += len(chunk)
    else:
      sleep(poll_interval)  // 前驱尚未产出新 chunk
```

APC 设计要点：
- 同模型家族共享 tokenizer → APC 存 tokens 而非 text，免去 tokenize/detokenize 开销
- chunk size 权衡：太小→过多 /prefill_only 请求（PE 初始化开销）；太大→更多暴露延迟（后继等 chunk 期间无 prefill 进展）
- APC 可以用内存中的 deque/buffer 实现，读写通过轻量锁保护
- 每个 agent 的生命周期内 APC 即建即销

术语一般如何实现？如何使用？
- 内存中的 thread-safe FIFO buffer，支持 append 和 offset-based poll
- 与 Shell Router 紧密耦合：Shell Router 持有所有相关 agent 的 APC 引用
- 也作为完整解码输出的中间存储，在 agent 不再需要增量 prefilling 后仍可读取

涉及论文标题：
- Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap
