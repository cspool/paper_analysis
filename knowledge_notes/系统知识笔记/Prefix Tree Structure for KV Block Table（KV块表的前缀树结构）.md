## Prefix Tree Structure for KV Block Table（KV块表的前缀树结构）

术语是什么？通过联网搜索让回答具体和精准。
Prefix Tree Block Table是PAT论文提出的一种将LLM serving framework（如vLLM）中二维block table转换为层次化树结构的数据结构，用于高效识别和利用decode batch内的跨请求共享KV cache prefix。在vLLM中，每个query的KV cache以fixed-size blocks组织为block table（二维数组：行=query，列=block ID序列）。PAT的pack scheduler将block table转换为prefix tree：树的每个internal node对应一段被多个query共享的KV prefix block序列，记录两个属性——共享KV长度l和共享该prefix的query数s；每个leaf对应一个query的完整KV路径（从root到leaf串联为该query的全部KV blocks）。该树结构将共享prefix从扁平的block table中显式化，为memory-centric profit model的packing决策提供结构基础。在ToolAgent和Conversation真实trace中，intra-batch shared prefixes覆盖2.8-82.6%的KV cache，每个batch平均2.72个distinct shared prefixes。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Prefix Tree的构建和使用流程：

```
// ===== Prefix Tree构建 =====
Input: vLLM block_table
  Q1: [KV-0, KV-1, KV-3, KV-6]  // 4个KV blocks
  Q2: [KV-0, KV-1, KV-3, KV-7]  // 前3个blocks与Q1共享
  Q3: [KV-0, KV-2, KV-4]        // 仅第1个block与Q1/Q2共享
  Q4: [KV-0, KV-2, KV-5]        // 前2个blocks与Q3共享

// 构建prefix tree (从block table逐行插入):
// root → KV-0 (s=4, l=block_size)  // 所有4个query共享
//   ├── KV-1 (s=2, l=block_size)   // Q1, Q2共享
//   │   ├── KV-3 (s=2, l=block_size)
//   │   │   ├── KV-6 → leaf Q1
//   │   │   └── KV-7 → leaf Q2
//   └── KV-2 (s=2, l=block_size)   // Q3, Q4共享
//       ├── KV-4 → leaf Q3
//       └── KV-5 → leaf Q4

// node属性示例:
// node(KV-0): l=block_size(如16或32 tokens), s=4
// node(KV-1): l=block_size, s=2
// node(KV-3): l=block_size, s=2
// leaf Q1: non-shared KV = [KV-6]

// ===== Lazy Update机制 =====
// Pack scheduler在CPU异步线程运行:
struct SchedulingCache:
  last_block_table_hash: uint64
  cached_CTA_partition: List[CTA]

On each decode step:
  current_hash = hash(block_table)
  if current_hash == cache.last_block_table_hash:
    return cache.cached_CTA_partition  // 复用上次结果
  else:
    prefix_tree = BuildPrefixTree(block_table)
    new_partition = TreeHeuristic(prefix_tree)
    cache.update(current_hash, new_partition)
    return new_partition
  // 调度结果通常在block table变化时才需要更新:
  // request arrive/depart 或 新KV block allocation
```

在系统架构中的定位：prefix tree是连接vLLM continuous batching（request-level scheduling）和PAT kernel execution（CTA-level execution）的中间数据结构。它把上层serving framework的paged KV cache抽象转换为kernel调度可直接使用的共享prefix结构，同时保持与vLLM KV paging机制的完全兼容（pack scheduler仅操作logical block IDs，不修改physical KV blocks）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PAT在C++中实现了prefix tree和pack scheduler。构建方式是遍历block table的每行（每个query的block ID序列），将公共前缀的leading block IDs逐步插入树中——相同block ID路径共享internal node。实际使用中，prefix tree的构建和维护对用户透明：用户仅需设置`VLLM_ATTENTION_BACKEND=PAT`，pack scheduler自动在每次decode step前从vLLM获取block table并构建tree。Lazy update大幅降低调度开销：在block table不频繁变化的serving场景（如长输出阶段），调度结果的复用率很高；在ToolAgent和Conversation trace测试中，pack scheduler的平均latency比pre-attention task latency低42.3%-49.6%，异步执行时不增加端到端延迟。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel
