## Marconi: Prefix Caching for the Era of Hybrid LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Marconi prefix caching系统中两个核心算法：(1) Taxonomy-based Judicious Admission算法——通过radix tree bookkeeping将前缀复用模式分类为Purely Input（系统提示词等被多请求共享）和Input+Output（对话历史续写），对每种模式采用不同缓存策略，每序列至多2个SSM state checkpoint；(2) FLOP-Aware Eviction算法——定义FLOP Efficiency = Total FLOPs saved / Memory consumption of cached states，Utility Score = recency + α × flop_efficiency，替代传统LRU-only eviction。α参数由config_tuner根据workload自动调优。

  实验比较算法baseline：(1) fine-grained checkpointing（naive admission，每x token存checkpoint，使用LRU eviction）；(2) SGLang+ LRU eviction（recency-only eviction）。评估指标：token hit rate (%)。从algorithms角度，比较的是admission policy的精准度（judicious vs naive）和eviction policy的计算感知能力（FLOP-aware vs recency-only）。

- 硬件平台是什么，配置是什么。
  论文实验为离线trace-based模拟评估，运行于Cloudlab节点（Ubuntu 22.04, 32-core CPU）。算法本身与具体GPU硬件解耦——admission/eviction策略通过radix tree操作实现，不依赖特定GPU kernel。

- 模型是什么。数据集和bench分别是什么。
  模型：NVIDIA Mamba2-Hybrid-7B，层结构为4 Attention + 24 SSM + 28 MLP layers。Tokenizer: meta-llama/Llama-2-7b-hf。实验也对不同SSM-to-Attention比例进行了sweep（如Jamba等架构）。数据集/workloads：LMSys-Chat-1M（conversational）、ShareGPT_Vicuna_unfiltered（conversational）、SWEBench（agentic，长上下文代码agent轨迹）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/ruipeterpan/marconi。

  Algorithm 1: Judicious Admission（基于radix tree的前缀复用模式分类）
  ```
  // Radix tree node structure
  class RadixNode:
    token_sequence: List[int]        // 从根到该节点的token路径
    kv_cache: List[Tensor]           // Attention层KV cache (per-token)
    ssm_states: Optional[List[Tensor]] // SSM层recurrent states (仅特定节点)
    reuse_type: Enum["purely_input", "input_output", "none"]
    children: Dict[int, RadixNode]

  // Admission on new request arrival
  def admit(request_tokens: List[int], radix_tree: RadixNode):
      node = radix_tree.root
      match_len = 0
      // Step 1: Traverse radix tree to find longest prefix match
      for i, token in enumerate(request_tokens):
          if token in node.children:
              node = node.children[token]
              match_len += 1
          else:
              break

      // Step 2: Speculative insertion - check if new branching point
      remaining = request_tokens[match_len:]
      curr = node
      for token in remaining:
          if token not in curr.children:
              curr.children[token] = RadixNode(token_sequence=...)
          curr = curr.children[token]

      // Step 3: Classify reuse pattern
      // Purely input: intermediate node shared by multiple requests
      if curr has multiple descendant leaf paths:
          curr.reuse_type = "purely_input"
          cache_all_states(curr)  // admit KV + SSM states

      // Input+output: leaf node (end of sequence)
      if curr is leaf:
          curr.reuse_type = "input_output"
          cache_final_ssm_state_only(curr)  // only last token's SSM state
  ```

  Algorithm 2: FLOP-Aware Eviction
  ```
  // Compute FLOP efficiency for a cached entry
  def flop_efficiency(entry: RadixNode, model_config: ModelConfig):
      // Total FLOPs saved = sum of FLOPs for all layers covering this prefix
      total_flops = 0
      for layer in model_config.layers:
          if layer.type == "attention":
              // Attention FLOPs: O(L^2 * d) for prefill
              total_flops += layer.attention_flops(entry.prefix_len)
          elif layer.type == "ssm":
              // SSM FLOPs: O(L * d_state * d_model)
              total_flops += layer.ssm_flops(entry.prefix_len)
          elif layer.type == "mlp":
              total_flops += layer.mlp_flops(entry.prefix_len)

      // Memory consumed by this cache entry
      memory_bytes = 0
      for layer in model_config.layers:
          if layer.type == "attention":
              // KV cache: 2 * L * d_head * num_heads * sizeof(fp16)
              memory_bytes += 2 * entry.prefix_len * layer.d_head * layer.num_heads * 2
          elif layer.type == "ssm":
              // SSM state: fixed size, d_state * d_model * sizeof(fp32)
              memory_bytes += layer.d_state * layer.d_model * 4

      return total_flops / memory_bytes  // FLOPs per byte

  // Eviction decision
  def evict(cache: Dict, α: float):
      scores = []
      for entry_id, entry in cache.items():
          recency = current_time - entry.last_access_time
          flop_eff = flop_efficiency(entry, model_config)
          // Utility = recency + α × flop_efficiency
          utility = recency + α * flop_eff
          scores.append((entry_id, utility))

      // Evict entry with lowest utility
      victim = argmin(scores, key=lambda x: x[1])
      cache.remove(victim)
  ```

  关键设计决策：
  - SSM state大小固定（与序列长度无关），而KV cache大小随序列长度线性增长——因此长前缀的FLOP efficiency更高，Marconi优先保留
  - α参数自动调优：config_tuner.py根据workload命中率反馈动态调整
  - 每个序列最多2个SSM state checkpoint：避免naive checkpointing的稀疏命中问题
  - 统一radix tree管理KV+SSM状态：因为所有层的state必须代表同一前缀才能复用
