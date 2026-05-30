## SnapKV: LLM Knows What You are Looking for Before Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  SnapKV 是一种无需微调的 KV cache 压缩算法，通过 prompt 末尾的 "observation window" 计算注意力权重，对 prefix 中的关键 KV 位置进行投票和聚类选择，从而在生成阶段使用恒定大小的压缩 KV cache。实验比较：(a) SnapKV 压缩 vs 全量 KV cache（Full KV）在 LongBench 16 个数据集上的准确率；(b) SnapKV vs H2O（Heavy-Hitter Oracle）在 LongBench 上的准确率对比；(c) Needle-in-a-Haystack 压力测试（最长 380K tokens）；(d) 解码延迟与 batch size/序列长度 scaling；(e) 消融实验——pooling 对 LongEval-Lines 检索准确率的影响；(f) Command-R 上的 RAG 任务（citation、generation、end-to-end）；(g) SnapKV + Medusa 并行解码的兼容性实验。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA A100-80GB GPU。解码速度和内存实验均在此平台上完成，HuggingFace 原生实现配合少量代码修改即可运行。

- 模型是什么。数据集和bench分别是什么。
  模型：LWM-Text-Chat-1M（1M 上下文）、LongChat-7b-v1.5-32k、Mistral-7B-Instruct-v0.2、Mixtral-8x7B-Instruct-v0.1（32K 上下文）、Command-R（35B，128K 上下文）。
  数据集/Benchmark：LongBench（16 个子任务：MultiFieldQA-en、Qasper、HotpotQA、2WikiMQA、Musique、GovReport、QMSum、MultiNews、TREC、TriviaQA、SAMSum、PassageCount、PassageRetrieval-en、RepoBench-P、LCC、PREC）、Needle-in-a-Haystack（扩展至 380K tokens）、LongEval-Lines、NarrativeQA、bioasq、HotpotQA（RAG）、QASPER。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/FasterDecoding/SnapKV。安装后 monkey-patch 模型即可使用：`from snapkv.monkeypatch.monkeypatch import replace_mistral; replace_mistral()`。已集成到 KVCache-Factory（https://github.com/Zefan-Cai/KVCache-Factory），vLLM 有相关 PR。

  **算法 pipeline 伪代码（基于论文 Listing 1）**：

  ```
  输入: query_states (B, H, L_prompt, D), key_states, value_states,
        window_size (L_obs), max_capacity_prompt, kernel_size
  输出: 压缩后的 key_states, value_states

  def snap_kv(query_states, key_states, value_states, window_size,
              max_capacity_prompt, kernel_size):
      # 仅 prompt 阶段执行压缩
      if q_len < max_capacity_prompt:
          return key_states, value_states

      # === Phase 1: Vote for important prefix positions ===
      # 计算 observation window 内 queries 对 prefix keys 的注意力权重
      # attn_weights: (B, H, L_obs, L_prefix) 其中 L_prefix = L_prompt - L_obs
      attn_weights = compute_attn(
          query_states[..., -window_size:, :],  # Q_obs: (B, H, L_obs, D)
          key_states,                            # K_full: (B, H, L_prompt, D)
          attention_mask
      )

      # 沿 query 维度求和得到每个 prefix 位置的累积注意力分数
      # vote: (B, H, L_prefix) — 每个 head 对每个 prefix token 的重要性投票
      vote = attn_weights[..., -window_size:, :-window_size].sum(dim=-2)

      # 1D 池化实现聚类——保留高注意力 token 周围上下文
      pool_vote = pool1d(vote, kernel_size=kernel_size,
                         padding=kernel_size//2, stride=1)

      # TopK 选择：选出 max_capacity_prompt - window_size 个最重要的 prefix 位置
      k = max_capacity_prompt - window_size
      indices = pool_vote.topk(k, dim=-1).indices  # (B, H, k)

      # === Phase 2: Compress and store ===
      # 扩展 indices 匹配 head_dim
      indices = indices.unsqueeze(-1).expand(-1, -1, -1, D)

      # 按 indices 收集压缩后的 prefix KV
      k_past_compress = key_states[..., :-window_size, :].gather(dim=2, index=indices)
      v_past_compress = value_states[..., :-window_size, :].gather(dim=2, index=indices)

      # 保留完整 observation window 的 KV（不做压缩）
      k_obs = key_states[..., -window_size:, :]
      v_obs = value_states[..., -window_size:, :]

      # 拼接: 压缩的 prefix KV + 完整的 observation window KV
      key_states = torch.cat([k_past_compress, k_obs], dim=2)
      value_states = torch.cat([v_past_compress, v_obs], dim=2)

      return key_states, value_states
  ```

  **核心张量计算流程**：
  1. Prefill 阶段：将完整 prompt 输入模型，在每层计算 QKV 投影
  2. 对于每层 attention，取 Q 的最后 `window_size` 个 token（observation window），计算它们对所有 K 的注意力权重 `W_obs ∈ R^{H × L_obs × L_prefix}`
  3. 沿 query 维度求和对每个 prefix 位置投票：`C_h = Σ_i W_obs[h, i, :]`，得到每个 head 的重要性向量 `C ∈ R^{H × L_prefix}`
  4. 1D max/avg pooling 平滑邻域（kernel_size=5~13，依模型调整）→ `pool_vote`
  5. TopK 选择保留 `max_capacity_prompt - window_size` 个位置 → 成压缩后的 prefix KV
  6. 拼接 observation window 的完整 KV → 形成最终压缩 KV cache
  7. 生成阶段：仅使用压缩后的 KV cache 进行 attention 计算，KV cache 大小恒定不变

  **关键超参数**：
  - `window_size`（observation window 大小）：Mistral 用 32，Command-R 用 64，LWM 用 16
  - `kernel_size`（pooling kernel）：Mistral 用 7，Command-R 用 13，LWM 用 5
  - `max_capacity_prompt`（压缩后 KV 大小）：1024/2048/4096
  - 压缩比：平均 input 13K tokens，1024 prompt KV → 92% 压缩率；4096 → 68% 压缩率
  - 极端压力测试：380K tokens → 1024 prompt KV → 380× 压缩比
