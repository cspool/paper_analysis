## Making MoE-based LLM Inference Resilient with Tarragon

- 属于算法pipeline的实现是什么？实验比较什么？
  TARRAGON 提出了一系列算法层面的故障恢复机制，核心包括：
  1. **异步增量 KV Cache Checkpointing**：AW 在每层 attention 计算完成后，利用 AW-EW 通信间隙（link idle 时段），通过 one-sided RDMA write 将新增的 KV cache segment（每 token 每层一个小 segment）异步写入 checkpoint store。使用 "async log + commit record" 设计保证顺序（基于单调递增的 RDMA work request ID 作为 sequence number），避免干扰正常 AW-EW 流量。
  2. **Per-Request KV Cache Restoration**：故障时仅恢复受影响请求的 KV cache。Checkpoint store 通过 GPUDirect one-sided RDMA write 将 KV cache segments 直接注入替代 AW 的 GPU 显存，替代 AW 从 committed token 继续 decoding，无需重放 prefill/decoding。
  3. **AW 侧自愈算法（EW 故障容忍）**：AW 对 EW 响应设置超时；超时后 REFE 立即将请求重路由到替代 EW（健康 primary 或 shadow expert），重播相同 token embeddings + metadata。因 expert 计算是 stateless 和 deterministic 的，重播产生相同结果。
  4. **EW 侧自愈算法（AW 故障容忍）**：EW 不再等待所有 AW 的输入。当收到"足够数量"健康 AW 的输入（或 batch 达到配置的最小大小）时即开始 expert 计算，省略未响应 AW 的 slots。
  5. **Shadow Experts**：在 EW GPU 显存中预加载 expert 权重的 inactive 副本，primary 故障时可立即激活，避免从存储重新加载权重（数百毫秒到秒级延迟）。
  实验比较了：
  - 不同 checkpointing 方案（No-CKPT / Pause-Checkpoint-Resume / TARRAGON incremental）的吞吐量开销
  - 不同 AW 恢复策略（Sequential replay / Parallel replay / TARRAGON per-request restoration）在 restoration time、transfer data volume、GPU recomputation cost 三个维度上的表现
  - 不同 failure point（已 decoding token 数量）下的恢复代价

- 硬件平台是什么，配置是什么。
  GCP A3 Ultra 节点，每节点 8x NVIDIA H200 GPUs (141 GB 显存), 8x 400 Gbps ConnectX-7 RDMA NICs (GPUDirect RDMA), NVLink 3.6 Tbps。3 节点：AWs 1 节点 + EWs 1 节点 + Checkpoint store 1 节点。Ubuntu 22.04, Linux 5.15, CUDA 12.8, PyTorch 2.6.0。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral-8×7B（32 层 MoE transformer, 8 experts/layer, top-2 expert selection），MoE 层 hidden_size=4096。
  - 数据集/Workload：
    - ShareGPT：自然变化的 prompt 长度，测试 prefill 和 decode 的真实请求异构性
    - Random（synthetic）：固定 10 input tokens + 128 generated tokens，强调 decoding 阶段
  - 请求到达：Poisson 过程，varied rates (30-70 RPS)
  - 评估指标：TTFT (Time-to-First-Token), TBT (Time-Between-Tokens, median + P95), Output-token throughput, T_stall (failure-induced stall time), GPU recomputation cost (GPU-time)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文声明将开源（"We will open-source TARRAGON"），截止论文阅读时尚未公开链接。
  
  **KV Cache Checkpointing 算法流程**：
  
  ```
  输入: AW 上每层 attention 完成后新产生的 KV cache segment
        segment 大小 C = 2 * H_kv * (N_hidden_size / H_attn) * S_elem
        对于 Mixtral-8×7B (GQA): C ≈ 12.5% of expert traffic V
        V = 2 * Top_k * N_hidden_size * S_elem
  
  // 初始化
  AW 分配连续 GPU KV cache region，通过 RDMA 注册
  Checkpoint store 分配对应 bucket，返回 base address
  seq_num = 0  // 单调递增的 RDMA work request ID
  
  // 增量更新（每 token 每层）
  on_attention_done(layer_l, token_t):
      segment = KV_cache[token_t][layer_l]  // 刚写入的 segment
      // 等待 AW-EW link idle（opportunistic interleaving）
      wait_until_aw_ew_link_idle()
      // One-sided RDMA write，不涉及 receiver CPU
      rdma_write(
          src = segment.gpu_addr,
          dst = checkpoint_store.bucket_base + offset,
          size = C,
          wr_id = seq_num  // sequence number 保证顺序
      )
      seq_num += 1
      // 写入 commit record（标记此 segment 已持久化）
      rdma_write_commit_record(token_t, layer_l)
  ```
  
  **Per-Request KV Cache Restoration 算法流程**：
  
  ```
  输入: failed_aw_id, 该 AW 上的活跃请求列表 requests[]
  输出: 恢复后的请求在替代 AW 上继续 decoding
  
  on_aw_failure(failed_aw_id):
      // 1. Orchestrator 识别 failed AW 上所有活跃请求
      for each request r in failed_aw.active_requests:
          committed_token = checkpoint_store.get_latest_commit(r)
          // committed_token: 最后一个已 checkpoint 的 token index
  
      // 2. 负载均衡分发到健康 AWs
      for each request r in round_robin:
          alt_aw = select_healthy_aw()
          assign(r, alt_aw)
  
      // 3. Per-request 恢复（并行执行）
      for each (r, alt_aw):
          // Step a: Checkpoint store 通知 alt_aw 恢复所需信息
          checkpoint_store → alt_aw: committed_token_id, kv_state_size
          
          // Step b: alt_aw 分配 KV cache 区域
          kv_region = alt_aw.allocate_kv_cache_region(kv_state_size)
          alt_aw → checkpoint_store: kv_region.offset
          
          // Step c: Checkpoint store 通过 GPUDirect RDMA 注入 KV cache
          for each layer l in {1..L}:
              for each segment s in {1..committed_token}:
                  rdma_write(
                      src = checkpoint_store.bucket[r][l][s],
                      dst = alt_aw.gpu_mem[kv_region.offset + l * segment_stride + s],
                      size = C
                  )
          
          // Step d: 确认完成，resume decoding
          checkpoint_store → alt_aw: HTTP restore_complete(r)
          alt_aw.resume_decoding(r, from_token=committed_token + 1)
  ```
  
  **AW 侧自愈（EW 故障）**：
  
  ```
  on_ew_timeout(ew_id, request_metadata, token_embeddings):
      // REFE 探测到 EW 无响应
      alt_ew = ERT.lookup_alternative(expert_id)
      if alt_ew is None:
          // 激活 shadow expert（已预加载在 GPU 显存中）
          alt_ew = ERT.activate_shadow(expert_id)
      // 重播请求到替代 EW（带优先级标记，避免 straggler）
      rdma_write_prioritized(alt_ew, metadata, token_embeddings)
  ```
  
  **EW 侧自愈（AW 故障）**：
  
  ```
  on_expert_batch_ready(layer_l, expert_e):
      // EW 收集来自各 AW 的 tokens
      received = buffer[expert_e][layer_l].tokens
      healthy_aws = liveness_probe_all_aws()
      if len(received) >= min(threshold, len(healthy_aws)):
          // 开始计算，不等所有 AW
          outputs = expert_ffn_forward(received)
          // 返回结果给各自的 AW
          for each aw in received.sources:
              rdma_write(aw, outputs[aw])
  ```
  
  **Shadow Expert 机制**：
  
  ```
  // 初始化：在 EW GPU 显存中预加载 inactive expert 副本
  for each expert e in primary_experts:
      shadow_e = load_weights(e.weights)  // 仅占 GPU 显存，不消耗 compute
      // 对于 DeepSeek-R1: 单个 expert 约 2.5 GB
      // 多个 active + shadow experts 可舒适放入 A100/H200 40-141 GB 显存
  
  // 故障时激活
  on_primary_ew_failure(failed_ew):
      for each expert e in failed_ew.experts:
          shadow = find_shadow_replica(e)
          shadow.activate()  // 开始接受请求
          ERT.update(e.id → shadow.physical_location)
  ```

  关键参数：
  - KV cache segment 大小 C：对于 Mixtral-8×7B (GQA)，仅为 expert traffic V 的 ~12.5%
  - Failure detection probing interval: 10 ms
  - 连续超时阈值: 3（RDMA QP 级别配置）
  - Shadow expert: inactive 时不消耗 compute，仅占 GPU 显存
