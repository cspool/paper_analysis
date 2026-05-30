## Fast Inference of Mixture-of-Experts Language Models with Offloading

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文提出三种针对 MoE 模型的推理加速策略：
    1. **Expert LRU Cache**：利用相邻 token 间 expert 复用的局部性（locality），在 GPU 显存中为每个 MoE 层缓存 k 个最近使用的 expert（LRU 策略）。当后续 token 复用同一 expert 时可即时获取。Mixtral-8x7B 用 k=2（12GB GPU）或 k=4（16GB GPU）。
    2. **Speculative Expert Loading（投机 expert 预加载）**：利用 Transformer 残差连接带来的归纳偏置——当前层的 hidden states 可作为下一层 hidden states 的近似估计。将下一层 MoE gate 函数应用于当前层 hidden states，预测下一层最可能被激活的 1-2 个 expert，在当前层计算期间在后台预取这些 expert 权重到 GPU。
    3. **混合量化（Mixed MoE Quantization）**：使用 HQQ（Half Quadratic Quantization）对 attention 层保持 4-bit，expert 层量化到 2-3 bit，获得最优尺寸-质量权衡。
  - 实验比较：
    - 4.1 节：LRU cache 命中率 vs cache 大小 k，speculative loading recall vs 预取 expert 数量（OpenAssistant 数据集，Mixtral-8x7B-Instruct）
    - 4.2 节：不同量化方案下 Mixtral-8x7B 的 WikiText2/C4 perplexity 和 MMLU 准确率
    - 4.3 节：完整系统在 T4/RTX 3060/RTX 3080 Mobile/A100 上的 tokens/sec，消融 LRU cache 和 pre-loading 的效果

- 硬件平台是什么，配置是什么。
  - T4 (free-tier Google Colab): 16GB VRAM, PCIe Gen.3
  - RTX 3080 Mobile (gaming laptop): 16GB, PCIe Gen.4
  - RTX 3060 (midrange desktop): 12GB, PCIe Gen.3
  - A100-80GB-SXM (data-center server, 用于参考对比)
  - 目标场景：足够系统内存容纳模型参数（量化后），GPU 显存仅能容纳 non-expert 层 + k 个缓存 expert

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral-8x7B（46.7B 总参数，expert 占 45.1B/96.6%），Mixtral-8x7B-Instruct
  - 数据集/benchmark：
    - OpenAssistant（对话生成，测量 tokens/sec）
    - WikiText2 perplexity（语言建模）
    - C4 perplexity（语言建模）
    - MMLU 5-shot accuracy（多任务语言理解）
    - 推理评测使用 batch size 1，按预测概率采样（无 temperature/nucleus sampling）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：代码开源在 https://github.com/dvmazur/mixtral-offloading
  - **算法 pipeline 解释**：

  **LRU Expert Cache 策略（Section 3.1）**：
  对于 Mixtral-8x7B 的每个 MoE 层（共 32 层），每层有 8 个 expert。每个 token 由 gate 选择 top-2 expert 参与计算。LRU cache 维护每层 k 个最近使用过的 expert 在 GPU 显存中：

  ```
  # 对于每个 MoE 层 l, 维护 LRU cache C_l (max size = k)
  # 初始化：C_l = 空（或随机 k 个 expert）
  
  for each token t:
      for each MoE layer l:
          gate_scores = W_gate[l] @ hidden_states[l]  # (num_experts,)
          top2_indices = topk(gate_scores, k=2)        # 选 top-2 experts
          
          for expert_id in top2_indices:
              if expert_id in C_l:
                  # cache hit: expert 已在 GPU，直接使用
                  expert_weights = GPU_cache[expert_id]
                  # 将该 expert 标记为 most recently used
              else:
                  # cache miss: 从 host RAM 加载 expert 到 GPU
                  expert_weights = load_from_host(l, expert_id)
                  if len(C_l) >= k:
                      evicted = C_l.evict_lru()  # 淘汰最久未用的 expert
                  C_l.add(expert_id)
                  GPU_cache[expert_id] = expert_weights
          
          # Expert computation (仅 top-2)
          hidden_states[l+1] = hidden_states[l]
          for expert_id in top2_indices:
              weight = gate_scores[expert_id] / sum(gate_scores[top2_indices])
              hidden_states[l+1] += weight * expert_ffn(expert_id, hidden_states[l])
  ```

  **Speculative Expert Loading（Section 3.2）**：
  利用残差连接的归纳偏置，用当前层的 hidden states 预测下一层的 gate 选择：

  ```
  # 在处理 MoE 层 l 时，同时预测并预取层 l+1 的 expert
  # 当前层 hidden states: h_l (pre-MoE gate input)
  
  # Step 1: 当前层 gate（正常执行）
  current_gate_scores = W_gate[l] @ h_l
  current_top2 = topk(current_gate_scores, k=2)
  
  # Step 2: 投机预测下一层 gate
  # 利用 h_l 近似 h_{l+1}（残差连接的归纳偏置）
  predicted_gate_scores = W_gate[l+1] @ h_l  # 用当前激活值运行下一层的 gate
  predicted_top2 = topk(predicted_gate_scores, k=2)
  
  # Step 3: 在 CUDA stream 上异步预取预测的 expert
  async_load(l+1, predicted_top2[0])  # 后台加载
  async_load(l+1, predicted_top2[1])  # 后台加载
  
  # Step 4: 继续当前层 expert 计算
  hidden = expert_compute(current_top2, h_l)
  
  # 当进入下一层时，投机加载的 expert 可能已就绪
  # 如果预测正确 → 即时可用；如果错误 → 重新加载正确 expert
  ```

  **系统内存管理（Section 3.3）**：
  - Expert 参数在连续内存 buffer 中分配，单次 host-to-device copy 完成传输
  - Host 侧使用 pinned memory（tensor.pin_memory()）加速传输
  - 分配 b=4 个临时 device buffer 用于异步拷贝/预取，所有 MoE 层共享
  - 总内存 = num_layers × num_experts 个 expert buffer（split 在 host/device 间）+ b=4 临时 buffer
  - 混合量化方案：attention 层 4-bit HQQ（group size 64, scale group size 256），expert 层 2-bit（group size 16, scale group size 128）或 3-bit（group size 64, scale group size 128）
