## BigMac A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **BigMac**，一种通信高效的 fine-grained MoE 模型结构，核心设计包括：

  **DCCA（descend-communicate-communicate-ascend）低维通信策略**：将 fine-grained MoE 原有的 CDAC（communicate-descend-ascend-communicate）方式改为 DCCA——在每个 MoE 层的入口增加 descending projection（$W'_{\downarrow}$）将 token hidden dimension 从 h 压缩至 r·h，再进行 All-to-All 通信分发 token 到各 expert，expert 计算完成后再通过 ascending projection（$W'_{\uparrow}$）恢复到原始维度 h。通信量从 $C = 2 \times top\_k \times \frac{ep-1}{ep} bsh$ 降至 $C' = 2 \times top\_k \times \frac{ep-1}{ep} bsr h$（减少至原来的 r 倍，r 为 downscaling factor，论文设 r=0.25）。

  **BigMac Expert 设计**：为适配 DCCA 策略，重新设计 expert 结构——将 fine-grained MoE 的 expert（$E_i(x) = \sigma(xW_{i,\downarrow})W_{i,\uparrow}$，先降维再升维）改为 BigMac expert（$E_i(x) = \sigma(xW_{i,\uparrow})W_{i,\downarrow}$，先升维再降维）。由于 DCCA 已将输入 token 维度缩减至 r·h，expert 内部先升维可保证总参数量与 fine-grained MoE 对齐，避免模型质量下降。

  实验比较：
  - **Pre-training convergence**：GPT-Vanilla（conventional MoE）、GPT-Fine-Grained（DeepSeekMoE 式 fine-grained）、GPT-BigMac，在 Wikipedia 3.6B tokens 上预训练，比较 validation perplexity 收敛曲线和 wall-clock time
  - **Downstream tasks（同时长训练后）**：BigMac vs Fine-Grained on PTB, WikiText103, WikiText2, LAMBADA, HellaSwag, WinoGrande, PIQA, RACE-H（基于 GPT3-XL）
  - **Downstream tasks（同 token 数训练后）**：BigMac vs Fine-Grained vs Vanilla on PTB, WikiText103, WikiText2, LAMBADA, HellaSwag, WinoGrande, PIQA, RACE-H（基于 GPT3-Medium）
  - **Long-context evaluation**：GovReport（summarization）、NeedleInAHaystack（retrieval），BigMac vs Fine-Grained
  - **Training latency（Megatron）**：GPT-BigMac vs GPT-Fine-Grained on four base models (GPT3-Medium/XL/2.7B/6.7B)，top-4/top-8 routing，不同 EP/TP 配置下的 step time breakdown
  - **Inference throughput（Megatron）**：GPT-BigMac vs GPT-Fine-Grained on 16/32 GPUs，不同 prompt length (128-1024)，top-4/top-8 routing
  - **Training on Tutel**：with 2DH All-to-All + overlap degree=4，fixed capacity factor f=1.2 vs dynamic capacity factor f=∞
  - **Inference on Tutel & DeepSpeed-Inference**：GPT-BigMac vs GPT-Fine-Grained on Tutel（不同 prompt length）+ DeepSpeed-Inference（不同 generation length 1/2/5/10）

- 硬件平台是什么，配置是什么。
  集群：4 machines connected with 100 Gbps InfiniBand。每 machine 含 8 GPUs，每 GPU 通过 PCIe 4.0 x 16 连接，48 GB HBM，149.7 TFLOPS（FP16），96 cores。训练时并行度：Tensor Parallelism = 4, Expert Parallelism = 4, Data Parallelism = 2（pre-training 阶段）；训练延迟/推理吞吐评估时配置 EP = 1~32, TP = 1~8（ep × tp = 32）。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - **GPT3-Medium**（hidden dim = 1,024, 用于下游任务同 token 数对比和 Tutel/DeepSpeed 评估）
  - **GPT3-XL**（hidden dim = 2,048, 用于 pre-training convergence 和下游任务同时长对比）
  - **GPT3-2.7B** 和 **GPT3-6.7B**（用于 Megatron 训练延迟 scaling 实验）
  - MoE 配置：64 experts/层，top-4/top-8 routing，expert capacity factor = 1.2，load balance type = aux_loss（系数 α=0.001），downscaling factor r = 0.25
  
  数据集：
  - Wikipedia dataset（3.6B tokens，用于 pre-training convergence）
  - OpenWebText2 dataset（14.8B tokens，用于 downstream task 评估）
  
  Benchmarks：
  - Perplexity: PTB, WikiText103, WikiText2 — PPL↓
  - Accuracy: LAMBADA, HellaSwag, WinoGrande, PIQA, RACE-H — ACC↑
  - Long-context: GovReport（summarization score）, NeedleInAHaystack（recall score across depths 10-90%）
  - Efficiency: Training step latency (ms), All-to-All latency (ms), Inference throughput (tokens/s)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：论文未提供开源代码仓库。BigMac 基于 Megatron-LM、Tutel、DeepSpeed-Inference 等开源框架评估，但模型结构和路由策略的修改代码未公开。

  **BigMac 算法 Pipeline 核心计算流程**：

  ```
  # === 符号说明 ===
  # b: global batch size, s: sequence length, h: hidden dimension
  # e: number of experts, top_k: activated experts per token
  # r: downscaling factor (论文 r=0.25, 如 DeepSeek-V2 从 5120→1536)
  # ep: expert parallelism degree

  # === BigMac MoE Layer Forward Pass (DCCA Strategy) ===
  def bigmac_moe_forward(x):                    # x ∈ R^{batch×seq×h}
      # Step 1: Gating (在降维前的 full dimension 做路由)
      gate_logits = x @ W_gate                  # W_gate ∈ R^{h × e}
      gate_probs = SoftMax(gate_logits)         # [batch, seq, e]
      topk_weights, topk_indices = TopK(gate_probs, k=top_k)

      # Step 2: Descend — 降维投影（DCCA 的第一步 D）
      x_low = x @ W'_down                       # W'_down ∈ R^{h × (r·h)}
                                                 # x_low ∈ R^{batch×seq×(r·h)}
      
      # Step 3: All-to-All Dispatch — 低维通信（DCCA 的 C·C）
      # Token dispatch 到各 expert 所在 GPU
      # 通信量: 2 × top_k × (ep-1)/ep × b × s × (r·h)（比 CDAC 减少 r 倍）
      dispatched_tokens = all_to_all_dispatch(x_low, topk_indices)

      # Step 4: Expert Computation（BigMac Expert, 先升后降）
      for each expert i in assigned_experts:
          # BigMac Expert: E_i(x) = σ(x @ W_{i,up}) @ W_{i,down}
          # W_{i,up}: [(r·h) → h_ff], W_{i,down}: [h_ff → (r·h)]
          h_up = tokens @ W_{i,up}              # 先升维：r·h → h_ff
          h_act = σ(h_up)                        # activation (e.g. GeLU/SwiGLU)
          h_out = h_act @ W_{i,down}            # 再降维：h_ff → r·h
          expert_outputs[i] = topk_weights[i] * h_out

      # Step 5: All-to-All Combine — 低维收集
      combined = all_to_all_combine(expert_outputs)

      # Step 6: Ascend — 升维投影（DCCA 的最后一步 A）
      y = combined @ W'_up                      # W'_up ∈ R^{(r·h) × h}
                                                 # y ∈ R^{batch×seq×h}
      return y

  # === Fine-Grained MoE (CDAC) 对比 ===
  def finegrained_moe_forward(x):
      # Step 1: Gating
      gate_logits = x @ W_gate
      gate_probs = SoftMax(gate_logits)
      topk_weights, topk_indices = TopK(gate_probs, k=top_k)

      # Step 2: All-to-All Dispatch — 高维通信（CDAC 的 C）
      # 通信量: 2 × top_k × (ep-1)/ep × b × s × h（在全维度 h 上进行）
      dispatched_tokens = all_to_all_dispatch(x, topk_indices)

      # Step 3: Descend — Expert 内降维（CDAC 的 D）
      for each expert i:
          h_down = dispatched_tokens @ W_{i,down}  # h → h_ff
          h_act = σ(h_down)
          # Step 4: Ascend — Expert 内升维（CDAC 的 A）
          h_out = h_act @ W_{i,up}                   # h_ff → h
          expert_outputs[i] = topk_weights[i] * h_out

      # Step 5: All-to-All Combine — 高维通信（CDAC 的 C）
      combined = all_to_all_combine(expert_outputs)
      return combined
  ```

  **关键张量计算对比**：
  
  | 指标 | GPT-Fine-Grained | GPT-BigMac |
  |------|-----------------|------------|
  | #Param | $4h^2 + 8h + (2rh^2 + 2rh)e$ | $4h^2 + 8h + (2rh^2 + 2rh)e + 2rlh^2$ |
  | #FLOPs | $12bslh^2(2+s/h+v/2lh+rtop\_k)$ | $12bslh^2(2+s/h+v/2lh+rtop\_k) + 12rbslh^2$ |
  | #A2A | $8bslhtop\_k\frac{ep-1}{ep}$ | $8bslhtop\_k\frac{ep-1}{ep}r$ |

  以 GPT3-XL + 64 experts, top_k=8, r=0.25, ep=32 为例：
  - #Param: Fine-Grained 3.73B → BigMac 3.78B (+1.35%)
  - #FLOPs: Fine-Grained 3,490.67 TFLOPs → BigMac 3,649.00 TFLOPs (+4.54%)
  - #A2A: Fine-Grained 1,488.00 GB → BigMac 372.00 GB (-75.00%)

  **额外优势**：
  - **Dropless Token Routing**：通信量大幅减少后，可移除 expert capacity 限制（不再丢 token），进一步提升模型质量
  - **Flexible top_k**：通信高效的 BigMac 可使用更大的 top_k 值以增强模型性能（如 Top8 BigMac 仍快于 Top4 Fine-Grained）
