## MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-Inference-Bench 是一个综合性 MoE 推理 benchmark 套件，不提出新的算法，而是系统性地评估多种已有算法优化技术在 MoE 模型上的推理性能。评估的算法优化包括：
    1. **FP8 量化**：将 Mixtral-8x7B 从 FP16 量化到 FP8 精度（使用 GPTQ/AWQ 方法），对比不同 batch size 和 sequence length 下的吞吐量。
    2. **Intra-expert 剪枝**：在每个 expert 内部减少 FFN dimension（12.5%/25%/50%），保持 expert 数量不变但降低每个 expert 的计算量。
    3. **Inter-expert 剪枝**：移除整个 expert 及其路由权重（12.5%/25%/50%），保持相同数量的 active experts 但减少总 expert 数量以降低显存占用。
    4. **投机解码（Speculative Decoding）**：使用 Qwen3 系列的小型 draft model（0.6B/1.7B/4B/8B）为 Qwen3-30B-A3B target model 生成候选 token，通过验证-接受机制加速解码。
  - 实验比较：(1) FP16 vs FP8 在不同 batch size（1/16/32/64）和 input/output length（128/256/512/1024/2048）下的吞吐量；(2) OLMoE-1B-7B 和 Qwen1.5-MoE-A2.7B 在不同剪枝比例（12.5%/25%/50%）和 TopK（1 到 baseline）下的吞吐量变化；(3) Qwen3-30B-A3B 搭配四种不同大小 draft model 在不同 input length 和 draft token count 下的投机解码吞吐量；(4) 六种 LLM（Mixtral-8x7B, DeepSeek-V2-Lite, Phi-3.5-MoE, OLMoE-1B-7B, Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B）在九个 lm-eval 任务上的准确率 vs 吞吐量/延迟 trade-off；(5) 三种 DeepSeek-VL2 模型（Tiny/Small/Base）在八个 VLMEvalKit 任务上的准确率 vs 吞吐量/延迟 trade-off。

- 硬件平台是什么，配置是什么。
  - 主要平台：NVIDIA H100 SXM5 80GB GPU（基于 TSMC 4N 工艺，80B 晶体管，80GB HBM3，50MB L2 cache，第四代 Tensor Cores，NVLink）
  - 多 GPU 实验：4× H100 GPUs（用于超参数 scaling 分析、剪枝实验、并行策略评估、Fused MoE 实验）
  - 对比平台：Cerebras CS-3 cloud inference system（WSE-3 wafer-scale engine，FP8 weight storage + FP16 computation）
  - 推理框架：vLLM（所有实验统一使用）

- 模型是什么。数据集和bench分别是什么。
  - LLM 模型（7种）：Mixtral-8x7B（47B total/12.9B active）、Qwen-1.5-MoE-A2.7B（14.3B total/2.7B active）、Qwen3-30B-A3B（30B total/5B active）、DeepSeek-V2-Lite（15.7B total/2.4B active）、Phi-3.5-MoE（41B total/9B active）、OLMoE-1B-7B（7.2B total/1.3B active）、Llama-4-Scout-17B-16E
  - VLM 模型（3种）：DeepSeek-VL2-Tiny（3B total/1B active）、DeepSeek-VL2-Small（16B total/2.8B active）、DeepSeek-VL2（27B total/4.5B active）
  - LLM Benchmark：lm-eval suite — ARC-c, ARC-e, BoolQ, HellaSwag, MMLU, OpenBookQA, RTE, WinoGrande
  - VLM Benchmark：VLMEvalKit — MME, TextVQA, AI2D, DocVQA, MMMU, InfoVQA, RealWorldQA, ScienceQA
  - 推理性能评估：自定义脚本基于 vLLM，通过限制 max output length=1 测量 TTFT，计算 ITL 和 throughput

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未明确说明 benchmark 代码是否开源。所有评估基于开源框架 vLLM（https://github.com/vllm-project/vllm）和开源模型（Mixtral、DeepSeek、Qwen、Phi、OLMoE 均可在 HuggingFace 获取）。
  - 算法 pipeline 示例 — FP8 量化 MoE 推理路径（以 Mixtral-8x7B on H100 + vLLM 为例）：
    ```
    # MoE Layer with FP8 Quantization (pseudocode)
    # Input: hidden_states [batch_size, seq_len, hidden_dim=4096]
    
    # Step 1: Router (kept in FP16 for accuracy)
    router_logits = hidden_states @ W_gate_fp16  # [B, S, num_experts=8]
    topk_weights, topk_indices = topk(softmax(router_logits), k=2)
    
    # Step 2: Expert FFN computation (FP8 quantized weights)
    for expert_id in range(8):
        token_mask = (topk_indices == expert_id)  # tokens routed to this expert
        if token_mask.sum() == 0: continue
        
        expert_input = hidden_states[token_mask]  # [num_tokens, 4096]
        
        # FP8 weight dequantization + INT8 matmul on Tensor Core
        # W_gate_fp8: [4096, 14336] stored as FP8, dequantized on-the-fly
        gate_out = fp8_matmul(expert_input, W_gate_fp8[expert_id])  # [n, 14336]
        gate_out = silu(gate_out)
        
        # W_up_fp8, W_down_fp8 FP8 matmul
        up_out = fp8_matmul(expert_input, W_up_fp8[expert_id])  # [n, 14336]
        expert_out = gate_out * up_out  # element-wise
        expert_out = fp8_matmul(expert_out, W_down_fp8[expert_id])  # [n, 4096]
        
        # Accumulate weighted output
        weight = topk_weights[token_mask].unsqueeze(-1)
        output[token_mask] += weight * expert_out
    
    # Step 3: Combine with residual
    final_output = output + residual
    ```
  - 投机解码 pipeline（Qwen3 系列为例）：
    ```
    # Target: Qwen3-30B-A3B, Draft: Qwen3-1.7B (shared vocabulary)
    
    for each decoding step:
        # Phase 1: Draft model generates k candidate tokens
        draft_tokens = []
        draft_kv_cache = copy(target_kv_cache)
        for i in range(num_draft_tokens):
            draft_logits = draft_model.forward(current_token, draft_kv_cache)
            next_token = sample(draft_logits)
            draft_tokens.append(next_token)
        
        # Phase 2: Target model verifies all draft tokens in parallel
        target_logits = target_model.forward([current_token] + draft_tokens, target_kv_cache)
        
        # Phase 3: Accept/reject with speculative sampling
        accepted_tokens = []
        for i, draft_token in enumerate(draft_tokens):
            p_draft = draft_model_probs[i][draft_token]
            p_target = softmax(target_logits[i])[draft_token]
            if random() < min(1, p_target / p_draft):
                accepted_tokens.append(draft_token)
            else:
                # Reject, sample from adjusted target distribution
                corrected_token = sample(max(0, p_target - p_draft))
                accepted_tokens.append(corrected_token)
                break
        
        # Output accepted tokens and advance
    ```
