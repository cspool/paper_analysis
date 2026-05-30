## MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-Inference-Bench 不修改 vLLM 源码，而是将其作为统一推理后端，系统性地评估多种 serving 级配置和优化策略在 MoE 模型上的效果。评估的 serving 级技术包括：
    1. **GPU 并行策略（TP/PP/EP）**：在 vLLM 中配置 Tensor Parallelism（张量并行，按行/列分布权重张量）、Pipeline Parallelism（流水线并行，按层分配）、Expert Parallelism（专家并行，按 expert 分配设备）及 Hybrid Parallelism（TP+PP+EP 混合）。
    2. **Fused MoE**：vLLM 内置的融合 MoE kernel，将 expert 选择、路由和 FFN 计算融合为单个 GPU kernel，减少中间显存传输和 kernel launch 开销。
  - 实验比较：(1) Mixtral-8x7B 和 OLMoE-1B-7B 在 1-4 GPU 上使用 TP-only、TP+EP、PP+EP、PP-only 的吞吐量对比；(2) Mixtral-8x7B 有/无 Fused MoE 在不同 batch size（1/16/32/64）和 input/output length（128/256/512/1024/2048）下的吞吐量对比；(3) Llama-4-Scout-17B-16E 在 H100 vs Cerebras CS-3 上的延迟和吞吐量硬件对比。

- 硬件平台是什么，配置是什么。
  - 主要平台：NVIDIA H100 SXM5 80GB GPU（基于 TSMC 4N 工艺，80GB HBM3，50MB L2 cache，第四代 Tensor Cores，NVLink）
  - 多 GPU：1-4× H100 GPUs（用于 TP/PP/EP 并行策略 scaling 实验），节点内通过 NVLink 高带宽互联
  - 对比平台：Cerebras CS-3 cloud inference system（WSE-3 wafer-scale engine，多数量级更高的内存带宽，减少 inter-device 通信；FP8 weight storage + FP16 computation）
  - 推理框架：vLLM

- 开源Serving框架是什么。修改了什么。
  - 开源框架：vLLM（https://github.com/vllm-project/vllm），论文未修改 vLLM 源码，直接使用其内置的并行策略配置和 Fused MoE kernel。
  - 评估的 vLLM 配置：
    - **Tensor Parallelism (TP)**：`tensor_parallel_size` 参数控制，按行/列分布层权重张量到多设备。vLLM 基于 Megatron-LM 风格的 TP 实现，在 attention 和 FFN 层均支持张量切分。设备间通过 NCCL all-reduce/all-gather 通信。
    - **Pipeline Parallelism (PP)**：`pipeline_parallel_size` 参数控制，按层分配模型到不同设备。vLLM 通过 `Ray` 或 `multiprocessing` 管理各 pipeline stage 间的 micro-batch 调度。
    - **Expert Parallelism (EP)**：`enable_expert_parallel` 参数控制，将 MoE layer 中的 expert 分配到不同设备，各设备激活其持有的 expert 子集。vLLM 通过 all-to-all dispatch/combine 通信收集/分发 tokens。
    - **Fused MoE**：vLLM 内置的 FusedMoE kernel（`vllm.model_executor.layers.fused_moe`），使用 Triton/CUDA 实现，将 router 输出的 token-to-expert mapping 与 expert FFN（silu-gate + up_proj + down_proj）融合为一个 kernel，避免中间结果的 HBM 往返。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文 benchmark 代码未明确说明开源。所有评估基于开源框架 vLLM 和开源模型（Mixtral、DeepSeek、Qwen、Phi、OLMoE 均在 HuggingFace 可获取）。
  - vLLM MoE 推理从输入到 H100 硬件执行的全过程（以 Mixtral-8x7B, TP=4, Fused MoE 为例）：
    ```
    [用户输入] → vLLM API Server (POST /v1/completions)
    
    ① Tokenization & Scheduling:
       - Tokenizer 将 prompt 转为 token IDs [batch_size, seq_len]
       - vLLM Scheduler 分配 KV-cache blocks（PagedAttention）
       - 若启用 TP=4：每张 GPU 获得 1/4 的权重分片
    
    ② Prefill Phase (逐层执行, Layer 0..31):
       For each MoE layer i:
         a) Self-Attention (TP=4):
            - 每张 GPU 计算 QKV_proj(1/4 列切分) → 本地 Q,K,V
            - FlashAttention 计算本地 self-attention
            - All-reduce 聚合同步 output_proj 结果
            - KV-cache 写入 PagedAttention block table
         
         b) Router:
            - hidden_states @ W_gate → logits [B,S,8]
            - Softmax + TopK(k=2) → routed_experts, routing_weights
         
         c) Fused MoE (单 kernel 执行):
            - Kernel 输入：hidden_states [B,S,4096], W_gate_fp8[8,4096,14336], 
              W_up_fp8[8,4096,14336], W_down_fp8[8,14336,4096], routing_map
            - Kernel 内部：
              ① Token-to-expert dispatch: 根据 routing_map 重排 tokens
              ② Grouped GEMM: 合并同 expert 的 tokens → batched matmul
              ③ SiLU(gate_out) * up_out → element-wise activation
              ④ down_proj matmul → expert output
              ⑤ Weighted sum: expert_output * routing_weight → final_output
            - Fused kernel 优势：消除 ① 的中间 tensor HBM write/read
              和 ③④ 之间的 kernel launch 开销（合计节省 12-20% 延迟）
    
    ③ Decode Phase:
       - 逐 token 自回归生成，每个 token 仅需 attention(单 token Q)
         + Fused MoE forward
       - TP all-reduce 通信仅在 attention output 和 MoE output 后各一次
       - KV-cache 追加新的 K,V block
    
    ④ 输出:
       - vLLM 收集所有 GPU 输出 → detokenize → 返回 text
    ```
  - 并行策略选择的关键 insight：在 H100 上 TP 扩展效率最高（1→4 GPU 吞吐量 >2×），因为 NVLink 高带宽使 all-reduce 通信开销被充分掩盖；PP 因 stage imbalance 和同步开销几乎无加速；EP 的 all-to-all dispatch/combine 开销在小 expert activation 场景下抵消了并行收益。
