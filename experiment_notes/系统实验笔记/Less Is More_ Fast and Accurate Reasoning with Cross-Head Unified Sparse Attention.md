## Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

- 属于Serving调度的实现是什么？实验比较什么？
  将 LessIsMore 的 CUSA 稀疏注意力机制集成到 SGLang（Zheng et al., 2024）serving 框架中，配合 FlashInfer attention kernel 库实现推理服务。修改 SGLang 的 attention 计算路径：对于 Full Attention Layers 保持标准 FlashInfer attention，对于 Sparse Attention Layers 替换为基于统一 token 索引 ρ 的稀疏 attention kernel（仅加载 ρ 中选中的 KV cache），减少 decode 阶段的 memory bandwidth 消耗。实验比较 SGLang + LessIsMore vs SGLang + TidalDecode vs SGLang + Quest vs SGLang + Full Attention（FlashInfer）在 DeepSeek-R1-Distilled-LLaMA-8B 上的端到端每 token 解码延迟，context lengths 16K/32K/64K，token budget 2K。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA A5000 GPU（SGLang 集成端到端延迟测试，Table 1）；单张 NVIDIA A100 80GB GPU（端到端 TBT speedup 测试，Table 3/Figure 6a）。FlashInfer 作为 attention 后端。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：SGLang（https://github.com/sgl-project/sglang）+ FlashInfer（https://flashinfer.ai/）attention kernel 库。LessIsMore 开源：https://github.com/DerrickYLJ/LessIsMore。
  修改内容：
  1. **Attention 路径扩展**：在 SGLang 的 attention 层中增加三种 layer 类型的 routing——Full Attention Layers 保持标准 FlashInfer kernel，Token Selection Layer 执行完整 attention + 跨 head 统一 token 选择（CUSA），Sparse Attention Layers 使用基于 ρ 的稀疏 attention kernel。
  2. **Token 索引管理**：维护 per-sequence 的统一 token 索引集 ρ，在 token selection layer 更新，后续层复用。
  3. **KV Cache 加载优化**：Sparse Attention Layers 仅从 KV cache 加载 ρ 中的 token 的 K/V，减少 HBM 带宽消耗。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源：https://github.com/DerrickYLJ/LessIsMore（论文仓库），集成到 SGLang + FlashInfer。

  **SGLang + LessIsMore 推理全流程（DeepSeek-R1-Distill-Llama-8B，单 GPU NVIDIA A5000）**：

  ```
  输入：用户 reasoning prompt → SGLang Tokenizer → token 序列
  ↓
  [1] Prefill Stage
    - 对所有层执行 Full Attention（FlashInfer kernel）
    - 生成完整 KV cache C（所有层，所有 token）
    - 首 token 生成（TTFT）
  ↓
  [2] Decoding Loop（逐 token 生成，最多 32K tokens）
    For each new token:
      a) QKV Projection: h → q, k, v（通过 W_qkv 矩阵乘法，cuBLAS）
      b) KV cache 更新: C.append(k, v)
      c) Per-layer Attention:
          Layer 0-1 (Full Attention Layers):
            - FlashInfer full attention kernel
            - 对所有 C[:] token 计算 O = softmax(qK^T/√d)V
          Layer 12 (Token Selection Layer for Qwen3-8B):
            - FlashInfer full attention kernel（计算 attention + token 重要性估计）
            - CUSA token selection:
              P = q @ C.K^T  # [32, 1, L_kv]
              各 head 独立 TopK: ρ_head = TopK(P, k=K·0.75)
              跨 head 统一: ρ_unified = UnionFlatten(ρ_head)
              ρ = ρ_unified[:K·0.75] ∪ Recent(K·0.25)
          Layer 2-11, 13-31 (Sparse Attention Layers):
            - 复用 token selection layer 的 ρ
            - Sparse FlashInfer kernel: 仅从 KV cache 加载 K[ρ], V[ρ]
            - 计算 O = softmax(qK[ρ]^T/√d)V[ρ]
      d) FFN: 标准 Feed-Forward Network
      e) lm_head → Sampling → next token
    Until EOS or max_tokens (32K)
  ↓
  输出：生成的 reasoning trace + 最终答案
  ```

  **端到端延迟**（Table 1, DeepSeek-R1-Distill-Llama-8B, A5000, budget=2K, ms/token）：

  | Method | 16K | 32K | 64K |
  |--------|-----|-----|-----|
  | LessIsMore | 23.0 | 23.4 | 24.1 |
  | TidalDecode | 24.3 | 24.7 | 25.4 |
  | Quest | 24.2 | 24.4 | 24.8 |
  | Full Attention | 25.3 | 28.4 | 34.4 |

  **端到端 TBT Speedup**（Table 3, A100, SGLang）：
  - LessIsMore-2K: 16K→1.11×, 32K→1.25×, 64K→1.51×
  - LessIsMore-4K: 16K→1.09×, 32K→1.22×, 64K→1.48×
