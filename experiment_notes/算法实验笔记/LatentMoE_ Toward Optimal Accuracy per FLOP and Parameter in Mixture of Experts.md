## LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - LatentMoE 提出一种新的 MoE 架构，通过将 token 投影到低维 latent space 来解耦 expert routing 和计算，并利用节省的 FLOPs/内存/通信来扩展 expert 数量和 top-k。核心实现包含两种变体：
    1. **ℓ-MoE_eff（效率导向变体）**：将 routed expert 的输入维度从 d 压缩到 ℓ（压缩比 α = d/ℓ），同时将总 expert 数 N 扩展 α 倍，保持 top-K 不变。communication cost 和 memory bandwidth cost 分别降低 α 倍，同时保持模型精度不变。定义为 ℓ-MoE_eff(x) = W_↑ · (Σ_{i∈T_{K,N'}} p'_i · E_i(W_↓·x; ℓ)) + Σ_{j=N'+1}^{N'+S} E_j(x; d)，其中 N' = α·N。
    2. **ℓ-MoE_acc（精度导向变体，推荐）**：在 ℓ-MoE_eff 的基础上，同时将 top-K 也扩展 α 倍（K' = α·K）。communication cost 和 memory bandwidth cost 与 standard MoE 相同，但通过指数级扩大 expert combination 空间（C(αN, αK) ≥ C(N, K)^α）提升了模型精度。
    3. **五项 Design Principles**：(I) 低延迟场景 memory bandwidth 是瓶颈，最大化 accuracy per parameter；(II) 高吞吐场景 all-to-all 通信是瓶颈，减少 routed hidden dimension d；(III) 保持 U_eff = K·m 非线性预算以维持质量；(IV) 存在任务特定特征秩 r_eff 作为 d 压缩下界；(V) 同时增加 N 和 K 指数级扩大专家组合空间。
  - 实验比较：
    - **Ablations (16BT-2BA)**：压缩比 α sweep (α=1,2,4,8)，验证 α≤4 时质量保留；expert scaling 消融（压缩 4× 有/无 expert 扩展），验证 expert scaling 必要性；ℓ-MoE_eff vs ℓ-MoE_acc vs baseline validation loss 对比。
    - **Scaling (95BT-8BA Transformer + Hybrid-73BT-8BA Mamba-Attention MoE)**：ℓ-MoE_eff/ℓ-MoE_acc vs baseline 在 MMLU Pro, MMLU, Code, Math, Commonsense 上的 accuracy。
    - **Inference Performance**：Hybrid-73BT-8BA 在 2× H100 GPU 上 vLLM FP8 的 tokens/s/GPU（concurrency=1,4,16,64,128），LatentMoE vs Standard MoE 最多差 6% throughput。
    - **Projected Trillion-Parameter**：Kimi-K2-1T vs Kimi-K2-1T-LatentMoE（EPM≈1.35×）的 throughput-latency Pareto frontier，1.24×-3.46× slowdown for iso-accuracy standard MoE。

- 硬件平台是什么，配置是什么。
  - **系统分析基准**：NVIDIA GB200 GPUs, NVLink ~1800 GB/s bidirectional, 900 GB/s per direction。EP=64 GPUs, FP4=10 PFLOPs, HBM BW=8 TB/s。
  - **Inference 实测**：2× NVIDIA H100 GPUs, vLLM, FP8 per-tensor quantization。
  - **Trillion-parameter 投影**：proprietary performance simulator, 200K+ operating points。
  - **训练硬件**：论文未明确说明具体 GPU 型号和数量，基于 DeepSeek-v2-lite 架构和超参训练。

- 模型是什么。数据集和bench分别是什么。
  - **模型配置**（Table 2）：
    - **16BT-2BA**：L=27, d=2048, N=64, K=6, S=2, m=1408, SwiGLU, 16 heads。基于 DeepSeek-v2-lite。
    - **95BT-8BA**：L=32, d=4096, N=128, K=6, S=2, m=2688, Squared-ReLU, 32 heads, cosine LR(max=1.2e-3), seqlen=8192, batch=768(~6M tokens)。
    - **Hybrid-73BT-8BA**：L=52(24 Mamba/MoE+4 Attn), d=4096, N=128, K=6, S=2, m=2688, Squared-ReLU, WSD LR schedule(max=8e-4)。
  - **LatentMoE 配置**：压缩比 α=4 (ℓ=512 for 16B, ℓ=1024 for 95B/Hybrid), N'=αN, K'=αK=24 (ℓ-MoE_acc)。
  - **Benchmarks**：MMLU Pro, MMLU, Code(HumanEval/+/MBPP/+ avg), Math(GSM8K CoT+MATH-500 avg), Commonsense(RACE, ARC-Challenge, HellaSwag, Winogrande avg)。
  - **训练数据**：论文未明确说明具体数据集，16B ablation 和 95B/Hybrid 分别训练至 300B-1T tokens。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文未给出独立 LatentMoE 代码仓库。LatentMoE 已集成到 NVIDIA Nemotron-3 Super/Ultra（arXiv:2512.20856）。TensorRT-LLM v1.2.0+ 支持 LatentMoE。Nemotron 配方开源：https://github.com/NVIDIA-NeMo/Nemotron。
  - **LatentMoE ℓ-MoE_acc 算法伪代码**：

```python
# === LatentMoE MoE Layer Forward (ℓ-MoE_acc) ===
# 输入: x ∈ R^{B×S×d}, latent dim ℓ, 压缩比 α = d/ℓ
# 共享: W_↓ ∈ R^{ℓ×d}, W_↑ ∈ R^{d×ℓ}, Router W_r' ∈ R^{N'×d}
# Routed Experts E_i(·;ℓ): W_FC1^{(i)}∈R^{m×ℓ}, W_gate^{(i)}∈R^{m×ℓ}, W_FC2^{(i)}∈R^{ℓ×m}
# Shared Experts E_j(·;d): 在原始空间 d 操作

def latente_moe_acc_forward(x):
    # 1. Router（原始空间 d）
    gate_logits = x @ W_r'.T                  # [B,S,N'] where N'=αN
    topk_vals, topk_ids = topk(softmax(gate_logits), K')  # K'=αK
    topk_vals /= topk_vals.sum(dim=-1, keepdim=True)
    
    # 2. Shared Down-Projection
    z = x.reshape(-1, d) @ W_↓.T              # [B×S, ℓ]
    
    # 3. Routed Expert 计算（在 latent space ℓ）
    moe_out = zeros(B*S, ℓ)
    for e in selected_experts:
        mask_e = token_assigned_to_expert(e)
        z_e = z[mask_e]                        # [n_e, ℓ]
        # SwiGLU/Squared-ReLU FFN in latent space:
        h_g = activation(z_e @ W_gate^{(e)}.T) # [n_e, m]
        h_u = z_e @ W_FC1^{(e)}.T              # [n_e, m]
        h   = h_g * h_u                        # [n_e, m]
        e_out = h @ W_FC2^{(e)}.T              # [n_e, ℓ]  FC2 down
        moe_out[mask_e] += gate_e * e_out
    
    # 4. Shared Up-Projection
    routed_y = moe_out @ W_↑.T                 # [B×S, d]
    
    # 5. Shared Experts（原始空间）
    shared_y = sum(shared_expert_ffn(x, E_j) for j in 1..S)
    return routed_y + shared_y
```

  - **关键张量流（ℓ-MoE_acc，单 token，d=4096, ℓ=1024, α=4, K'=24）**：
    1. Router: x[1,4096] @ W_r'[512,4096] → probs[1,512] → top-24 → 24 gate_weights
    2. Down-proj: W_↓[1024,4096] @ x → z[1,1024]（共享，所有 experts 复用）
    3. All-to-All dispatch（latent space）: z[1024] → 24 experts。通信量 ∝ 24×1024 = 24576（vs standard MoE 6×4096 = 24576，相同）
    4. Expert FFN（latent space）: 每个 expert 权重 m×ℓ+ℓ×m（减少 4× vs d×m），memory BW per expert ↓4×
    5. All-to-All combine + Up-proj: W_↑[4096,1024] @ z_combined → routed_out[1,4096]
    6. Shared Experts（d=4096）: 2 shared experts 在原始空间计算
    7. Expert 组合空间: C(512,24) vs C(128,6)，指数级增长

- 属于算法pipeline的实现是什么？实验比较什么？
  - FlyLoRA 提出一种受果蝇嗅觉回路启发的隐式 MoE-based LoRA 变体，核心设计：(1) 将 LoRA 的下投影矩阵 A 替换为**冻结的稀疏随机投影矩阵**（每行仅 p < n 个非零元素，采样自 N(0, 1/r²)），(2) 在 B 矩阵中执行 **rank-wise top-k 专家激活**——对 Ax 投影结果的 r 维分量中取 top-k 幅值，仅激活对应的 B 列。A 同时承担下投影和隐式 router 的双重角色，**无需显式 router 参数**。通过稀疏随机投影的近似距离保持性（Theorem 3.1）实现隐式路由，并通过 top-k 稀疏性实现 rank 间梯度去相关（Theorem 3.3: 协方差降低因子约 k²/r²），同时不同 task 的独立随机 A_i/A_j 天然近似正交（Theorem 3.4），实现多任务模型合并时的 inter-task 去耦合。
  - 实验比较：(1) 单任务 SFT：vanilla LoRA(r=8)、LoRA(r=32)、Split-LoRA(4×8)；更强 baseline AdaLoRA、SoRA、HydraLoRA。(2) 多任务模型合并（weight averaging、TIES-MERGING、DARE）：与上述 LoRA 变体比较合并前后性能下降幅度。额外与 KnOTS、L-LoRA 等高级合并方法比较。(3) 消融：负载均衡策略（loss-free vs loss-controlled）、A 冻结/可训练、sparsity ratio、activated rank k、total rank r、k-selection 策略、A 初始化方案等。

- 硬件平台是什么，配置是什么。
  - 主要实验：Linux server, Ubuntu 20.04.4 LTS, Intel Xeon Platinum 8358P CPU @2.60GHz, 8× NVIDIA GeForce RTX 3090 (24GB), CUDA 11.7。
  - Qwen-2.5-14B 实验：8× NVIDIA A100 GPUs。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-3.1-8B, Qwen-2.5-7B, Qwen-2.5-14B（扩展实验）。
  - 训练数据集：MMLU（99,842条多选题，涵盖57学科）、ScienceQA（12,726条，仅用文本部分）、GSM8K（7,473条数学应用题）、CodeAlpaca-20k（20,022条代码指令对）。
  - 评估 benchmark：MMLU（14,042条测试）、ScienceQA（4,241条测试）、GSM8K（1,319条测试）、HumanEval（164条 Python 编程题，Pass@1/5/10）。
  - 评估方式：zero-shot，HumanEval 用 pass@k，其余用 accuracy。所有结果报告 3 个随机种子误差条。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源链接：https://github.com/gfyddha/FlyLoRA
  - FlyLoRA 算法 pipeline（基于 Section 3.1, Eq. 7-11）：
    ```
    Input: 输入 token x ∈ R^n, 预训练权重 W0 ∈ R^{m×n}, 
           冻结稀疏随机投影 A ∈ R^{r×n}（每行 p 个非零 ~N(0,1/r²)）,
           可训练 B ∈ R^{m×r}, 负载均衡偏置 d ∈ R^r,
           激活 rank k, 总 rank r, scaling factor α=2r
    Forward:
      y = A @ x                                  # sparse projection: O(r·p)
      y_biased = y + d                           # load balancing bias
      I_topk = argtopk(y_biased, k)              # 选择 top-k 激活维度
      mask = zeros(r); mask[I_topk] = 1          # k 个 1, r-k 个 0
      delta = (α/r) * (B * mask) @ y             # 仅激活 k 个 B 列, O(m·k)
      output = W0 @ x + delta
    Load Balancing Update (每步, loss-free):
      for i in 1..r:
        d_i += u * sign(expected_count_i - actual_count_i)
    
    Backward: 仅 B 被更新，A 保持冻结
      grad_B_masked = grad_loss @ y^T ⊙ mask   # 只有 k 列有非零梯度
      B = B - lr * grad_B_masked
    ```
  - FlyLoRA 与 baseline 的关键张量计算对比：
    - LoRA(r=8):  W0·x + B_{m×8}·(A_{8×n}·x)，激活参数 2·d·8
    - LoRA(r=32): W0·x + B_{m×32}·(A_{32×n}·x)，激活参数 2·d·32
    - Split-LoRA(4×8): W0·x + Σ_i G(x)_i·B_i·(A_i·x)，4个8-rank expert + router W_g∈R^{4×n}，激活参数 2·d·8 + d·4
    - FlyLoRA(k=8): W0·x + Σ_{i∈I_topk} b_i·(a_i·x)，仅激活 8 个 rank-1 expert 在 B，激活参数 d·8（无 router 开销）
  - 训练配置（Appendix C）：Total rank r=32, activated k=8, scaling factor α=64, target modules {q,k,v,o,gate,down,up}_proj, optimizer AdamW, warmup ratio 0.01, gradient accumulated batch 128, dropout 0.0。数据集特定配置见 Table 19（epochs 1~20, max seq len 128~512, micro batch size 8, learning rate 3e-4~6e-4）。混合精度训练（16-bit）。
