## AWQ (Activation-Aware Weight Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AWQ（Activation-aware Weight Quantization）是 Lin et al. (MLSys 2024 Best Paper, MIT HAN Lab) 提出的硬件友好的 LLM 低比特 weight-only 后训练量化方法。核心发现：LLM 中权重并非同等重要——仅保护 0.1%-1% 的显著（salient）权重通道即可大幅降低量化误差（OPT-6.7B INT3-g128 PPL 从 43.2 降至 13.0）。关键洞察：要识别显著权重通道，应参考**激活分布**而非权重分布——激活幅度更大的通道对应的权重更重要（这些通道处理更重要的特征）。为避免硬件低效的混合精度实现（部分通道 FP16 + 部分 INT），AWQ 通过数学推导证明：对显著通道的权重乘以 s > 1，并对激活除以 s（等效变换），可以降低显著权重的相对量化误差（误差比例 `Δ'/Δ · 1/s < 1`，因为 `Δ' ≈ Δ` 且 `s > 1`）。Per-channel scale s 通过简洁的参数化搜索空间确定：`s = s_X^α`（s_X 为 per-channel 平均激活幅度，α ∈ [0,1] 通过 20 步网格搜索找到最优值），目标是最小化量化后输出与原始输出的 MSE。整个过程不依赖反向传播或 block-wise reconstruction，因此：(1) 仅需极少校准数据（16 条序列 vs GPTQ 的 192 条），(2) 不对校准集过拟合，可泛化到多模态 VLM、代码生成、数学推理等不同领域。可与 GPTQ 结合：AWQ per-channel scaling 作为前置步骤 → GPTQ 二阶误差补偿，在 INT2-g64 极端低比特下进一步提升性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-7B INT4-g128 量化为完整例子：

**数学原理（Eq. 1-3）：**
量化函数：`Q(w) = Δ · Round(w/Δ)`，其中 `Δ = max(|w|) / (2^{N-1} - 1)`。

对显著权重 w 乘以 s > 1，反向缩放激活 x：
```
Q(w·s) · (x/s) = Δ' · Round(w·s/Δ') · x · (1/s)
```
误差比例 = `Δ'/Δ · 1/s`。由于：(1) RoundErr ≈ 0.25（均匀分布），(2) 单元素缩放通常不改变组的 max → `Δ' ≈ Δ`，(3) s > 1 → 相对误差降低。但当 s 过大时，Δ' 会因 max 值改变而增大，导致 non-salient 通道误差放大。因此需通过网格搜索找到平衡点。

**完整算法 Pipeline：**
```python
# Step 1: 收集激活统计量
# 用 16 条 Pile 校准序列前向传播
for layer in model.layers:
    X = cached_input_activation[layer]     # [C_in, L]
    s_X = X.abs().mean(dim=1)              # [C_in], per-channel 平均激活幅度

# Step 2: 网格搜索最优 α
    best_alpha, best_loss = None, float('inf')
    for alpha in linspace(0, 1, 20):      # grid_size=20
        s = s_X ** alpha                   # per-channel scale

        W_scaled = W * s.unsqueeze(0)      # [C_out, C_in] * [1, C_in]
        W_q = groupwise_quantize(W_scaled, bits=4, group_size=128)
        # groupwise_quantize: 每 128 个元素一组，组内计算 Δ = max(|w_group|)/7
        #                    W_int = Round(W_scaled / Δ)

        X_scaled_inv = X / s.unsqueeze(1)  # [C_in, L]
        Y_q = dequantize(W_q) @ X_scaled_inv
        Y_fp = W @ X
        loss = MSE(Y_q, Y_fp)

        if loss < best_loss:
            best_alpha, best_loss = alpha, loss

# Step 3: 应用最优 scale 并量化
    s_final = s_X ** best_alpha
    W_final_q = groupwise_quantize(W * s_final.unsqueeze(0), bits=4, group_size=128)
    # diag(s)^{-1} 融合进前一层（LayerNorm weight 或前一层 Linear weight）

# Step 4 (可选): 与 GPTQ 结合
# 做完 AWQ per-channel scaling → 再做 GPTQ 逐列 Hessian 误差补偿
# INT2-g64 下 AWQ+GPTQ PPL: 15.71 vs GPTQ alone: 16.65 (OPT-6.7B)
```

**Table 1 关键数据（OPT-6.7B, INT3-g128, WikiText-2 PPL↓）：**
- FP16: 10.86
- RTN (全量 INT3): 23.54
- RTN + 1% FP16 (基于激活分布选): 11.39 ← 与 AWQ 全 INT3 精度相当
- RTN + 1% FP16 (基于权重分布选): 22.37 ← 无效
- AWQ (全 INT3, 用 per-channel scaling 保护显著通道): 11.92

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AWQ 官方开源：https://github.com/mit-han-lab/llm-awq (MIT License)。Python 使用：
```python
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model.quantize(tokenizer, quant_config={
    "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"
})
```
已被广泛集成到生产系统：
- **vLLM**: `vllm/model_executor/layers/quantization/awq.py`
- **HuggingFace Transformers**: `from transformers import AwqConfig`
- **NVIDIA TensorRT-LLM**: 原生支持 AWQ INT4 weight-only
- **LMDeploy**: TurboMind 引擎支持 AWQ 推理
- **Intel Neural Compressor**: 支持 AWQ 量化
- **llama.cpp**: 通过 GGUF 格式支持 AWQ

AWQ 覆盖的模型范围：LLaMA/Llama-2 (7B-70B)、OPT (1.3B-30B)、Mistral-7B、Mixtral-8x7B (MoE)、Falcon、MPT、StarCoder、Vicuna (指令微调)、OpenFlamingo-9B (VLM)、LLaVA-13B (VLM)、VILA-7B/13B (VLM)、CodeLlama-7B。首次实现多模态 LLM 的低比特量化。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- AFPQ Asymmetric Floating Point Quantization for LLMs
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

AWQ 的安全影响：Q-resafe 的安全评估发现，AWQ（PTQ w/o FT）INT4 量化后 Llama-2-7B-Chat 的 ASR 从 0.3%（FP16）飙升至 42.4%（decoding attack 下），Gemma-7B-Instruct 从 9.2% 升至 17.9%。由于 AWQ 不使用校准数据集微调，无法通过数据集选择控制安全风险，必须在量化后使用 Q-resafe 进行安全修补。Q-resafe 对 AWQ 采用不同的修补策略：在全精度模型上识别安全关键权重保留为 FP16，其余权重 AWQ INT4 量化——不执行 DPO 训练。修补后 ASR 降至 baseline+0.8%（Llama）和 baseline+0.4%（Gemma）。

---
