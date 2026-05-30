## QLoRA (Quantized Low-Rank Adaptation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QLoRA（Quantized Low-Rank Adaptation）由 Dettmers et al. (2023) 提出，是一种将 4-bit NormalFloat (NF4) 量化与 LoRA 参数高效微调结合的 LLM 微调方法。其核心流程：(1) **量化阶段**：对预训练权重 W 执行 NF4 量化得到 Q = q_NF4(W)，将 FP16 权重压缩至 4-bit（存储为 4-bit index + block-wise absmax scale），同时对 scale 做双重量化（8-bit FP + 32-bit FP）进一步压缩；(2) **LoRA 附加**：在所有量化后的线性层附加低秩适配器 A, B（A ∼ N(0,σ²), B=0），标准 LoRA 零初始化；(3) **微调阶段**：freeze 量化权重 Q，仅优化 LoRA 参数。前向传播时 Q 临时解量化为 simulated FP16 参与计算 Y = X · dequant(Q) + X · A B^T。QLoRA 可在单张 48GB GPU 上微调 65B 模型（~18GB 显存）。

QLoRA 的核心缺陷——也是 LoftQ 论文的动机：量化误差导致初始权重 Q+AB^T = Q ≠ W，即微调起点偏离原始预训练权重。在低比特（2-bit/3-bit）时该偏差可能导致模型不收敛。LoftQ 通过交替优化解决了这一问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# QLoRA Pipeline

# Step 1: NF4 量化（离线，每个权重矩阵执行一次）
W_block = W.reshape(d_out, d_in // 64, 64)        # block_size=64
for each block:
    s = absmax(block)                               # FP32 scale
    W_norm = block / s                              # 归一化到 [-1,1]
    W_q[block] = NF4_quantize(W_norm)               # NF4 查表量化 → 4-bit index
    s_FP8 = FP8_quantize(s)                         # 双重量化 scale
    s_FP32 = FP32(s - s_FP8)                        # 残差

# Step 2: LoRA 初始化
lora_A = nn.Linear(d_in, r, bias=False)            # N(0, σ²) 初始化
lora_B = nn.Linear(r, d_out, bias=False)           # 零初始化

# Step 3: 微调
for batch in data:
    # 前向
    W_sim = NF4_dequantize(W_q, s_FP8, s_FP32)    # simulated dequantization
    h = x @ W_sim^T + (x @ lora_A @ lora_B^T) * α/r
    # 反向
    loss.backward()                                 # 梯度仅流经 lora_A, lora_B
    optimizer.step()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/artidoro/qlora。基于 HuggingFace Transformers + PEFT + bitsandbytes。使用方式：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` → `prepare_model_for_kbit_training()` → `LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])`。关键设计：NF4 量化（信息论最优 4-bit 正态分布量化）+ 双重量化（压缩 scale 存储，每参数从 0.5 bit 降至 0.127 bit）+ 分页优化器（CPU offload 梯度检查点避免 OOM）。局限：2-bit 时失效（LoftQ 论文验证）；仅支持 weight-only 量化（激活保持 FP16/BF16）。

涉及论文标题：
- LoftQ: LoRA-Fine-Tuning-aware Quantization for Large Language Models
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- QA-LoRA Quantization-Aware Low-Rank Adaptation of Large Language Models
- QERA: an Analytical Framework for Quantization Error Reconstruction
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

QeRL (Huang et al., NVIDIA, 2025) 发现 QLoRA (NF4+LoRA) 在 RL 训练中表现不佳：(1) NF4 的 unpack+lookup table 反量化导致 rollout 比 BF16 LoRA 慢 0.7-0.8×；(2) NF4 量化噪声是静态且确定性的，对 RL 后期训练的 exploitation 阶段不利——QeRL 引入 Adaptive Quantization Noise (AQN) 解决此问题；(3) QeRL (NVFP4+LoRA+AQN) 在 GSM8K 上 7B 达 90.8%，比 QLoRA (NF4+LoRA) 高 5.8 点。

QA-LoRA (Xu et al., Huawei, 2023) 指出 QLoRA 的核心局限：(1) QLoRA 微调后将 s·AB 加回量化权重，使最终模型恢复为 FP16，若需量化推理则必须做 PTQ（GPTQ 后处理），导致不可控的精度损失——尤其在 INT3/INT2 低位宽下退化严重（LLaMA-7B INT2: MMLU 5-shot 仅 25.0-25.8%）；(2) NF4 缺乏 CUDA 算子优化，训练和推理速度均慢于 INT4。QA-LoRA 通过 group-wise 操作使 LoRA 权重仅合并到零点矩阵 β（不改变 Ŵ 和 α），保持 INT 格式，解决了两个问题。QA-LoRA 论文称其 INT4 实现的 QLoRA 变体与原 NF4 QLoRA 精度差异在 ±0.5% 以内。

QLoRA 的安全影响：Q-resafe (Chen et al., ICML 2025) 的系统安全评估显示，QLoRA INT4 量化 Llama-2-7B-Chat 在 benign 校准数据集（UltraChat）上 ASR 从 0.3%（FP16）飙升至 42.3%，在直接有害数据集（AdvBench）上更升至 85.3%，是所有评估方法中安全退化最严重的。这是因为 QLoRA 优先优化效用而牺牲了安全——仅通过 LoRA 低秩适配调整少量参数，不足以保护模型的安全能力。Q-resafe 通过 DPO + 安全关键权重选择性修补可在 1.2 GPU-hours 内将 QLoRA INT4 ASR 从 42.3% 恢复至 2.4%。
