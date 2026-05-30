## VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：VisualRWKV 是首个将线性 RNN（RWKV）应用于多模态视觉语言模型的架构。核心创新包括三部分：(1) Data-dependent Recurrence：将 RWKV 的数据独立 token shift 和 time mixing 升级为数据依赖版本——Token Shift 通过 ddlerp (data-dependent linear interpolation) + LoRA 实现，`ddlerp_α(a,b)=a+(b-a)⊙(λ_α+tanh(a+(b-a)⊙μ_x)A_α)B_α`；Time Mixing 的 decay 从固定 w 变为动态 w_t，`d_t=lora_d(ddlerp_d(x_t,x_{t-1})), w_t=exp(-exp(d_t))`；(2) Sandwich Prompt：在 instruction token 中间插入 image token，使模型先阅读指令再处理图像再继续指令，解决 RNN 无法回溯的问题；(3) 2D Image Scanning：将 RWKV 的因果单向扫描扩展为双向(BiDir: Forward+Backward)和多向(MultiDir: Forward+Backward+Upward+Downward)交替排列，不增加计算量但增强 2D 视觉信息提取。训练流程为两阶段：(1) 视觉-语言对齐预训练（冻结 vision encoder 和 RWKV LLM，仅更新 projector）；(2) 视觉指令微调（同时更新 projector 和 RWKV LLM）。使用 CLIP-L (0.3B) 作为 vision encoder，RWKV-5/RWKV-6 系列作为 LLM backbone。
  - 实验比较：(1) Main Results（Table 2）：VisualRWKV 1.6B/3B/7B 在 8 个 benchmark（VQA-v2, GQA, ScienceQA, TextVQA, POPE, MME, MMBench, MMBench-CN）上对比 LLaVA-1.5、BLIP-2、InstructBLIP、MiniGPT-4、Qwen-VL、Shikra、MobileVLM 等 SOTA Transformer VLM；(2) Scaling 消融（Table 1）：VisualRWKV-Base → +Data-dep Recurrence → +Bidirection + Sandwich → +Better LR → Scale up 3B → Scale up 7B；(3) Prompting 方法消融（Table 3）：Image First vs Image Last vs Sandwich Prompt；(4) Scanning 方法消融（Table 4）：UniDir vs BiDir vs MultiDir；(5) 学习率消融（Table 10）：1.6B/3B/7B 各 scale 的最优学习率搜索；(6) 效率分析（Figure 1）：VisualRWKV 7B vs LLaVA-1.5 7B 的推理速度和 GPU 内存随序列长度变化（最高 24K tokens），VisualRWKV 速度优势 3.98×，节省 54% GPU 内存；(7) Text-only 能力（Table 5）：验证视觉指令微调后纯文本能力不退化；(8) 单阶段 vs 两阶段训练（Figure 5）；(9) Loss reduction 方法消融（Table 7/8）：batch-level vs sample-level reduction；(10) Weight decay 消融（Table 13）；(11) VisualRWKV Hybrid（Table 14）：添加 Tiny Attention layer 的混合模型。

- 硬件平台是什么，配置是什么。
  - 训练：8× NVIDIA A100-80GB GPU（标准训练和 benchmark 评估），VisualRWKV 7B 使用 6× A100 GPU（因 8 GPU 显存不够）
  - 效率分析：单张 L20-48GB GPU
  - 优化器：AdamW，cosine decay LR schedule，无 weight decay
  - 训练框架：NVIDIA PyTorch NGC Container (23.07-py3)，lightning 1.9.5，DeepSpeed 0.12.6
  - 计算预算：VisualRWKV 1.6B 单 epoch 53.6 GPU hours（8×A100）；3B 单 epoch 90.4 GPU hours（8×A100）；7B 单 epoch 159 GPU hours（6×A100）

- 模型是什么。数据集和bench分别是什么。
  - 模型：VisualRWKV-Base（RWKV-5 1.6B LLM backbone）、VisualRWKV 1.6B（RWKV-6 1.6B + CLIP-L 0.3B = 1.9B 总参数）、VisualRWKV 3B（RWKV-6 3.1B + CLIP-L = 3.4B）、VisualRWKV 7B（RWKV-6 7.6B + CLIP-L = 7.9B）、VisualRWKV-Hybrid（7B + Tiny Attention layer）。对比模型：LLaVA-1.5（Vicuna-7B/13B）、BLIP-2（Vicuna-13B）、InstructBLIP（Vicuna-7B/13B）、MiniGPT-4、Shikra、Qwen-VL/Chat、MobileVLM-3B、VL-Mamba、LLaVA-Phi、IDEFICS-9B/80B、Otter、mPLUG-Owl
  - 训练数据（与 LLaVA-1.5 完全一致）：(1) 视觉-语言对齐预训练：558K subset of LAION-CC-SBU；(2) 视觉指令微调：150K GPT-generated multimodal instruction-following data + ~515K academic VQA datasets（OK-VQA, TextVQA, GQA, VQA-v2）
  - Benchmark：VQA-v2（test-dev split）、GQA（test-dev split）、ScienceQA-IMG（test set, zero-shot）、TextVQA（validation set）、POPE（test set, COCO random/common/adversarial, avg F1）、MME-Perception、MMBench（development set）、MMBench-CN（中文版）、LAMBADA、English benchmarks（PIQA/StoryCloze16/HellaSwag/WinoGrande/ARC-Challenge/Easy/HeadQA/OpenBookQA/SciQ）、Multilingual benchmarks（xLAMBADA/xStoryCloze/xWinoGrande/xCOPA）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/howard-hou/VisualRWKV（Apache-2.0 license）
  - RWKV 预训练模型：https://huggingface.co/BlinkDL/rwkv-5-world（RWKV-5）、https://huggingface.co/BlinkDL/rwkv-6-world（RWKV-6）
  - 算法 pipeline 伪代码（VisualRWKV forward pass, 推理时逐 token 生成）：
    ```
    # === Step 1: Vision Encoding ===
    # Input: 原始图像 I → CLIP-L (ViT-L/14, 336×336 resolution)
    Z_v = CLIP_ViT_L14(I)  # penultimate layer features: [577, 1024]

    # === Step 2: Projector ===
    H_v = Projector(Z_v)  # 2-layer MLP: [577, D_llm], 576 image tokens

    # === Step 3: Sandwich Prompt ===
    # 格式: [System] [Q_prefix] <image_tokens> [Q_suffix]
    Input = concat([S_embed, Q_prefix_embed, H_v, Q_suffix_embed])
    # Input ∈ R^{L×D_llm}

    # === Step 4: VisualRWKV Blocks (RWKV-6 backbone) ===
    # 根据 layer index 决定 scanning 方向，交替排列
    For layer l = 0..L-1:
        # ---- Data-dependent Token Shift ----
        # ddlerp_α(a,b) = a + (b-a) ⊙ (λ_α + tanh((a+(b-a)⊙μ_α)A_α)B_α)
        r_t = ddlerp_r(x_t, x_prev) @ W_R  # receptance
        g_t = ddlerp_g(x_t, x_prev) @ W_G  # SiLU gate
        k_t = ddlerp_k(x_t, x_prev) @ W_K  # key
        v_t = ddlerp_v(x_t, x_prev) @ W_V  # value

        # ---- Data-dependent Time Decay ----
        d_t = lora_d(ddlerp_d(x_t, x_prev))
        w_t = exp(-exp(d_t))  # ∈ (0,1), channel-wise dynamic decay

        # ---- WKV Linear Attention (per-head, head_dim=64) ----
        # Recurrent state update:
        wkv_cur = diag(u) @ k_t^T @ v_t       # current token bonus
        wkv_state = diag(w_t) @ wkv_state + k_t^T @ v_t  # accumulated past
        wkv_total = wkv_cur + wkv_state

        # LayerNorm per head + output gating:
        o_t = concat(SiLU(g_t) ⊙ LayerNorm(r_t @ wkv_total)) @ W_O

        # ---- Channel Mixing (FFN) ----
        r'_t = ddlerp_r'(x_t, x_prev) @ W_R'
        k'_t = ddlerp_k'(x_t, x_prev) @ W_K'
        v'_t = ReLU(k'_t)^2 @ W_V'  # squared ReLU
        c_out = σ(r'_t) ⊙ v'_t

        x_t = x_t + o_t + c_out  # residual

    # === Step 5: Output ===
    logits = LM_Head(x_t)  # next token prediction
    ```

    关键张量形状（7B, D_llm=4096, h=64 heads）：
    - x_t ∈ R^{4096}，ddlerp 矩阵 A_α ∈ R^{4096×32}, B_α ∈ R^{32×4096}（LoRA rank=32）
    - WKV state 矩阵 S ∈ R^{64×64} per head（矩阵状态，替代标量状态）
    - 推理时无 KV cache，仅需维护 L × h 个 64×64 的 state 矩阵（恒定量，与序列长度无关）
    - GPU 内存恒定为 54% of LLaVA-1.5 @ 24K tokens
    - 推理速度恒定 O(1) per token，24K tokens 时 3.98× faster than LLaVA-1.5

    数据依赖 vs 数据独立的对比：
    ```
    # Data-independent (RWKV-5 / VisualRWKV-Base):
    α_t = (μ_α ⊙ x_t + (1-μ_α) ⊙ x_{t-1}) W_α     # μ 固定可学习
    w = exp(-exp(ω))                                   # ω 固定可学习
    wkv_t = sum(diag(w)^{t-1-i} @ k_i^T @ v_i)       # 固定 decay

    # Data-dependent (RWKV-6 / VisualRWKV):
    α_t = ddlerp_α(x_t, x_{t-1}) W_α                  # ddlerp with LoRA
    w_t = exp(-exp(lora_d(ddlerp_d(x_t, x_{t-1}))))  # 动态 decay
    wkv_t = sum(diag(Π w_j) @ k_i^T @ v_i)            # 时变 decay
    ```
