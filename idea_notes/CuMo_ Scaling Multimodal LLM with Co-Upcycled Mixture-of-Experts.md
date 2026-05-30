## CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts

- baseline方法是什么？
  - Baseline 方法：LLaVA v1.5 架构（CLIP ViT-L 作为视觉编码器 + 两层 MLP 连接器 + Mistral-7B LLM），所有模块均为 **dense MLP** 结构。
  - Baseline 全栈执行例子（从请求到 token 输出）：
    - **模型推理算法层**：输入图像 → CLIP ViT-L（dense MLP blocks）提取 visual tokens → 两层 dense MLP 将 visual tokens 投影到 word embedding 空间 → Mistral-7B LLM 的 dense MLP blocks 执行自回归解码，每个 token 必须通过全部 MLP 参数（7.1B 激活参数）
    - **系统框架层**：基于 LLaVA 框架，使用标准 PyTorch + HuggingFace Transformers 加载模型；推理时所有 dense 参数常驻 GPU 显存，使用 greedy decoding 策略
    - **编译框架层**：论文未明确说明
    - **kernel 调度层**：使用 flash-attention 加速 attention 计算；dense MLP 使用标准 PyTorch Linear + GELU kernel
    - **硬件架构层**：NVIDIA A100 GPU（8/16/32 卡），ZeRO-2/ZeRO-3/ZeRO-3-offload 分布式策略

  - Baseline 的痛点：
    1. **视觉侧扩展低效**：现有方法通过多编码器、更大 ViT、或复杂连接器（如 Q-Former）来增强视觉能力，但这些方法增加大量额外参数和 visual tokens，导致 LLM 处理负担加重
    2. **dense 模型的参数效率瓶颈**：dense MLP 的每个 token 必须通过全部参数，无法通过条件计算选择性激活相关专家来提升模型容量
    3. **MoE 在 LLM 侧已成熟，但视觉侧的 MoE 探索几乎空白**：MoE-LLaVA 仅在小型 LLM 中采用 MoE，未涉及视觉编码器或连接器

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：CuMo = CLIP-MoE + MLP-MoE + (可选 Pre-trained LLM-MoE)，通过 Co-Upcycling 将 pre-trained dense MLP 权重初始化为 MoE 专家，配合三阶段训练和辅助负载均衡损失。
  - **对应解决**：
    - **缺陷 1（视觉侧扩展低效）** → 将 CLIP ViT 和 MLP 连接器的 dense MLP 替换为 Top-2-in-4 稀疏 MoE 块，仅激活 50% 专家（CLIP-MoE 激活 0.50B/总 0.91B，MLP-MoE 激活 0.05B/总 0.10B），以极少的额外激活参数（7.1B → 7.8B，仅 +9.8%）显著提升视觉理解能力
    - **缺陷 2（dense 参数效率瓶颈）** → 每个 visual token 在 MoE 块中仅通过 Top-K 选中的 2 个 expert MLP（而非全部 4/8 个），通过 Router 的条件计算使模型容量增加而激活 FLOPs 几乎不变
    - **缺陷 3（视觉 MoE 缺乏训练方法）** → 提出 Co-Upcycling：用预训练/预微调 dense MLP 权重初始化 MoE 专家（而非随机初始化），避免训练不收敛；三阶段训练：MLP 预训练 → 全参数预微调（ALLaVA 标注数据温热模型）→ 含 MoE 的指令微调；辅助 bzloss（L_balance + L_z_loss）维持专家负载均衡

  - 论文方法全栈执行例子（对比 baseline）：
    - **模型推理算法层**：
      - 输入图像 → CLIP ViT-L（**每个 transformer 层中 dense MLP → Top-2-in-4 MoE**）：`X_out = Σ(i=1..2) W_K[i] ⊙ MLP_expert[i](X)`，仅 2/4 专家被激活 → 输出 visual tokens
      - MLP 连接器（**dense MLP → Top-2-in-4 MoE**）：visual tokens 通过稀疏门控路由，仅 2/4 专家参与投影 → word embedding tokens
      - LLM 解码（**Mistral-7B dense → 可选 Mixtral 8×7B pre-trained MoE**）：每 token 通过 Top-2 专家（12.9B 激活/46.7B 总）
      - 总激活参数：CuMo Mistral-7B = 7.80B（vs baseline 7.1B）；CuMo Mixtral-8×7B = 13.45B
    - **系统框架层**：基于 LLaVA + CuMo 自定义模块（`cumo/serve/`），Gradio Web UI / CLI 推理，支持 4-bit/8-bit 量化减少显存；训练使用 DeepSpeed ZeRO-3-offload
    - **编译框架层**：论文未明确说明
    - **kernel 调度层**：使用 flash-attention；MoE 的 Top-K 路由和加权求和通过标准 PyTorch scatter/gather 操作实现，无自定义 CUDA kernel
    - **硬件架构层**：NVIDIA A100 GPU（8/16/32 卡），三阶段训练逐步增加 GPU 数量（8→16→32），MoE 引入的额外参数通过 ZeRO-3-offload 卸载到 CPU 内存以节省 GPU 显存
