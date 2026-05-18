## Visual Autoregressive (VAR) Model

术语是什么？
Visual Autoregressive (VAR) 是一种将图像生成建模为自回归 next-token prediction 的生成模型范式。与 Diffusion（多步去噪）不同，VAR 首先将图像通过 VQGAN 等 tokenizer 编码为离散的 visual token grid（如 256×256 图像编码为 16×16=256 个 token），再像 LLM 一样逐个预测 next visual token。VAR 天然与 LLM/多模态系统统一，且在大规模下保持较好的质量扩展趋势（scale-out/scale-up 潜力）。但其自回归 token-by-token 解码将生成延迟推高（256×256 需 256–4096 次串行 Transformer 调用），单张图常需 10–60 秒。

从算法pipeline角度拆解术语：
VAR 的标准推理 pipeline：
1. VQGAN encoder 将输入图像编码为离散 visual token grid（如 16×16=256 tokens）
2. 初始化：所有 visual token 位置置为 [MASK]
3. 自回归循环（每步）：
   a. 当前 token sequence（含已解码和 masked token）输入 generative Transformer
   b. Transformer 输出所有位置 logits
   c. 按预定顺序（如 raster scan）选取下一个待解码位置
   d. 对该位置做 Gumbel/argmax sampling 得到新 token
   e. 将该位置 mask 置为 False，token 写入 sequence
4. 所有 token 解码完成→VQGAN decoder 将 visual token grid 还原为像素图像
关键特点：每次 Transformer invocation 仅生成 1 个 token，串行步数等于 token 总数（256-N×N），attention 复杂度 O(N²) 随序列长度二次增长。

术语一般如何实现？如何使用？
VAR 模型通常使用基于 ViT/DeiT backbone 的 generative Transformer + VQGAN tokenizer。训练用 ImageNet + cross-entropy loss + AdamW optimizer。VAR-Turbo 论文使用 DeiT-based Transformer 在 ImageNet 上训练 500 epochs、4×V100、batch size 256。推理时通用平台为 GPU（如 V100），延迟瓶颈主要在串行调用次数和每次调用的 attention/FFN 计算量。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

