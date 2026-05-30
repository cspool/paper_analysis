## MoH: Multi-Head Attention as Mixture-of-Head Attention

- baseline方法是什么？
  Baseline 是标准的 multi-head attention（MHA），将输入投影到 h 个低维子空间，每个 head 独立做 scaled dot-product attention，最后将所有 head 输出求和（summation form）：MultiHead(X, X') = Σ_{i=1}^{h} H^i · W_O^i。所有 head 对所有 token 均等激活，不存在 token 级别的选择性。

  全栈执行例子（以 ViT 图像分类为例）：
  - **算法pipeline**：一张 224×224 图像 → patch embedding 为 T 个 token (T×d_in) → 每层 multi-head attention：所有 h 个 head 并行计算 Q_i/K_i/V_i → Scaled Dot-Product Attention → 求和拼接 → 输出投影 → FFN → 下一层。标准 MHA 中所有 h head 对所有 T token 全激活。
  - **系统框架**：训练时使用 TransNeXt 框架在 8 GPUs 上做数据并行；推理时标准 PyTorch 前向传播，无特殊调度。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 cuBLAS batch GEMM 实现 multi-head attention，所有 head 的 QKV 投影和 attention 计算均为 dense 矩阵乘法。论文 Inference Time 实验（Tab.7）中，head dim=64, seq=256 时 MHA 耗时 0.360ms，seq=512 时 1.376ms。
  - **硬件架构**：论文未明确说明具体 GPU 型号。
  核心缺陷：（1）**注意力头冗余**——多个 head 可能学习相似特征，许多 head 可被剪枝而不影响精度（Voita et al. 2019, Michel et al. 2019）；（2）**token 级无差别计算**——所有 token 经过所有 head，但不同 token 可能需要不同 head 的关注模式，造成计算浪费；（3）**求和缺乏灵活性**——标准 MHA 对所有 head 等权求和，无法根据 token 内容动态调整各 head 贡献。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 MoH（Mixture-of-Head Attention），将 attention head 视为 MoE expert，引入 router 为每个 token 动态选择 Top-K head。核心设计：（1）**Heads as Experts + Router**——通过可学习的 W_s/W_r 将 token routing 到不同的 attention head，仅激活 Top-K 个 routed head（+ 所有 shared head），显著减少激活 head 数（50%~90%）；（2）**Shared Heads**——前 h_s 个 head 始终激活，捕获通用知识（如语法规则），剩余 head 动态路由处理 token-specific 信息，减少冗余；（3）**Two-Stage Routing**——W_h 产生 α_1/α_2 系数动态平衡 shared 和 routed head 权重，实现加权求和替代标准等权求和；（4）**Load Balance Loss**——防止 routing collapse，确保所有 routed head 得到充分训练。

  全栈执行例子（以 MoH-ViT 为例）：
  - **算法pipeline**：一张 224×224 图像 → patch embedding 为 T 个 token → 每层 MoH attention：token x_t 输入 router → W_s/W_r 计算 routing score → shared head 全部激活 + Top-K routed head 激活 → 仅激活的 head 计算 Q_i/K_i/V_i 和 Scaled Dot-Product Attention → 以 g_i 加权求和 → 输出投影 → FFN → 下一层。激活 head 的预算在各层不均匀分布——浅层激活少、深层激活多。
  - **系统框架**：训练基于 TransNeXt（ViT）、DiT（扩散模型）、Megatron（LLM）框架；LLaMA3-8B 的 continue-tuning 分两阶段——第一阶段 300B tokens 做数据分布适配，第二阶段 100B tokens 转为 MoH 模型（含参数无关 router + straight-through estimator 量化 routing score 保持输出分布稳定）。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：论文 Inference Time 实验（Tab.7）展示：将 Q/K/V 特征通过 router mask 转为稀疏矩阵，用稀疏矩阵乘法替代 dense 矩阵乘法。head dim=64, seq=256 时：90%激活 0.352ms，75%激活 0.321ms，50%激活 0.225ms（分别比 MHA 的 0.360ms 快 2.2%/10.8%/37.5%）。seq=512 时加速更显著（50%激活 0.863ms vs MHA 1.376ms，快 37.3%）。
  - **硬件架构**：论文未明确说明。

  关键效果：MoH-ViT-B 用 75% head 达 84.9% Top-1 Acc（TransNeXt 100% head 84.8%）；MoH-DiT-XL/2 用 90% head 达 FID 2.94（DiT-XL/2 100% head FID 3.22）；MoH-LLaMA3-8B 用 75% head 在 14 benchmark 上平均 64.0%（LLaMA3-8B 61.6%），仅需 100B continue-tuning tokens。
