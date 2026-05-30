## I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts

- baseline方法是什么？
  - **Vanilla Multimodal Fusion**（如 Early Fusion, Late Fusion, MulT, LRMF）：使用统一的融合参数处理所有模态交互。流程为：各模态编码器 E_i 分别编码输入 → 融合方法 F 将所有隐嵌入融合为单一向量 x = F(e1, e2, ..., en) → 预测头 H(x) = ŷ。核心缺陷是使用相同参数建模所有交互类型（唯一性/协同/冗余），无法区分"图像独有的视觉线索"、"文本独有的语义信息"、"两者协同产生的新信息"、"两者共享的冗余信息"。在 Figure 1 的 IMDB 电影分类例子中，Horror 依赖图像唯一性、Romance 依赖语言唯一性、Fantasy 依赖冗余信息、Drama 依赖协同信息——vanilla fusion 对这些不同交互一视同仁。
  - **SwitchGate & MoE++**：在 MulT 中将 MLP 层替换为稀疏 MoE 层，但 MoE routing 仅做 conditional computation（负载均衡），不鼓励专家按交互类型分化。本质上仍是 implicit interaction modeling。
  - **MMoE (Yu et al. 2024)**：唯一显式建模交互类型的 MoE 方法，但将交互建模作为预处理步骤（非端到端），限制灵活性和可解释性。
  - 全栈执行例子（Baseline MulT 在 IMDB 上做电影分类）：
    - **模型推理/训练算法层**：图像用 VGG16 提取特征，文本用 Google Word2vec 提取特征 → VGG11 编码器处理 → MulT 多模态 Transformer（cross-modal attention）融合 → 线性分类头输出 23 类 logits → CrossEntropy 损失。所有模态交互通过同一套 attention 参数处理。
    - **系统框架层**：PyTorch + torchvision，标准训练脚本，单卡 A100。论文未明确说明框架级定制。
    - **编译框架层**：论文未明确说明。PyTorch eager mode，cuDNN/cuBLAS 后端。
    - **kernel 调度层**：论文未明确说明。标准 cuBLAS GEMM（attention 矩阵乘法）+ PyTorch autograd。
    - **硬件架构层**：单卡 NVIDIA A100 GPU。论文未明确说明 GPU 架构级优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **I²MoE 方法**：基于 PID 框架设计端到端 MoE，包含三种核心机制：
    1. **Interaction Experts 分化**：用 n+2 个独立融合模型（含各自参数和预测头）替代单一融合模块。每个专家通过不同的弱监督交互损失（perturbation-based）被迫专精——唯一性专家通过 Triplet Margin Loss 学习"去除了某模态就无法正确预测"的信息；协同专家通过最小化与遮蔽输入的 Cosine Similarity 学习"必须联合两模态才有的"信息；冗余专家通过最大化与遮蔽输入的 Cosine Similarity 学习"任一模态都能做到的"信息。
    2. **Adaptive Reweighting**：MLP 重加权模型根据输入样本的模态特征动态分配 wi，替代均匀融合。这使得模型对不同样本可自适应选择交互策略（某样本偏重图像唯一性、另一样本偏重文本唯一性或协同）。
    3. **端到端可解释性**：重加权模型输出 w_i 天然提供样本级局部解释（每个交互的贡献度）+ 测试集 w_i 统计提供数据集级全局解释（整体交互趋势）。
  - 对应解决 Baseline 缺陷：
    - Baseline 用同一参数建模所有交互 → I²MoE 用 4 个独立专家（各自参数+各自损失）显式分化。
    - Baseline 无法区分交互类型 → I²MoE 通过 perturbation-based 弱监督为每种交互提供明确训练信号（TripletLoss/CosSim/MSE）。
    - Baseline 缺乏可解释性 → I²MoE 的 w_i 天然提供 local/global 两层可解释性，人类评估 70.4% 正面。
  - 全栈执行例子（I²MoE-MulT 在 IMDB 上做电影分类）：
    - **模型推理/训练算法层**：图像用 VGG16 提取特征，文本用 Google Word2vec 提取特征 → VGG11 编码器处理 → 4 个 MulT 交互专家（F_uni_img, F_uni_lang, F_syn, F_red）各做 cross-modal attention 融合 + 线性预测头输出各 23 类 logits → 训练时每个专家额外做 2 次 masked 前向（遮蔽图像/遮蔽文本，用随机向量 r 替换嵌入）→ 计算交互损失 → MLP 重加权模型输出 w_i = softmax(MLP(e_img, e_lang) / temperature) → ŷ = Σ w_i · ŷ_i。推断时仅 1 次完整模态前向 + 1 次重加权。任务损失 + λ_int · 交互损失联合优化。
    - **系统框架层**：PyTorch + torchvision。训练脚本在 https://github.com/Raina-Xin/I2MoE/tree/main/scripts/train_scripts。论文未明确说明框架级定制。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **kernel 调度层**：论文未明确说明。标准 PyTorch 算子，每次训练需 (n_experts) × (n_modalities+1) 次 forward pass，额外计算开销约为 MulT 的 (n_modalities+2) 倍。训练时间开销：IMDB 上 MulT 3.62s/epoch → I²MoE-MulT 44.20s/epoch；推断开销：MulT 0.53s → I²MoE-MulT 3.23s。
    - **硬件架构层**：单卡 NVIDIA A100 GPU。论文未明确说明 GPU 架构级优化。
