## I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - I²MoE 提出一种端到端的 MoE 框架，用于显式建模多模态交互（modality interaction），基于 Partial Information Decomposition (PID) 将模态交互分解为四种类型：唯一性(uniqueness for modality 1)、唯一性(uniqueness for modality 2)、协同(synergy)、冗余(redundancy)。核心设计：
    1. **Interaction Experts**：使用 n+2 个交互专家（n 个唯一性专家 + 1 个协同专家 + 1 个冗余专家），每个专家是一个完整的融合模型（fusion model + prediction head），通过弱监督的交互损失（interaction loss）学习专精于特定交互类型。
    2. **Perturbation-based Weak Supervision**：训练时使用随机向量替换某个模态的嵌入来模拟单模态场景（masked modality），对不同专家施加不同的交互损失：唯一性专家使用 Triplet Margin Loss 使完整模态输出接近未遮蔽目标模态、远离遮蔽模态；协同专家使用 Cosine Similarity 使完整模态输出与所有遮蔽输出最大化差异；冗余专家使用 Cosine Similarity 使完整模态输出与所有遮蔽输出最大化相似。
    3. **Re-weighting Model**：一个 MLP 根据所有模态嵌入输出 soft weights [w_uni1, w_uni2, ..., w_syn, w_red]，对每个交互专家的预测加权求和得到最终预测 ŷ = Σ w_i · ŷ_i。
    4. **Dual-objective Loss**：L_total = L_task + λ_int · L_int，其中 L_task 用带权融合输出计算，L_int 鼓励专家分化。
  - 实验比较：
    - Baselines：Early Fusion (EF)、Late Fusion (LF)、Low-Rank Multimodal Fusion (LRMF)、Multimodal Transformer (MulT)、InterpretCC、SwitchGate (Switch Transformer)、MoE++
    - 消融实验：(1) No-Interaction（去掉交互损失）、(2) Latent-Contrastive（在隐空间嵌入而非输出上施加交互损失）、(3) Simple-Weight（用全局可学习权重替代MLP重加权）、(4) Less-Forward（仅遮蔽2个模态而非全部）、(5) Synergy-Redundancy（仅保留协同和冗余专家）
    - 通用性实验：I²MoE 与 SwitchGate、InterpretCC、MoE++ 结合
    - 评估指标：Accuracy、AUROC、Micro F1、Macro F1
    - 可解释性评估：样本级局部解释（logits + weights 分解）、数据集级全局解释（权重分布统计）、人类评估（15人，300次评价）

- 硬件平台是什么，配置是什么。
  - **单卡 NVIDIA A100 GPU**（论文 Table 6 记录所有实验在单 A100 上运行）
  - 训练/推理时间及参数量详见表6

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - Modality-Specific Encoders：ADNI用3D-CNN(图像)+MLP(基因组/临床/生物样本)；MIMIC用LSTM(所有模态)；MOSI用GRU(视觉/音频/文本)；ENRICO用VGG11(截图+线框图)；IMDB用VGG16(图像)+Google Word2vec(语言)
    - Prediction Head：所有实验使用线性分类头
    - 参数规模：最小 673,935 (MOSI)，最大 6,696,728 (ADNI)
  - **数据集**：
    - **ADNI**：2,380样本，阿尔茨海默病三分类(CN/MCI/AD)，四模态(Image/Genetic/Clinical/Biospecimen)
    - **MIMIC-IV**：9,003患者记录，一年死亡率二分类，三模态(Lab/Notes/Codes)
    - **IMDB**：25,959电影，23类多标签分类，双模态(Image/Language)
    - **MOSI**：2,199 YouTube片段，情感分析回归[-3,3]→二分类，三模态(Vision/Audio/Text)
    - **ENRICO**：1,460 Android截图，20类UI设计分类，双模态(Screenshot/Wireframe)
  - **训练配置**：70%/15%/15% train/val/test split，batch_size=32，Adam优化器，lr=0.0001，训练30-50 epochs，3次随机种子取平均

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/Raina-Xin/I2MoE
  - 算法伪代码（I²MoE 训练流程，以两模态为例）：
    ```
    输入：modalities X_1, X_2, 标签 T
    编码器：E1, E2
    交互专家：F_uni1, F_uni2, F_syn, F_red (各含融合模型+预测头H_i)
    重加权模型：W (MLP)

    训练阶段：
    1. 编码模态: e1 = E1(X_1), e2 = E2(X_2), r ~ random vector
    2. 对每个专家 F_i, i ∈ {uni1,uni2,syn,red}:
       - 完整模态前向: ŷ_i^(12) = H_i(F_i(e1, e2))
       - 遮蔽模态2前向: ŷ_i^(-2) = H_i(F_i(e1, r))
       - 遮蔽模态1前向: ŷ_i^(-1) = H_i(F_i(r, e2))
       - 交互损失:
         * L_uni1 = TripletLoss(ŷ_uni1^(12), ŷ_uni1^(-2), ŷ_uni1^(-1))  # anchor, positive, negative
         * L_uni2 = TripletLoss(ŷ_uni2^(12), ŷ_uni2^(-1), ŷ_uni2^(-2))
         * L_syn = CosSim(norm(ŷ_syn^(12)), norm(ŷ_syn^(-1))) + CosSim(norm(ŷ_syn^(12)), norm(ŷ_syn^(-2)))  # 最小化相似度
         * L_red = [1 - CosSim(norm(ŷ_red^(12)), norm(ŷ_red^(-1)))] + [1 - CosSim(norm(ŷ_red^(12)), norm(ŷ_red^(-2)))]
    3. 计算权重: [w_uni1, w_uni2, w_syn, w_red] = softmax(W(e1, e2) / temperature)
    4. 融合预测: ŷ = Σ w_i · ŷ_i^(12)
    5. 任务损失: L_task = CrossEntropy(ŷ, T)
    6. 总损失: L_total = L_task + λ_int · (mean of all L_i)
    7. 反向传播更新所有参数

    推理阶段：
    1. e1 = E1(X_1), e2 = E2(X_2)
    2. 每个专家 F_i 仅计算完整模态前向: ŷ_i^(12) = H_i(F_i(e1, e2))
    3. 权重: [w_i] = softmax(W(e1, e2) / temperature)
    4. 输出: ŷ = Σ w_i · ŷ_i^(12)
    5. 可选：返回各专家预测 ŷ_i 和权重 w_i 用于可解释性分析
    ```
  - 扩展至 n 模态：交互专家数 = n + 2（n个唯一性专家 + 1个协同 + 1个冗余）。每个专家要进行 1+n 次前向（1次完整输入 + n次遮蔽单个模态）。唯一性专家 i 以完整输出为 anchor，遮蔽模态 i 为 negative，其余遮蔽为 positive。协同专家所有遮蔽为 negative。冗余专家所有遮蔽为 positive。
