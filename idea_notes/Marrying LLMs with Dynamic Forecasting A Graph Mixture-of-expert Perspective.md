## Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

- baseline方法是什么？
  - Baseline 为传统数据驱动的 GNN 动态系统建模方法（以 EGNN 为代表），包括 Linear、Dynamic（物理匀速模型）、GNN（Kipf & Welling 2017a）、Radial Field（Köhler et al. 2019）、EGNN（Satorras et al. 2022）、EGNO（Xu et al. 2024）。以 EGNN 为例说明全栈执行路径：
    - **算法层**：给定初始状态 X⁽⁰⁾ 和交互图 G，EGNN 通过 L 层等变消息传递迭代更新 node representation hᵢ 和 coordinate xᵢ。每层中，ϕ 网络学习边交互（输入 hⱼ, xⱼ, hᵢ, xᵢ → 输出 eᵢⱼ），AGG 聚合邻居边信息，COM^H 和 COM^X 分别更新 node/coordinate（Eq. 1-3）。最终通过 Decoder 输出预测状态 X̂⁽ᵗ⁾ = Decoder(H^L)。训练时最小化 MSE loss。模型在训练数据上学习动力学规律，环境变化的应对完全依赖训练数据分布覆盖。
    - **系统框架层**：标准 PyTorch 训练和推理，无特殊 Serving 框架修改。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：论文未明确说明（标准 PyTorch CUDA kernel，无自定义 kernel）。
    - **硬件架构层**：论文未明确说明（无特定硬件/模拟器要求）。
  - Baseline 核心缺陷：
    1. **分布偏移下泛化能力差**：EGNN/EGNO 等数据驱动方法在训练环境（如 Spring strength=1.0）下表现良好，但当环境参数变化（Hard: strength=1.10, Soft: strength=0.90, Temporal Shift: 不同起止状态）时，MSE 显著上升。原因是模型仅从数据中隐式学习动力学，无法利用显式的环境上下文信息来适应变化。
    2. **缺乏环境感知能力**：不同系统参数（如弹性系数 k、电荷量 q₁,q₂）产生不同的演化规律，但传统 GNN 将所有环境的输入统一处理，无法区分"当前处于什么环境"以及如何调整预测策略。
    3. **单一模型无法覆盖多模态动力学**：同一系统中不同环境可能对应本质上不同的动力学模式（如周期性振动的不同阶段），单个 GNN 模型难以同时拟合所有模式，出现模式平均（mode averaging）导致预测模糊。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - LEGO 通过"LLM as context-aware routing function + Graph MoE with diversity enhancement"三层设计解决上述缺陷。全栈执行路径（以 Spring 系统，K=5 experts，Llama 3.1 8B 为例）：
    - **算法层 — Hierarchical Prompt Engineering**：
      1. 从三个层次提取环境上下文转化为文本：(a) System level：系统参数和背景描述，如 "The force on the balls are significant, and forces between them result in strong accelerations" + 弹簧系数 k 等数值；(b) Object level：各物体初始位置 (x,y,z) 和速度 (vx,vy,vz) 向量，数值 digit 作为 token（遵循 Gruver et al. 2024）；(c) Edge level：边连接关系，如 "ball 2 connects ball 0, ball 1, ball 3"。三层 prompt 共同构成对当前环境的完整文本化描述。
    - **算法层 — Graph Mixture-of-Expert with Diversity Enhancement**：
      1. K 个独立的 EGNN experts（同架构不同参数 θ¹...θᴷ），每个 expert 对输入并行预测，生成 K 个候选状态 X̂⁽ᵗ⁾,¹...X̂⁽ᵗ⁾,ᴷ（Eq. 6：通过 one-hot routing 获得各 expert 独立预测）。
      2. Diversity-enhanced contrastive loss（Eq. 9-10）：最大化同 expert 内表征相似度、最小化不同 expert 间表征相似度，确保各 expert 学习不同的动力学模式（如某些 expert 擅长高能量模式、某些擅长低能量模式）。
    - **算法层 — LLM Judge for Context-Aware Routing**：
      1. 将 hierarchical prompt + K 个 candidate predictions 的描述送入预训练 LLM（Llama 3.1 8B），LLM 基于环境上下文推理选择最合适的 expert，而非直接生成数值预测（避免 LLM 在复杂张量生成上的不可靠性）。
      2. Label smoothing（Eq. 7）：选中 expert 权重 α，其余 (1-α)/(K-1)，避免错误路由的硬性惩罚积累。
      3. 交替优化（Algorithm 1）：LLM routing weights 每隔若干 epoch 更新一次（减少 LLM API 调用成本），graph expert 参数通过梯度下降持续优化。LLM 推理只需文本理解能力，无需微调。
    - **系统框架层**：标准 PyTorch 训练，LLM 通过 API 调用（论文未说明具体服务框架）。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：论文未明确说明。
    - **硬件架构层**：论文未明确说明（无特定硬件/模拟器）。
  - 对比 baseline 的改进映射：
    - **分布偏移下泛化能力差 → LLM Judge 利用环境上下文自适应路由**：传统 EGNN 从数据中隐式学习、无法利用显式环境描述 → LEGO 将环境参数、物体状态、连接关系转化为三层 prompt，LLM 以 zero-shot 方式理解环境变化并选择最匹配的 expert。在 Charged 数据集上，EGNN+LEGO 相比 EGNN 在 Hard/Soft/Temporal Shift 三种 OOD 场景下分别取得 25.4%/27.3%/16.0% 的 MSE 降低（Table 1），验证了 LLM 对环境上下文的有效利用。
    - **缺乏环境感知能力 → Hierarchical Prompt 提供三层环境描述**：Ablation（Table 4）验证了多层 prompt 的必要性——V1（仅 system level）MSE=0.761，V2（system+edge）MSE=0.735，V3（完整三层）MSE=0.728，每增加一层信息均带来一致的性能提升。Edge level 信息（连接关系文本化）的贡献尤为显著（V1→V2 降幅 > V2→V3）。
    - **单一模型无法覆盖多模态动力学 → Graph MoE + Diversity Loss**：多个 experts 通过 contrastive diversity loss 被迫学习互补的动力学模式。LLM 根据环境选择合适的 expert，实现"不同环境用不同 expert"的 specialization。参数实验（Figure 3b）显示 K=5 时性能最佳，过少（K=3）覆盖不足、过多（K=15/20）LLM Judge 判断困难导致性能下降。
    - **LLM 直接生成不可靠 → LLM-as-Judge 而非 LLM-as-Predictor**：Table 5 显示 LLM Forecasting（直接生成预测）的 MSE=6.4201 远高于 LEGO 的 0.0072（约 890× 差距），且 LLM Forecasting 推理时间更长（1.270s vs 0.438s per sample），验证了"判断优于生成"的设计哲学。
