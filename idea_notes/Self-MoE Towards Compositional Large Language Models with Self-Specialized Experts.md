## Self-MoE Towards Compositional Large Language Models with Self-Specialized Experts

- baseline方法是什么？
  Baseline方法：(1) **单体LLM微调（Monolithic Fine-tuning）**：直接在目标领域数据上对整个模型进行全参数或参数高效微调（如LoRA），得到一个在特定领域增强的模型。然而这种方式存在灾难性遗忘问题——在目标领域上的提升往往以牺牲非目标领域性能为代价（如Knowledge Self-Spec在MMLU上从58.4→64.0，但在BBH上从56.1→41.7）。(2) **Instance Merging（多任务微调）**：将所有领域的合成数据合并后直接微调单一模型（LoRA），得到一个多任务模型，但缺乏动态适应能力，无法根据具体输入选择最合适的专家知识。(3) **Weight Merging（TIES/DARE）**：将多个独立训练的专家模型通过权重平均或参数融合合并为单一模型，但合并后的模型参数是静态的，不同专家之间可能产生参数干扰（interference），且丧失了语义可解释性。(4) **传统MoE预训练方法（Switch Transformer, Mixtral, BTX）**：使用FFN层作为expert，随expert数量增长参数总量线性增加（Mixtral 8x7B总参数47B），且expert在预训练中隐式学习，缺少显式的语义区分。需要大量计算资源（BTX需要900 GPU天）。
  全栈执行例子（Baseline: Instance Merging方案，基于Gemma-7B，单A100-80GB）：
  - **算法Pipeline层**：收集所有目标领域的合成数据 D_knowledge ∪ D_reasoning ∪ D_math ∪ D_coding（共20K样本），用LoRA(rank=8)直接微调Gemma-7B，得到单一适配器 ΔΘ_merged，推理时对所有输入统一使用 h = θ_0 x + Δθ_B Δθ_A x，无动态路由选择。
  - **系统框架层**：使用HuggingFace PEFT加载base模型+单一LoRA adapter，使用Alpaca prompt template进行推理，evaluation通过LM Evaluation Harness统一调用。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法Self-MoE构建了**MiXSE（MiXture of Self-specialized Experts）**——一个组合式模块化系统。通过两阶段设计解决baseline的缺陷：
  **(1) 自专业化（Self-Specialization）解决标注数据瓶颈**：不依赖昂贵的人工标注数据或更强的教师模型，而是利用base LLM自身的生成能力，通过Instruction Brainstorming→Response Generation的自我循环，从仅100条种子数据自动扩展出每领域5K条高质量合成训练数据。这解决了MOLE依赖人工标注、PHATGOOSE依赖预训练外部专家、GLAN依赖GPT-4蒸馏等baseline对数据的强假设。
  **(2) 模块化LoRA专家+动态路由解决遗忘与静态融合问题**：每个领域独立训练轻量LoRA专家模块（<0.3%参数），保持base LLM Θ_0 不变，在推理时通过训练好的路由层 θ_r 对每个token动态计算 top-k 软权重选择最相关专家：h = θ_0 x + Σ α_i Δθ_i x。路由器的自学习仅需各领域合成数据的聚合进行轻量训练（冻结所有专家）。这直接解决了：Instance Merging/TIES/DARE的静态参数无法动态适应不同任务的问题；单体微调的性能权衡（遗忘）问题；传统MoE（如Mixtral用FFN expert）参数膨胀问题（Self-MoE总参数仅增加~1% vs Mixtral 8x7B的47B）。
  **(3) 语义专家的显式区分增强可解释性**：每个expert对应明确的语义领域（知识/推理/数学/编程），路由权重可视化验证了路由器正确将任务分配到对应专家，且能跨领域协同（如推理专家参与数学和编程任务），解释了MiXSE为何超越所有单独专家。
  全栈执行例子（Self-MoE/MiXSE，基于Gemma-7B，单A100-80GB）：
  - **算法Pipeline层**：输入token序列，每层LoRA处：x → θ_r计算4个expert的路由logits → softmax → top-k mask → 加权组合Δθ_i的输出 → 与base θ_0结果相加。路由决策是token级别的（per-token routing），top-1配置下仅激活一个expert（活跃参数7B+0.3%），top-2激活两个。训练时仅优化θ_r（线性层，可忽略参数量），冻结所有ΔΘ_i保持语义区分。
  - **系统框架层**：使用HuggingFace PEFT管理多个LoRA adapter的加载与切换，XLoRA实现MoE-compatible的LoRA路由机制（在每个LoRA层前插入路由层），训练使用Alpaca prompt template，评估通过LM Evaluation Harness调用模型生成并在benchmark上计算准确率/pass@k。整个MiXSE训练仅需约1 GPU天（vs BTX的900 GPU天）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。
