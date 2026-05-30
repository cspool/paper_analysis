## Steering MoE LLMs via Expert (De)Activation

- baseline方法是什么？
  Baseline 是标准的 MoE LLM 推理方式：不对 router 做任何干预，模型的 top-K routing 完全由训练好的 router weights 决定。对于给定的输入 prompt，每层的 router 计算 z = W_r h，softmax 后取 top-K experts，加权求和输出。模型的行为（faithfulness、safety）完全由原始训练和 RLHF alignment 决定，推理时无法控制或调节。
  全栈执行例子（Baseline: GPT-OSS-120B 标准推理，GPU 推理）：
  - 算法pipeline层：输入 token h ∈ R^d → router 计算 z = W_r h → softmax 得概率 p → TopK(p, k=4) 选择 4/32 experts → Expert_i(h) 加权求和输出。所有 expert 的路由完全按训练好的 router weights 决定。在 RAG faithfulness 场景中，即使给定了 document context，模型可能仍然依赖 parametric knowledge（而非 document content），产生幻觉。在 safety 场景中，尽管 RLHF alignment 训练了 refusal 行为，但 alignment 集中在部分 experts 的稀疏子网络上——unsafe routing pathways 仍然存在，一旦被精心设计的 jailbreak 触发，模型仍会输出 unsafe 内容。
  - 系统框架层：标准的 HuggingFace transformers 推理，无 serving 框架修改。每个 token 前向通过所有 MoE layers，无额外控制。论文未修改任何 serving 框架。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：论文未明确说明硬件平台。使用标准 NVIDIA GPU 集群加载 HuggingFace 模型权重进行推理。
  Baseline 的核心缺陷：(a) MoE router 被当作纯粹的计算分配机制，忽视了其作为行为控制接口的潜力——router 的 expert 选择实际编码了 behavior-specific signals；(b) RLHF alignment 训练的安全行为并非存在于所有 expert routing paths 中，而是集中在稀疏的"safe expert"子网络上——unsafe routing pathways 在 aligned 模型中依然存在，形成"alignment faking"状态（alignment concentrated in a subset of experts, neglecting alternate routing paths that can catastrophically bypass alignment when triggered）；(c) 无法在推理时动态控制模型行为——当需要对 faithfulness 或 safety 做精细调节时，baseline 无接口可操作。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 SteerMoE 将 MoE router 重新解释为可控和可解释的行为调制机制（controllable and interpretable mechanism），而不仅仅是计算分配的工具。通过三个关键设计解决 baseline 缺陷：
  (1) Paired-Example Routing-Difference Detection：利用对比行为对（safe/unsafe, faithful/unfaithful）的 expert activation 差异，通过 Risk Difference (RD) 量化每个 expert 的行为关联强度。这直接解决了 baseline 中将 router 当作黑盒的问题——SteerMoE 揭示了哪些 experts 编码了特定的行为信号。
  (2) Inference-Time Expert (De)Activation：在推理时，通过 hook router 的 log-softmax scores，将行为关联 experts 的分数调整至 s_max + ε（激活）或 s_min - ε（去激活），然后重新 softmax 归一化执行 top-K routing。无需修改模型权重、无需额外训练。这解决了 baseline 中无法在推理时控制行为的问题。
  (3) Soft Steering Design：仅将目标 expert 的 logit 推至 max/min + ε，而非极端值（±∞），保留多 expert 加权平均结构，避免 MoE topology 坍缩。这保证了 steering 后的模型质量稳定（fluency 几乎不变）。

  全栈执行例子（论文方法：SteerMoE 在 GPT-OSS-120B 上 unsafe steering，GPU 推理）：
  - 算法pipeline层：
    (a) Detection Phase：使用 BeaverTails 构造对比对——x^(1) = "User: {Prompt} Assistant: I'm sorry, but I can't assist with that."（safe refusal），x^(2) = "User: {Prompt} Assistant: {Unsafe Response}"（unsafe compliance）。对每个 MoE layer ℓ 和 expert i，统计激活率 p_{ℓ,i}^{(1)} 和 p_{ℓ,i}^{(2)}，计算 Δ_{ℓ,i} = p_{ℓ,i}^{(1)} - p_{ℓ,i}^{(2)}。负 Δ 表示该 expert 与 unsafe 行为关联更强。top-K 负 Δ experts 作为 A^-（去激活集合，即"safe experts"被去激活），top-K 正 Δ experts 作为 A^+（激活集合，即"unsafe experts"被激活）。实际配置（Table A.2）：GPT-OSS-120B 的 Unsafe steering 去激活 100 个 experts、激活 0 个（仅去激活 safe experts 即可释放 unsafe 行为）。
    (b) Steering Phase：对每个 token h，router 输出 z ∈ R^32（32 experts per layer），计算 s = log softmax(z) ∈ R^32。对于去激活集合 A^- 中的 100 个 experts（跨层），在各自所在层执行 s_e ← s_min - 0.01，使这些 originally "safe" experts 的概率降至最低；安全相关 experts 被从路由中排除后，其余（unsafe）experts 自然承接路由。然后 p = softmax(s)，TopK(p, 4) 选择 4 experts，output = Σ p̃_i · Expert_i(h)。
    (c) 效果：GPT-OSS-120B 在 AdvBench 上从 100% safe（baseline 完全拒绝所有 harmful prompts）降至 0% safe（与 AIM jailbreak 结合后完全被攻破）。所有 jailbreak 方法均被 bypass。
  - 系统框架层：基于 HuggingFace transformers 的 hook 机制修改 router logits。具体实现：在 MoE layer 的 router forward 中插入 hook——计算 log-softmax 后应用公式 7/8，然后继续标准 MoE forward。无需修改 serving 框架。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：论文未明确说明。

  对比 baseline 的关键改进：
  - Baseline 将 router 视为固定的计算分配 → SteerMoE 将 router 视为可控的行为接口，通过对比激活统计发现行为-specific routing pathways
  - Baseline 的 safety alignment 仅存在于部分 experts 的稀疏子网络 → SteerMoE 揭示并利用了这一"alignment faking"漏洞——仅去激活 100/4608 experts 即能使完全 aligned 的模型 100% 被 jailbreak
  - Baseline 无法动态控制 faithfulness/safety → SteerMoE 提供了推理时可调的双向控制（既可增强安全又可削弱安全）
  - SteerMoE 的跨语言泛化（英文检测对发现的 safety experts 在意大利语/泰语上同样有效）表明 behavior-linked experts 编码的是行为本身而非语言特征
