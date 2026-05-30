## Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models

- baseline方法是什么？
  Baseline方法是基于attention map的视觉token剪枝方法（如FastV、SparseVLM、VisionZip），通过在VLM早期层利用attention scores选择并保留重要visual tokens来加速prefilling阶段，但存在两个主要缺陷：(1) 早期层attention信号对任务不敏感，导致剪枝后精度大幅下降；(2) KV-cache机制和FFN block使decode阶段加速有限，长response生成时速度瓶颈严重。

  Baseline（以FastV on LLaVA-1.5-7B，K=2, R̄=64为例）全栈执行例子：
  - 算法层：Image → CLIP Vision Encoder → 576 visual tokens → 拼接text tokens → LLaVA-1.5-7B 前2层处理 → 取第2层attention map，计算text-to-visual attention scores → 选择top-R(=41)最重要的visual tokens → 丢弃其余 → 剩余30层处理pruned序列 → KV-cache缓存 → decode阶段逐token自回归生成。88.9% pruning ratio下RelAcc仅77.0%（FastV）。
  - 系统框架层：HuggingFace Transformers推理。剪枝仅在prefilling阶段生效，decode阶段FFN计算量不变，KV-cache使SA加速有限。
  - 编译框架层：论文未明确说明
  - kernel调度层：标准FlashAttention（causal mask），decode阶段每步只处理1个token，GPU利用率极低
  - 硬件架构层：8×NVIDIA A100 GPU服务器

  Baseline的缺陷：
  1. **早期attention信号质量差**：第2层attention对multimodal语义理解不充分，选出的visual tokens与prompt无关（论文Fig.2可视化：D=2选出的token在不同prompt间几乎相同），大量有用visual信息被错误丢弃。更深层attention(D=18)虽能提供更精准信号，但若在深层pruning则前面的计算冗余已产生。
  2. **Decode阶段加速有限**：KV-cache机制下SA block加速效果被削弱，FFN block完全无加速。当response length≥32 tokens时，prefilling时间可忽略，但decode时间线性增长（论文Fig.3）。FastV仅在prefilling阶段加速，长response(MM-Vet, S̄≈100)下RelSpd仅~104%。
  3. **剪枝信号未专门优化**：attention map的token选择能力仅作为next-token prediction训练的副产品出现，未针对剪枝任务直接优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：TwigVLM/TwigVLM++通过三项核心设计解决上述缺陷：

  (1) **Twig-guided Token Pruning (TTP)** → 解决早期attention信号差。在base VLM第K层后附加T层twig block（初始化为VLM第K+1至K+T层权重），使用twig最后一层的attention map指导token剪枝。由于twig最后一层靠近prediction head（距离loss函数更近），其attention对multimodal关系理解更精准（Fig.2证实attention quality随depth增加而提升）。同时twig仅需轻量post-training（~10% base VLM训练时间），不修改base VLM权重。配合FinalWipe策略在Kf层后移除所有visual tokens（因深层visual tokens贡献极小），在固定R̄下允许更大的R，进一步提升精度。

  (2) **Self-Speculative Decoding (SSD)** → 解决decode阶段加速不足。利用TwigVLM天然的一体两用架构：浅层子网络Ms（前K层+twig）作draft model、深层子网络Mb（完整base VLM）作target model。Draft自回归生成5个候选tokens → target并行验证并接受匹配tokens。关键优势：(a) draft和target共享前K层KV-cache，减少冗余计算；(b) draft model极浅(K+T≪L)，生成开销低；(c) 并行验证充分利用GPU并行能力。长response场景(MM-Vet)RelSpd达154%（vs FastV 104%）。TwigVLM++的Tree-based SSD进一步通过构建token tree（E=10, K=10, D=4）增加每次验证的候选路径覆盖，RelSpd达~197%。

  (3) **Multi-head Twig + RL-based Pruning Optimization (TwigVLM++)** → 解决剪枝信号未专门优化。解耦D-Head（next-token prediction）和P-Head（专用于token重要性评分）。Stage-1通过PredKL蒸馏（teacher=base VLM, student=shallow VLM）和AttnKL蒸馏（teacher=deep layer attention, student=P-Head score）提供额外监督。Stage-2用GRPO式RL直接最大化pruned输入下的参考答案log-probability：P-Head产生token重要性分布 → 无放回采样R个位置得action → reward = pruned输入生成参考答案的mean log-prob → group-level advantage归一化 → 纯on-policy policy gradient更新。Dynamic pruning ratio schedule使单个模型支持多种pruning ratio。88.9% pruning下LLaVA-1.5-7B RelAcc从96.0%提升到97.7%。

  对比baseline的全栈执行例子（TwigVLM++ on LLaVA-1.5-7B, K=2, T=3, Kf=24, R̄=64）：
  - 算法层：Image → CLIP Vision Encoder → 576 visual tokens → base VLM前2层 → twig block 3层（D-Head预测next token + P-Head计算token importance score s via gated attention Eq.7）→ TTP按s选择top R=41 visual tokens → FinalWipe在24层移除所有visual tokens → 剩余深层的base VLM处理pruned序列 → 同时twig作为draft用tree-based方式构建token tree → base VLM用tree attention并行验证多条路径 → 从根遍历接受匹配tokens → 追加bonus token → 迭代至生成EOS。
  - 系统框架层：两阶段训练：Stage-1用LLaVA-665K (665K samples) + L_NTP + α·L_PredKL + γ·L_AttnKL，仅更新twig block (~10 GPU hours)；Stage-2用50K SFT samples + GRPO式RL，仅更新P-Head参数。推理时draft/target共享前K层KV-cache，tree attention使用topology-aware causal mask。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention v2.3.2。Tree-based SSD的tree attention用topology-aware causal mask替代标准causal mask，在单次前向中处理整个token tree（最多60个candidate nodes）。标准SSD每step验证1条sequence、接受~3 tokens；tree-based SSD每step验证K·E^D条路径、接受更多tokens。
  - 硬件架构层：8×NVIDIA A100 GPU服务器。Prefilling阶段TTP剪枝88.9% visual tokens，compute减少显著；Decoding阶段tree-based SSD每step处理batch of tree nodes（而非单token），GPU利用率大幅提升。长response (MM-Vet) RelSpd达~197%，短response (TextVQA) RelSpd达~139%。
