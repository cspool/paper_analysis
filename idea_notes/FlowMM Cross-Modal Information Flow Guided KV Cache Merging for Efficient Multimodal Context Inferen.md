## FlowMM Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference

- baseline方法是什么？
  **现有KV cache压缩方法（eviction-based和merging-based）直接应用于多模态场景**：主要包括两类——(1) Eviction方法（StreamingLLM、H2O、D2O），基于attention scores评估token重要性后丢弃低重要性token，但不可逆的信息丢失导致context fragmentation和hallucination（Jiang et al., 2025）；(2) Merging方法（KVMerge）将所有eviction候选token合并到保留token中以保留更丰富的上下文信息，以及multimodal-specific方法（LOOK-M）针对多模态设计的KV cache合并。但这些方法均为text-based或仅简单适配multimodal，未能考虑MLLM中跨模态信息流的层间差异性——浅层以intra-modal交互为主（提取低层特征），深层以inter-modal交互为主（跨模态融合和高层语义抽象）。对所有层使用统一合并策略导致：浅层跨模态合并造成模态信息混淆（modality information confusion），深层仅做模态内合并导致跨模态语义融合不充分（insufficient cross-modal fusion）。

  全栈执行例子（以Qwen2.5-VL-7B处理ALFRED任务的一个多模态样本，单A100 GPU，使用KVMerge baseline）：
  - **模型推理算法层**：MLLM将visual tokens（来自ViT编码器的patch embeddings）和text tokens拼接为输入序列X = {X_1^T, X_1^I, ..., X_N^T, X_M^I} ∈ R^{L_p×d}。在prompt encoding阶段计算K_0 = XW^K, V_0 = XW^V（公式2）。生成阶段逐token更新KV cache: K_t = [K_{t-1}, k_t], V_t = [V_{t-1}, v_t]（公式3）。Attention输出: o_t = Softmax(q_t K_t^T / √d) V_t（公式4）。KVMerge baseline在所有L层使用统一策略——基于token相似度将non-pivot tokens合并到pivot tokens，不论该层是以intra-modal还是inter-modal交互为主。
  - **系统框架层**：论文未明确说明使用特定serving框架。KV cache合并算法作为HuggingFace Transformers推理pipeline的插件式KV cache后处理模块，在每层attention计算后对KV cache进行压缩。
  - **编译框架层**：论文未明确说明。使用标准PyTorch推理路径，KV cache合并操作为纯PyTorch tensor操作（cosine similarity + weighted averaging）。
  - **kernel调度层**：论文未明确说明。KV cache合并操作（token similarity计算、top-k selection、weighted averaging）在PyTorch层通过GPU kernel执行，无自定义CUDA kernel。
  - **硬件架构层**：单张NVIDIA A100 80GB GPU。Baseline KVMerge在20% cache budget下GPU memory从2.06 GiB降至0.44 GiB，但ALFRED accuracy从36.92%降至27.94%（Qwen2.5-VL-7B, Table 1），说明统一合并策略在multimodal场景下造成显著信息损失。

  Baseline缺陷：
  - (a) **忽略跨模态信息流的层间差异性**：MLLM的浅层（layers 1-N/2）以intra-modal attention为主（visual→visual, text→text），深层（layers N/2+1到L）以inter-modal attention为主（visual↔text）。统一合并策略在浅层做跨模态合并导致modal information confusion（Figure 3b: misaligned merging仅达full cache的~50% accuracy），在深层做模态内合并导致cross-modal fusion不充分。
  - (b) **无token敏感度保护**：所有token在合并决策中被平等对待（仅基于相似度），导致高敏感度task-critical token被合并后信息被稀释（dilution），尤其在TextNeedle等需要精确保留特定token信息的任务中表现明显。
  - (c) **合并策略与任务无关**：统一合并策略不考虑当前推理任务的语义需求，仅基于底层token表示相似度决策。
  - (d) **多模态分布偏移**：visual tokens和text tokens存在显著的distributional divergence，indiscriminate merging（不区分模态的合并）可能导致语义扭曲（semantic distortion）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlowMM：Cross-Modal Information Flow Guided KV Cache Merging + Sensitivity-Adaptive Token Matching**。核心设计一（解决缺陷a,c）：通过分析每层的cross-modal attention interaction ratio ρ^l（公式6-7），动态判断每层应以intra-modal还是inter-modal方式合并——浅层（ρ^l<θ）做intra-modal保留各模态的低层特征完整性，深层（ρ^l≥θ）做cross-modal促进跨模态语义融合。核心设计二（解决缺陷b）：在token匹配时引入sensitivity threshold τ（公式10），仅允许将non-pivot tokens合并到低敏感度的pivot tokens（I_j ≤ τ），保护高敏感度task-critical tokens不被合并稀释。核心设计三（解决缺陷d）：intra-modal merging策略将visual和text tokens分别聚类合并，避免跨模态分布偏移导致的语义混淆。

  全栈执行例子（同样Qwen2.5-VL-7B, ALFRED任务, FlowMM, cache budget=20%, θ=0.2, τ=0.3, 单A100 GPU）：
  - **模型推理算法层**：同一MLLM前向推理流程。FlowMM在每层attention计算后进行两阶段处理：(Phase 1 离线) 在校准样本上计算每层ρ^l确定merge_strategy——浅层4层ρ^l<0.2→intra-modal merging，深层28层ρ^l≥0.2→cross-modal merging；(Phase 2 在线) 每层KV cache merge时：(1) 用proxy tokens（最后~16个prompt tokens）聚合计算每个token的重要性I(i) = Σ_{j∈P} α_{j→i}；(2) 选top-20% tokens作为pivot set K^p；(3) 对每个non-pivot token i，在K^p中找cosine similarity最高且sensitivity I_j ≤ τ的pivot j，将(K_i, V_i)加权平均合并到(K_j, V_j)；(4) 搜索空间受merge_strategy约束——intra-modal层仅在同类模态token内搜索。ALFRED accuracy从KVMerge的27.94%提升至35.43%（Table 1）。
  - **系统框架层**：论文未明确说明使用特定serving框架。FlowMM在HuggingFace Transformers推理pipeline中作为KV cache后处理插件，替换原有KV cache compression逻辑。
  - **编译框架层**：论文未明确说明。FlowMM的KV cache合并操作为纯PyTorch实现，无自定义编译框架修改。
  - **kernel调度层**：论文未明确说明。Cross-modal ratio计算、cosine similarity矩阵计算、weighted averaging合并等操作使用PyTorch GPU kernel（如torch.matmul, torch.cosine_similarity），无自定义CUDA kernel优化。
  - **硬件架构层**：同一A100 GPU上，FlowMM 20% cache budget: GPU memory 0.44 GiB（~80% reduction），decoding latency 17.35 ms/token（vs full cache 29.08 ms/token，~1.7×加速）。在ALFRED上accuracy从KVMerge的27.94%恢复至35.43%（接近full cache的36.92%），在TextNeedle上accuracy从KVMerge的9.69%提升至10.00%（Table 1）。消融实验（Table 4）：移除information flow guidance后ALFRED降至33.58%（-1.85%），移除sensitivity-adaptive matching后降至33.75%（-1.68%），两者都移除降至31.01%（-4.42%）。

  关键设计选择与Baseline缺陷的对应：
  - **defect (a): 忽略跨模态信息流的层间差异性** → 方案：Cross-modal information flow analysis。通过公式ρ^l = (1/H)·Σ_h(A_{v→t}^{l,h} + A_{t→v}^{l,h})/A^{l,h}量化每层的跨模态交互强度。实证发现（Figure 3a）：浅层cross-modal attention比例低（<0.2），深层高（>0.2），且该pattern在ALFRED/MMCoQA/TextNeedle三个不同任务上一致。由此设定threshold θ=0.2（由Table 3验证θ在0.2-0.3区间最优），浅层做intra-modal merging避免模态混淆，深层做cross-modal merging促进跨模态融合。与align vs misalign实验（Figure 3b）相呼应——aligned merging接近full cache性能，misaligned merging大幅退化。
  - **defect (b): 无token敏感度保护** → 方案：Sensitivity-Adaptive Token Matching。定义token sensitivity为其对模型输出保真度的贡献——高敏感度token合并后对模型准确度有显著负面影响。使用attention scores作为sensitivity的零开销近似度量（near-zero-overhead approximation），设置threshold τ过滤高敏感度pivot tokens（公式10: I_j ≤ τ）。消融实验（Table 4）显示：移除sensitivity protection后TextNeedle从10.00%降至6.32%（-3.68%），ALFRED从35.43%降至33.75%（-1.68%），证明sensitivity保护在需要精确保留特定token信息的任务中尤为关键。
  - **defect (c): 合并策略与任务无关** → 方案：Proxy token-based重要性评估。使用prompt末尾少量proxy tokens（capture task-specific contextual information）聚合的attention scores作为token重要性度量（公式8: I^{l,h}(i) = Σ_{j∈P} α_{j→i}^{l,h}），使pivot selection偏向当前任务相关的关键token。相比统一使用累积attention的biased评估，proxy tokens提供更公平（equitable）的token重要性估计。
  - **defect (d): 多模态分布偏移** → 方案：Intra-modal merging in shallow layers。浅层ρ^l<θ时，intra-modal merging将visual tokens和text tokens分别聚类合并——visual tokens仅在visual token内部搜索最近邻合并，text tokens仅在text token内部搜索，避免浅层的cross-modal merging造成visual-text embedding分布偏移导致的语义混淆。深层ρ^l≥θ且cross-modal interactions已充分建立后，才允许跨模态合并。
  - **额外设计：无fine-tuning和plug-and-play** → FlowMM无需fine-tuning，作为plug-and-play KV cache压缩模块直接应用于已有MLLM。所有合并策略由离线校准的ρ^l pattern和运行时动态计算的token importance/sensitivity决定，无需修改模型权重。
  - **性能结果**：80%-95% KV cache memory reduction，1.3×-1.8× decoding latency reduction。在InternVL2.5-8B上20% cache budget下平均accuracy degradation仅0.12%（vs full cache, Table 1）。在低cache budget（<10%）时优势尤其显著（Figure 4），在40% budget时已达full cache相当性能。
