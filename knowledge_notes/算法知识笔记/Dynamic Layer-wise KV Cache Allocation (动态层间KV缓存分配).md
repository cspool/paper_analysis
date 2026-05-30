## Dynamic Layer-wise KV Cache Allocation (动态层间KV缓存分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dynamic Layer-wise KV Cache Allocation 是 MEDA 提出的根据每层注意力特性动态而非均匀或静态线性分配 KV cache 大小的方法。核心创新在于使用跨模态注意力熵 E_CM^l 作为分配权重，通过 inverse entropy softmax 公式确定每层 cache 比例 α_l：

$$\alpha_l = \frac{\exp(E_{CM}^l)}{\sum_{k=1}^{L} \exp(E_{CM}^k)} \cdot L \cdot \rho$$

$$S_l = \alpha_l \cdot S$$

其中 L 为层数，ρ 为总压缩比（如 0.1 即总 cache 缩减到原来的 10%），S 为总 KV cache budget。因子 L 使 Σ_l (α_l / L) = ρ，确保各层 α_l 之和为 L·ρ。该公式确保：高熵层（注意力分散）获得较大的 α_l（更多 KV cache），低熵层（注意力集中）获得较小的 α_l（更少 KV cache）。

该设计的核心洞察来自 Figure 2 的实证观察：MLLM 的不同层的跨模态注意力密度存在显著差异——早期层注意力分散需要更多 cache 捕捉广泛的跨模态交互，深层注意力已收敛到关键 token 对需要较少的 cache。这与 PyramidKV 的静态线性递减（前层多后层少但无关实际注意力分布）形成鲜明对比。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**动态分配流程**：
```
# 输入：所有 L 层的跨模态注意力熵 [E_CM^1, ..., E_CM^L]
# 参数：总压缩比 ρ，总 KV cache budget S

# Step 1: Softmax 归一化
softmax_weights = softmax([E_CM^1, ..., E_CM^L])    # [L]，sum = 1

# Step 2: 计算每层分配比例
for l in 1..L:
    α_l = softmax_weights[l] · L · ρ                # 确保 sum(α_l) = L·ρ

# Step 3: 计算每层实际 KV cache 大小
for l in 1..L:
    S_l = α_l · S                                   # 若 ρ=0.1，平均每层保留 10%

# Step 4: 每层独立执行 KV pair selection + merging
for l in 1..L:
    N_l = floor(S_l / (1 + M_ratio))                # 按 β1:β2 = 3:1 分配
    K_c[l], V_c[l] = select_and_merge(K[l], V[l], budget=N_l)
```

**与 Uniform/Static Allocation 的对比**：

| 维度 | Uniform (H2O/SnapKV/LOOK-M) | Static Progressive (PyramidKV) | Dynamic (MEDA) |
|------|---------------------------|-------------------------------|----------------|
| 分配依据 | 无，所有层相同 | 固定线性递减（前多后少） | 实时跨模态注意力熵 |
| 是否感知层间差异 | 否 | 不感知实际差异 | 是，自适应 |
| 参数 | ρ 仅控制总 budget | ρ 控制总 budget + 线性递减率 | ρ 控制总 budget，α_l 自动计算 |

术语一般如何实现？如何使用？

实现为 prefill 后的单次分配步骤，不与特定硬件或框架绑定。计算开销 O(L) 可忽略。在 MEDA 中与 text-prior KV pair selection + average merging 组合使用，三者共同形成完整的即插即用 KV cache 压缩 pipeline。消融实验 (Table 5) 验证移除 Dynamic Allocation 导致 CLEVR-Change ROUGE-L 从 18.9 降至 17.8、Spot-the-Diff 从 18.2 降至 17.5。代码开源：https://github.com/AIoT-MLSys-Lab/MEDA。

涉及论文标题：
- PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling（先驱工作，首次提出跨层不均匀 KV cache 分配——基于 Pyramidal Information Funneling 的静态算术序列递减）
- MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference（将静态分配扩展为基于跨模态注意力熵的动态分配）

---

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

MileBench 是首个专门设计用于评估多模态大语言模型（MLLM）长上下文能力的综合 benchmark，由 Song et al. (2024) 在 COLM 2024 发表。包含 6,440 个多模态长文本样本，来自 29 个数据集（21 个已有 + 8 个自建），平均每个样本含 15.2 张图像和 422.3 个词。

MileBench 分为两大评估集合：(1) **Realistic Evaluation**——测试 MLLM 在多模态长上下文场景下的理解和推理能力，包括 Temporal Multi-image Tasks（T-1 到 T-4：动作理解与预测、物体与场景理解、视觉导航与空间定位、反事实推理与状态变化）和 Semantic Multi-image Tasks（S-1 到 S-5：知识 QA、富文本图像 QA、视觉关系推理、对话、空间理解）；(2) **Diagnostic Evaluation**——测试 MLLM 的长距离信息检索和干扰排除能力，包括 Needle in a Haystack Tasks（N-1 Text Needle、N-2 Image Needle）和 Image Retrieval（I-1）。

评估指标包括 Accuracy 和 ROUGE-L，按子任务内各数据集的平均计算。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**MileBench 评估流程**：

```
# 每类任务的典型输入格式
# T-2 (Object & Scene): 多张时序图像 + "Is the mug still on the counter?"
# N-1 (Text Needle): 多张图像 + 长文本背景 + 插入特定文本片段 + "What was the needle?"

# 评估伪代码
for each sample in MileBench:
    images = load_images(sample.image_paths)       # 平均 15.2 张
    prompt = sample.question                        # 平均 422.3 词
    visual_tokens = visual_encoder(images)          # CLIP ViT-L → 576 tokens/image
    input_seq = interleave(visual_tokens, text_tokens)
    kv_cache, output = mllm.prefill(input_seq)      # 多模态 KV cache 构建
    answer = mllm.decode(kv_cache, max_new_tokens)
    score = metric(answer, sample.ground_truth)     # Accuracy 或 ROUGE-L
```

**MileBench 子任务分类（完整 Taxonomy）**：

| 类别 | 任务 | 数据集数 | 评估指标 |
|------|------|---------|---------|
| T-1: Action Understanding | Action Localization/Prediction/Sequence | 3 | Accuracy |
| T-2: Object & Scene | Object Existence/Interaction/Moving/Shuffle | 4 | Accuracy |
| T-3: Visual Navigation | Egocentric Navigation/Moving Direction | 2 | Accuracy |
| T-4: Counterfactual & State | Counterfactual Inference/State Change/Character Order/Scene Transition | 4 | Accuracy |
| S-1: Knowledge QA | Webpage/Textbook/Complex Multimodal/Long Text QA | 4 | Accuracy |
| S-2: Text-Rich QA | Slide QA/OCR QA/Document QA | 3 | Accuracy |
| S-3: Visual Relation | Visual Change Captioning/Relationship Expressing | 2 | ROUGE-L |
| S-4: Dialogue | Multimodal Dialogue/Conversational Embodied Dialogue | 2 | Accuracy/ROUGE-L |
| S-5: Space Understanding | Space Understanding | 1 | Accuracy |
| N-1: Text Needle | Text Needle In A Haystack | 1 | Accuracy |
| N-2: Image Needle | Image Needle In A Haystack | 1 | Accuracy |
| I-1: Image Retrieval | Image Retrieval | 1 | Accuracy |

术语一般如何实现？如何使用？

MileBench 数据集可从 HuggingFace 和百度网盘下载。评估框架开源在 https://github.com/MileBench。使用时配置 MLLM 的推理接口（支持 LLaVA、InternVL、MobileVLM 等架构），设置 batch_size=1（多数数据集），按子任务分别评估后汇总。MileBench 被认为是评估多模态长上下文 MLLM 的事实标准 benchmark，被 LOOK-M、Cross-Self KV Cache Pruning 等多篇论文采用。代码开源：https://github.com/MileBench。

涉及论文标题：
- LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference
- LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models

---
