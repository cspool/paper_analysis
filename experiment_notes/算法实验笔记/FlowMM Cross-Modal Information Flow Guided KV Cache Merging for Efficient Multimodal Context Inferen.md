## FlowMM Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  实现FlowMM框架的两个核心组件：(1) **Cross-Modal Information Flow Guided Merging**——通过分析MLLM各层的cross-modal attention比例$\rho^l$（公式6-7），当$\rho^l$超过阈值$\theta$时执行跨模态合并（inter-modal merging），低于$\theta$时执行模态内合并（intra-modal merging），实现层自适应（layer-specific）的KV cache合并策略。同时使用proxy tokens评估token重要性$\mathcal{I}^{l,h}(i)$（公式8），选出top-B pivot tokens保留最关键任务信息，将non-pivot tokens合并到pivot set中。(2) **Sensitivity-Adaptive Token Matching**——通过余弦相似度$u_{i,j}$（公式9）评估token相似度，同时用attention scores作为token敏感度（sensitivity）的近似度量，设置敏感度阈值$\tau$来保护高敏感度token（公式10：$\mathcal{I}_j \le \tau$），仅合并低敏感度token以最小化任务关键信息损失。
  实验比较：(i) 在Qwen2.5-VL-7B、InternVL2.5-8B、MobileVLM-V2-3B三个MLLM上与5个baseline对比——StreamingLLM、H2O（eviction类）、D2O、KVMerge（text-based merging类）、LOOK-M（multimodal-specific merging类），cache budget=20%下的accuracy/ROUGE-L（Table 1）；(ii) 不同cache budget（5%-60%）下的性能对比（Figure 4）；(iii) 效率分析：decoding latency和GPU memory usage vs full cache（Table 2）；(iv) 消融实验：cross-modal merging threshold $\theta$的影响（Table 3）、各组件有效性（Table 4）。

- 硬件平台是什么，配置是什么。
  - 单张NVIDIA A100 Tensor Core GPU（80GB），用于效率分析和消融实验
  - 所有实验在单卡上进行，测量decoding speed（ms/token）和GPU memory consumption（GiB）

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen2.5-VL-7B（Bai et al., 2025）、InternVL2.5-8B（Chen et al., 2024b）、MobileVLM-V2-3B（Chu et al., 2024），涵盖不同架构设计的MLLM
  - Benchmark：MileBench（Song et al., 2024）——首个专门测试MLLM长上下文多模态能力的benchmark，平均每样本15.2张图和422.3个词
  - 评估的7个任务：ALFRED（Conversational Embodied Dialogue, ROUGE-L）、IEdit（Visual Relationship Expressing, ROUGE-L）、STD（Visual Change Captioning, ROUGE-L）、MMCoQA（Multimodal Dialogue, Accuracy）、CLEVR-C（Visual Change Captioning, ROUGE-L）、TextNeedle（Text Needle In A Haystack, Accuracy）、ImageNeedle（Image Needle In A Haystack, Accuracy）
  - 评估框架：论文未明确说明（直接使用MileBench提供的评估脚本）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未给出开源链接。论文未在正文或附录中提供代码仓库URL。

  算法pipeline（FlowMM一次完整前向推理，以Qwen2.5-VL-7B推理一个多模态输入为例）：

  **Phase 1 - 信息流分析（离线或首次推理时进行）**：
  ```
  输入: MLLM的L层transformer, 校准样本集D, 阈值θ
  对每个校准样本:
    对每层 l = 1..L:
      对每个attention head h = 1..H:
        计算cross-modal attention: 
          A_{v→t}^{l,h} = Σ_{v∈V} Σ_{t∈T} α_{v→t}^{l,h}  # visual→text attention
          A_{t→v}^{l,h} = Σ_{t∈T} Σ_{v∈V} α_{t→v}^{l,h}  # text→visual attention
      计算cross-modal interaction ratio:
        ρ^l = (1/H) · Σ_h (A_{v→t}^{l,h} + A_{t→v}^{l,h}) / A^{l,h}
    判断每层合并策略:
      if ρ^l ≥ θ: 层l → cross-modal merging（跨模态合并）
      else:        层l → intra-modal merging（模态内合并）
  ```

  **Phase 2 - KV Cache Merging（每层运行时执行）**：
  ```
  输入: 第l层的KV cache K_t, V_t ∈ R^{L_t×d}, head h
  策略: merge_strategy = cross-modal 或 intra-modal（由Phase 1决定）
  预算: 保留比例B（如20%）
  
  Step 1 - Token重要性评估:
    选proxy tokens P = 最后若干prompt tokens
    For each token i:
      I^{l,h}(i) = Σ_{j∈P} α_{j→i}^{l,h}  # proxy attention聚合
  
  Step 2 - 选择pivot set:
    按I排序，选top-B tokens作为pivot set K^p
    其余tokens作为non-pivot set K^n
    # K^p保留最关键任务信息，K^n将被合并
  
  Step 3 - Sensitivity-Adaptive Token Matching（仅在K^n→K^p方向，受merge_strategy约束）:
    设定sensitivity threshold τ
    For each token i in K^n:
      计算与K^p中所有token j的cosine similarity:
        u_{i,j} = k_i^T k_j / (||k_i|| · ||k_j||)
      找最近邻（仅考虑低敏感度pivot）:
        k_*^{nearest} = Argmax_{j∈K^p, I_j≤τ}(u_{i,j})
      合并: 将token i的KV状态合并到k_*^{nearest}:
        K_*^{merged} = weighted_avg(K_*, K_i)  # attention-weighted averaging
        V_*^{merged} = weighted_avg(V_*, V_i)
      # 若merge_strategy=intra-modal，仅在同模态内搜索最近邻
      # 若merge_strategy=cross-modal，允许跨模态搜索
  
  输出: 压缩后的K^{merged}, V^{merged} ∈ R^{L_compressed×d}
  ```

  关键张量计算：
  - Cross-modal ratio $\rho^l$对每层为标量，O(L·H·|V|·|T|)但仅需一次前向即可确定
  - Token importance: $\mathcal{I}^{l,h} \in \mathbb{R}^{L_t}$，每个token一个标量值
  - Cosine similarity矩阵: $u \in \mathbb{R}^{|K^n|\times|K^p|}$，按merge_strategy过滤候选集
  - KV合并: 加权平均，O(d) per merge operation
  - 总计算开销: 相比full cache inference增加约5-10%（主要来自cosine similarity计算和attention score聚合），但无fine-tuning需求

  与Baseline的关键差异：
  - Baseline（KVMerge等）：所有层使用统一合并策略（uniform merging），未区分cross-modal vs intra-modal层
  - FlowMM：每层根据cross-modal attention flow自适应选择合并策略——浅层（$\rho^l$低）做intra-modal保留模态特征，深层（$\rho^l$高）做cross-modal促进跨模态融合
  - Baseline（LOOK-M等）：基于相似度直接合并，不考虑token sensitivity
  - FlowMM：引入sensitivity threshold $\tau$保护高敏感度token（如包含任务关键信息的特殊token），仅合并低敏感度token
