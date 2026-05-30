## VisiPruner: Decoding Discontinuous Cross-Modal Dynamics for Efficient Multimodal LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**VisiPruner**，一个无需训练的 MLLM 推理剪枝框架，通过揭示 MLLM 跨模态交互的三阶段规律实现对视觉 token 的分层压缩：
  (1) **浅层（Task Recognition）**：在 LLaVA-v1.5 7B 的 layer 1，将所有视觉 token 的 cross-attention 合并到单个随机视觉 token 上作为 attention sink；layer 2+ 完全跳过视觉相关 attention（cross-attention + visual self-attention），仅保留 FFN 对视觉 token 的处理。
  (2) **中层（Sparse Cross-Modal Fusion）**：提出 **influence-based 动态 token 选择方法**——在每个过滤层，计算每个视觉 token 被 mask 后对最后输入 token 的 attention output 的影响，使用 **cosine similarity**（阈值 < 0.995 定义为过滤层）和 **L2 distance**（阈值 < 0.2 的 token 被丢弃）双指标联合评估。平均将 576 个视觉 token 压缩至 10.3 个关键 token。算法具体步骤：① 在每层计算原始 cross-attention output O_i；② 逐个 mask 视觉 token j（设 W'_{i→j}=0），重算 masked attention output O'_{i masked}；③ 计算 CosineSim(O_i, O'_i) 和 L2Dist(O_i, O'_i)；④ 若 cosine < 0.995，定义该层为 filtering layer，将 L2 < 0.2 的 token 丢弃，仅保留剩余 key tokens 进入后续层。
  (3) **深层（Linguistic Alignment）**：在 middle layer 之后持续追踪保留 token 的影响，若连续两个层均无 measurable impact，则定义后一层为 vision exit layer（ℓ_exit，LLaVA-v1.5 7B 平均在第 23.9 层）。超过 ℓ_exit 后移除所有保留的视觉 token，进一步消除冗余计算。

  实验比较：与现有 training-free token pruning 方法对比——
  - **FastV** (Chen et al., 2024a)：基于 last-to-vision attention 选择最重要视觉 token
  - **FitPrune** (Ye et al., 2024b)：基于 attention-distribution saliency 剪枝
  - **SparseVLM** (Zhang et al., 2025b)：基于 cross-attention importance rank-based 剪枝
  - **PyramidDrop** (Xing et al., 2024)：在多个阶段逐步减少视觉 token
  所有比较在相同 visual attention computation reduction ratio 下进行（如 -98.3% 对应的 retained tokens 可能不同）。

- 硬件平台是什么，配置是什么。
  论文未明确说明具体 GPU 型号。主干模型为 LLaMA 2 7B/13B（Touvron et al., 2023）作为 MLLM 的 LLM backbone，使用标准的 GPU 推理环境。FLOPs 分析基于 LLaMA 2 7B 架构：hidden dim d=4096, FFN intermediate dim m=11008, 32 layers, 32 attention heads。论文提到"due to hardware constraints, our analysis was limited to models with up to 13 billion parameters"。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-v1.5 7B、LLaVA-v1.5 13B、InternVL2.5 8B、Qwen2-VL 7B、MobileVLM-v2 3B。
  数据集/benchmark：
  - GQA (Hudson and Manning, 2019)：视觉问答
  - MME (Fu et al., 2024)：综合多模态评估
  - POPE (Li et al., 2023b)：物体幻觉检测
  - MMBench (Liu et al., 2024)：多模态综合能力
  - MMStar (Chen et al., 2024b)：多模态评估
  - ScienceQA / SQA (Lu et al., 2022)：科学推理
  - TextVQA / VQAT (Singh et al., 2019)：OCR 视觉问答
  - MM-Vet (Yu et al., 2024)：多模态综合能力（需要生成式回答）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：代码地址 https://github.com/EIT-NLP/VisiPruner，Apache 2.0 许可证。仓库包含 `llava/cli_pruning.py`（核心剪枝 CLI）、`scripts/`（GQA/MME/TextVQA 评估脚本）、`visualization/`（logits lens、attention 可视化、L1 norm 分析 Notebook）。

  伪代码（VisiPruner 推理前向传播）：
  ```
  Input: visual_embeddings H_v (N_v x d), text_embeddings H_t (N_text x d)
  Hyperparams: shallow_mid_layer S_mid, cosine_threshold=0.995, l2_threshold=0.2
  Output: generated answer tokens

  # Stage 1: Shallow layers (1 to S_mid)
  for l in 1..S_mid:
      if l == 1:
          # Attention Merging: merge all cross-attn to one random visual token k
          A_cross = softmax(Q_t @ K_v^T / sqrt(d))
          A_merged = zeros_like(A_cross)
          A_merged[:, k] = sum(A_cross, dim=1)  # all weights -> token k
          H_cross = A_merged @ V_v
          H_t = TransformerBlock(H_t + H_cross)  # only FFN+self-attn for text
      else:  # l >= 2
          # Skip all visual attention (cross + visual self-attn)
          H_t = TransformerBlock_text_only(H_t)
          H_v = FFN_only(H_v)  # no self-attention among visual tokens
      H = concat(H_v, H_t)

  # Stage 2: Middle layers (S_mid+1 onward)
  filtering_layer_found = False
  for l in S_mid+1..L:
      if not filtering_layer_found:
          # Compute original attention output for last text token
          O_last = Attention(Q_t[-1], K_all, V_all)
          for each visual token j:
              W'_i->j = 0  # mask token j
              O'_masked = Attention_masked(Q_t[-1], K_all, V_all, mask=j)
              cos_sim[j] = dot(O_last, O'_masked) / (||O_last|| * ||O'_masked||)
              l2_dist[j] = ||O_last - O'_masked||_2
          if min(cos_sim) < 0.995:
              filtering_layer_found = True
              H_v = H_v[l2_dist >= 0.2]  # keep only influential tokens
      # Continue with retained visual tokens only
      H = TransformerLayer(concat(H_v, H_t))

  # Stage 3: Deep layers - Vision Exit
      if filtering_layer_found:
          if tokens_have_no_impact_for_2_consecutive_layers:
              H_v = []  # remove all vision tokens, exit at this layer
              # Continue with text-only processing
      H = TransformerLayer(H)

  return generated_tokens
  ```

  复杂度分析（LLaVA-v1.5 7B，576 visual + 74 text tokens）：
  - 视觉相关 attention 计算最大减少 99.0%（-98.3% 配置下）
  - 总 FLOPs：从 3.82T 降至 1.76T（-53.9%）
  - FFN FLOPs 公式：3 × n × d × m；attention FLOPs：2 × n² × d
  - 视觉 FLOPs 总体减少 62.8%（考虑到仍保留 FFN 对视觉 token 的处理）
  - KV cache 大幅缩减：浅层和深层不存视觉 KV，中间仅存 ~10 tokens
