## StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression

- baseline方法是什么？
  Baseline 是 **ReKV**（Di et al., 2025, ICLR），第一个为 Streaming Video QA 引入 KV-cache 检索机制的 Online Video-LLM。ReKV 全栈执行过程：

  **算法pipeline**：将视频流按固定帧数均匀切分为 uniform segments，每段编码后存储完整 KV cache（不做压缩）。收到用户问题时，基于问题 query vector 对所有历史存储的 KV caches 做 similarity-based 检索，选出 query-relevant 的 KV blocks 送入 LLM 生成答案。检索策略为 uniform allocation：每层分配相同数量的 KV blocks。

  **Serving调度**：基于 LLaVA-OneVision-Qwen2-7B-OV，NVIDIA H20 GPU (96GB)，0.5 FPS 处理视频流，local window = 15K tokens。均匀分段 → 顺序编码 → 全部 KV cache 存入 KV Bank → 问题到来时检索 → 生成答案。

  **kernel调度/编译框架/硬件架构/芯片设计**：论文未明确说明（使用标准 PyTorch + HuggingFace Transformers 推理）。

  Baseline 缺陷：(1) **均匀分段打断语义连续性**：固定帧数切分无视视频内容的语义边界，可能在关键事件中间切断，破坏语义信息完整性；(2) **存储全部 KV cache 导致显存爆炸**：长视频下累积存储所有历史 KV caches，无压缩机制，显存随视频时长线性增长；(3) **检索策略僵化**：uniform per-layer allocation 未考虑不同 transformer 层信息分布的差异性，低效利用检索预算；(4) **检索精度不足**：需要检索更多帧才能确保相关信息被包含（Figure 4），引入噪声信息反而降低 QA 准确率。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  StreamKV 提出 training-free 框架，通过四个核心设计解决 ReKV 的缺陷：

  **算法pipeline**（全栈改进）：

  (1) **语义分段划分 → 解决均匀分段破坏语义连续性**：基于相邻帧 ViT embedding 的 cosine similarity 检测语义边界（s_t = cos_sim(f_{t-1}, f_t), threshold=0.99），配合 exclusion window (m=4) 和 segment merging (M=64) 保证段长合理。使分段尊重视频内容的自然语义结构，避免关键信息被切分。每个段计算 summary vector（空间位置平均）保留 segment-level information。Table 2 消融显示语义分段在所有压缩率下均优于均匀分段（如 50% 压缩率: 59.07% vs 57.32%）。

  (2) **Guidance Prompt 驱动的 KV 压缩 → 解决显存爆炸**：每段编码后立即应用压缩（非解码阶段离线压缩），引入 guidance prompt 捕获段内关键语义元素——salient entities（人/物体/场景）、key events/actions（发生了什么）、temporal/causal relationships（事件时序因果链）、contextual cues（场景切换/对话/叙事变化）、factual details（计数/摘要等）。以 guidance prompt 的平均 query vector 作为 selection criterion，选出每段最 informative 的 KV blocks 保留，压缩率 0%-90% 可调。实验显示 60% 压缩率下 StreamKV Overall 58.9% vs ReKV (无压缩) 53.5%，90% 极端压缩下仍保持 56.7%。

  (3) **Unified Layer-Adaptive KV Selection Module → 解决检索策略僵化**：将压缩和检索统一为同一模块。每层计算 softmax-normalized 相似度分布，通过 binary search 确定全局 cumulative score threshold p，使每层预算 K_l 与该层信息集中度成正比——信息越集中的层获得越多预算。Table 4 消融：Ada.+Ada.（压缩和检索均自适应）在 50% 压缩率下准确率 59.07%，优于 Uni.+Uni. 的 58.12%，单独对压缩或检索使用自适应也优于全 uniform。

  (4) **Precise Retrieval Strategy → 解决检索精度不足**：基于 question vector 作为 selection criterion 的层自适应检索。Figure 4 显示 StreamKV 仅需检索 8 帧即达最优准确率，检索更多帧反而因引入不相关噪声导致性能下降（"Lost in the Middle" 效应），与 ReKV 需要检索更多帧才能覆盖相关信息的趋势完全相反。这证明了 StreamKV 检索的高精度。

  **Serving调度**：基于 LLaVA-OneVision-Qwen2-7B-OV，NVIDIA H20 GPU (96GB)，FP16。0.5 FPS 处理帧率，local window = 15K。语义分段 → sliding-window encoding（含 summary vector）→ 即时 KV 压缩 → 存入 KV Bank。收到问题 → KV 检索 → 生成答案。Figure 1 显示 StreamKV 在准确率、显存、延迟三个维度均优于 ReKV，显存约为 ReKV 的 50-60%（60% 压缩率下）。

  **RoPE 策略**：encoding 阶段 RoPE 仅应用于 local window 内（inspired by LM-Infinite），QA 阶段基于 relative positions 应用 RoPE，缓解长序列下 RoPE 远距离 attention 衰减问题。

  **kernel调度/编译框架/硬件架构/芯片设计**：论文未明确说明。
