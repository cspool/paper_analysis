## AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种 training-free 的自适应多模态 LLM 推理方法，由两个阶段组成：(1) LLM 前基于 Token Embedding 余弦相似度的迭代 Token 合并（Token Merging），将输入 LLM 的冗余视觉 token 两两配对合并，每次迭代最多减半；(2) LLM 内部基于 PageRank 的渐进式 Token 剪枝（Token Pruning），在每层 Transformer 对 Self-Attention 权重矩阵应用 PageRank 算法计算每个视觉 token 的重要性分数，保留高分 token 并剪除低分 token，同时设计分段线性 Scheduler 控制各层的 retention ratio。实验比较与 Base 模型（LLaVA-OV-7B / LLaVA-1.5-7B）和 training-free baseline 方法（FastV、VTW、PDrop、LLaVA-Prumerge）在不同 FLOPs/prefill time 下的准确率 trade-off。

- 硬件平台是什么，配置是什么。
  FLOPs 和 prefill time 使用 LLM-Viewer 库计算，假设 video LLM 有 100 个 text token、image LLM 有 40 个 text token。GPU 硬件论文未明确说明具体型号。

- 模型是什么。数据集和bench分别是什么。
  Video LLM 模型：LLaVA-OneVision-7B（LLM backbone 为 Qwen2-7B，28 layers），采样 32 frames/video。
  Image LLM 模型：LLaVA-1.5-7B（LLM backbone 为 Vicuna-v1.5-7B，32 layers）。
  Video Benchmarks：VideoMME、MVBench、MLVU、EgoSchema、NextQA、PerceptionTest。
  Image Benchmarks：GQA（12,578 samples）、VQAv2（107,394 samples）、MME（2,374 samples）、TextVQA（5,000 samples）、SQA-IMG（2,017 samples）、MMB（4,377 samples）、POPE（8,910 samples）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/LaVi-Lab/AIM（ICCV 2025），基于 PyTorch 2.3.1 + CUDA 12.1。

  **Stage 1: Token Merging (LLM 前)**：
  输入：视觉 token embedding v⁰ ∈ R^{N⁰×D}
  流程：
  ```
  // 单次迭代（每次减半）
  将 N 个 visual tokens 分为 A（偶数位置）和 B（奇数位置）
  对 A 中每个 token i，在 B 中找余弦相似度最高的 token j：
      j* = argmax_j  cos_sim(v_A[i], v_B[j])
  对最高相似度的 token pair 取均值合并：v_merged = (v_A[i] + v_B[j*]) / 2
  合并后保留 N/2 个 token
  // 重复 I 次，保留率为 r_merge = (1/2)^I
  ```
  视频任务中仅在单帧内（spatial）合并，不跨帧（temporal）合并。

  **Stage 2: Token Pruning (LLM 内部)**：
  对第 l 层，输入 token x^l = [v^l; t^l]（视觉 + 文本），用 Attention 权重矩阵 A^l ∈ R^{(N^l+M^l)×(N^l+M^l)} 计算 PageRank 重要性分数：
  $$s_i^l = \frac{1}{N^l + M^l} \sum_{j=1}^{N^l + M^l} \mathbf{A}_{i,j}^l \cdot s_j^l$$
  初始 s_j 均匀分布。仅对视觉 token 按 s_i 排序剪枝，文本 token 保持完整。

  **Scheduler（分段线性保留率）**：
  $$r^l = \begin{cases} 1, & l < l_1 \\ 1 - k(l - l_1), & l_1 \leq l \leq l_2 \\ 0, & l > l_2 \end{cases}$$
  其中 k = 1/(l₂-l₁)。l₁ 控制开始剪枝层，l₂ 控制完全移除层。

  **默认配置**：
  - Video：r_merge=25%, l₁=14, l₂=22（共 28 层 Qwen2）
  - Image：r_merge=12.5%, l₁=13, l₂=21（共 32 层 Vicuna）

  **执行流程**：
  1. Visual Encoder 产生 N⁰ 个 visual tokens
  2. 通过 I 次迭代 Token Merging（基于余弦相似度），保留 N⁰ × r_merge 个 token
  3. Merged visual tokens + text tokens → LLM Layer 1
  4. 每层：计算 Self-Attention → 用 A^l 计算 PageRank 分数 → 按 Scheduler 的 r^l 保留 top-K visual tokens（K = N^{l-1} × r^l）→ 被剪枝 token 从后续层 KV Cache 中移除
  5. l > l₂ 后所有视觉 token 已移除，仅剩 text token 继续推理

  额外开销极小：Token Merging 88.25 GFLOPs + Token Pruning 4.18 GFLOPs（合计 92.43 GFLOPs），仅占 Qwen2-7B LLM 推理 FLOPs（14757 GFLOPs）的 0.6%。
