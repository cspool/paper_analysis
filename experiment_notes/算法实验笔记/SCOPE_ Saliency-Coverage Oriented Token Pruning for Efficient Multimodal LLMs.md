## SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种免训练的视觉 token 剪枝算法 SCOPE（Saliency-Coverage Oriented token Pruning），在 MLLM 的 vision encoder 之后、LLM 之前插入剪枝模块，联合建模 token 的显著性（saliency）和语义覆盖度（coverage），通过迭代贪心选择最大化 SCOPE score 的 token 子集来替代原始全量 visual token。实验比较不同 token 保留数量（192/128/64 for LLaVA-1.5，640/320/160 for LLaVA-Next）下各方法相对原始完整模型（Upper Bound）的性能保持率，baseline 包括 FastV、SparseVLM、VisionZip、PDrop、DivPrune。

- 硬件平台是什么，配置是什么。
  4 × NVIDIA A100 GPU。推理 batch size 设置为 1。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-1.5 7B / 13B、LLaVA-Next 7B / 13B（图像理解），Video-LLaVA（视频理解），Qwen2-VL 7B。
  图像 benchmark：GQA (testdev_balanced_instructions, 12578 samples)、MMBench (~3000 MCQs)、MME (dev split, 4377 samples)、POPE (test split, 9000 samples, F1 score)、ScienceQA、TextVQA (test split, 5000 samples, EM)、SEEDBench (19000 MCQs)、MMVet (test split, 218 samples, GPT evaluator)、DocVQA、ChartQA、OCRBench。
  视频 benchmark：TGIF、MSVD、MSRVTT、ActivityNet。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/kinredon/SCOPE。基于 lmms-evals 评估框架实现。

  算法 pipeline：
  1. Vision Encoder（如 CLIP ViT-L/14）将输入图像编码为 N 个 visual token V = {v_1, ..., v_N} ∈ R^d（LLaVA-1.5: N=576, LLaVA-Next: N=2880/5图）。
  2. 从 vision encoder 倒数第二层（layer -2）提取 CLS token 对各 visual token 的 attention score A_v 作为 saliency。
  3. 预计算所有 token 对之间的 cosine similarity 矩阵 S_{uv} = sim(u, v) = u^T v / (||u||·||v||)。
  4. 初始化空集 S = ∅，coverage scores c_u = 0 ∀u ∈ V。
  5. 迭代 K 次（K 为保留 token 数）：
     a. 对每个候选 token v ∈ V\S，计算 marginal gain: Δ(v; S) = Σ_{u∈V} max(S_{uv}, c_u) - c_u
     b. 计算 SCOPE score: Δ(v, A_v^α; S) = Δ(v; S) · A_v^α
     c. 选择 SCOPE score 最高的 token v*，S = S ∪ {v*}，更新 c_u = max(c_u, S_{uv*})
  6. 将选出的 K 个 visual token S 与 text token 拼接后送入 LLM 进行 autoregressive 生成。
  7. 缩放因子 α 默认为 1.0。

  与 saliency-only（仅 Top-K attention）对比：SCOPE 额外计算 token 间相似度矩阵（O(N^2) 存储），并通过迭代贪心选择（O(K·N^2) 时间）替代单次排序。在 LLaVA-1.5 7B 上 K=64（↓88.9%）时仍保持 96.0% 平均性能，saliency-only baselines 最强者 VisionZip 仅 93.5%。在 LLaVA-Next 7B K=160（↓94.4%）时保持 95.1% 性能，VisionZip 仅 92.5%。
