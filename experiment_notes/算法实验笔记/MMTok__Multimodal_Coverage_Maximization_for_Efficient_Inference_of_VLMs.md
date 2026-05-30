## MMTok__Multimodal_Coverage_Maximization_for_Efficient_Inference_of_VLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：MMTok —— 基于多模态覆盖最大化（Multimodal Coverage Maximization）的 training-free 视觉 token 选择算法。核心思路是将 token 选择建模为最大覆盖问题（Maximum Coverage Problem），从 n 个 vision token 中选择 k 个（k ≪ n）以最大化覆盖文本 token（T-V coverage）和全体视觉 token（V-V coverage）的信息量。覆盖函数定义为子模函数（submodular function）f(S; M) = (1/m) Σᵢ maxⱼ M_{i,j}，通过贪心算法获得 (1-1/e) 近似最优解。
  具体计算流程：(1) 视觉编码器提取 vision token {v₁',…,vₙ'}（投影前）和 {v₁,…,vₙ}（投影后）；(2) 计算 text-vision 相似度矩阵 M^{tv} = tᵢᵀvⱼ（使用投影后的 vision token 以对齐文本语义）；(3) 计算 vision-vision 相似度矩阵 M^{vv} = vᵢ'ᵀvⱼ'（使用投影前的 vision token 以保留纯视觉信息）；(4) 对两个相似度矩阵分别做 softmax 归一化校准（温度 τ_t=0.02, τ_v=0.2）：M^{tv'}_{i,j} = exp(M^{tv}_{i,j}/τ_t) / Σⱼ exp(M^{tv}_{i,j}/τ_t)；M^{vv'} 同理；(5) 合并目标 f(S; M^{tv'}, M^{vv'}) = f(S; M^{tv'}) + α·f(S; M^{vv'})，α=0.5；(6) 贪心算法 O(kn) 选 k 个 vision token，每次选使得增量覆盖最大的 token。方法为 training-free，无需任何微调或额外训练参数。

  实验比较：(a) 性能对比 —— 在 LLaVA-1.5-7B (576 tokens → 192/128/64)、LLaVA-1.5-13B (576 → 192/128/64)、LLaVA-NeXT-7B (max 2880 → 640/320/160)、LLaVA-NeXT-13B (max 2880 → 640/320/160)、Qwen-2.5-VL-7B (dynamic tokens → 20%/10%/5%) 上，对比 FastV (vision-only)、SparseVLM (language-only)、VisionZip (CLS-attention-based)、DivPrune (diversity-based)、VisionZip fine-tuned。(b) 高 IC 任务极端压缩 —— 在 5 个高 Image Contribution 任务 (POPE, MME, MMB, SEED, GQA + TextVQA for NeXT) 上，压缩至 64→32→16→8→4→2 tokens，对比 VisionZip 和 DivPrune。(c) 效率分析 —— H100 上 LLaVA-NeXT-13B 和 A6000 上 Qwen2.5-VL-7B 的推理时间、GPU 利用率、显存对比。(d) 消融实验 —— T-V only vs V-V only vs Softmax 变体 vs MMTok full；token selection vs 图像 resize 策略对比；温度参数 τ_v 自适应搜索；word pooling 策略 (Mean/Max/First, Pre/Post) 对比；decoder 内 token 二次选择。(e) 推理任务 —— MMStar benchmark。(f) MMTok++ 改进 —— 排除 padding patches + 修复 overflow bug，进一步测试 32/16/8/4/2 tokens 极限性能。

- 硬件平台是什么，配置是什么。
  推理效率测试：NVIDIA H100 (80GB) 单卡 —— LLaVA-NeXT-13B loading 25.42GB，POPE dataset total infer time 测量。NVIDIA A6000 (48GB) 单卡 —— Qwen2.5-VL-7B loading 15.87GB，MME task infer time 测量；A6000 还用于 MMTok 贪心选择算法运行时间 profiling（100 runs 平均）。主要实验（各 benchmark 评估）论文未明确说明硬件平台，推测使用与效率实验相同的 GPU。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) LLaVA-1.5-7B —— Vision Encoder: CLIP-ViT-L-336px (576 fixed vision tokens), LLM: Vicuna-7B。(2) LLaVA-1.5-13B —— 同上架构，LLM: Vicuna-13B。(3) LLaVA-NeXT-7B —— 动态多分辨率，最高 5 图像 × 576 = 2880 tokens。(4) LLaVA-NeXT-13B —— 同上，最大 2880 tokens。(5) Qwen-2.5-VL-7B-Instruct —— 动态分辨率 + token merging layer，平均 token 数 276.9~976.5（随数据集而异）。

  数据集和 Benchmark：(1) GQA (Acc.) —— 真实世界视觉推理与组合问答。(2) MMBench / MMB (Acc.) —— 多模态全方位评估。(3) MME (Perception+Cognition, P+C) —— 多模态大模型综合评测。(4) POPE (F1) —— 物体幻觉检测。(5) ScienceQA-IMG / SQA (Acc.) —— 科学多模态推理（低 IC 数据集）。(6) VQA-v2 Test-Dev (Acc.)。(7) TextVQA (Acc.) —— 图中文字问答。(8) MMMU (Acc.) —— 多学科多模态理解。(9) SeedBench-Image / SEED-I (Acc.)。(10) OCRBench (Acc.) —— Qwen 实验专用。(11) MMStar (Coarse/Fine-Grained/Instance/Logical/Math/Sci&Tech Acc.) —— 推理任务 benchmark。
  评估框架：Lmms-eval (https://github.com/EvolvingLMMs-Lab/lmms-eval)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Ironieser/mmtok
  
  算法 pipeline 详细解释（张量计算级别）：

  **输入**：图像 I ∈ R^(3×H×W)，文本查询 T = {t₁,…,tₘ}，目标 token 数 k。
  
  **Step 1 — 视觉编码**：图像通过 CLIP-ViT 视觉编码器 → vision tokens V' = {v₁',…,vₙ'} ∈ R^(n×d)（投影前）。Vision tokens 通过 MLP projection layer → V = {v₁,…,vₙ} ∈ R^(n×d')（投影后，与 LLM embedding 空间对齐）。对于 LLaVA-1.5，n=576 (336×336 image)。
  
  **Step 2 — Text-Vision 相似度**：文本 T 通过 LLM tokenizer + embedding → text hidden states {t₁,…,tₘ}（使用 projection 后的 vision token 计算）。归一化为单位向量：∥tᵢ∥₂ = ∥vⱼ∥₂ = 1。计算 T-V 相似度矩阵 M^{tv} = T · Vᵀ ∈ R^(m×n)，其中 M^{tv}_{i,j} = tᵢᵀ vⱼ。
  
  **Step 3 — Vision-Vision 相似度**：使用投影前的 vision token V'。归一化后计算 V-V 相似度矩阵 M^{vv} = V' · V'ᵀ ∈ R^(n×n)，其中 M^{vv}_{i,j} = vᵢ'ᵀ vⱼ'。
  
  **Step 4 — Softmax 校准**：对两个相似度矩阵分别按行做 temperature-scaled softmax。M^{tv'}_{i,j} = exp(M^{tv}_{i,j}/τ_t) / Σⱼ exp(M^{tv}_{i,j}/τ_t)，τ_t=0.02。M^{vv'}_{i,j} = exp(M^{vv}_{i,j}/τ_v) / Σⱼ exp(M^{vv}_{i,j}/τ_v)，τ_v=0.2。
  
  **Step 5 — 多模态覆盖贪心选择 (Alg. 2)**：
  ```
  输入: M^{tv'} ∈ R^(m×n), M^{vv'} ∈ R^(n×n), k, α=0.5
  S = ∅
  for i = 1 to k:
    for each s ∈ N \ S:
      // 计算增量覆盖 f(S ∪ {s}; M^{tv'}, M^{vv'})
      // f(X; M^{tv'}) = (1/m) Σᵢ₌₁ᵐ max_{j∈X} M^{tv'}_{i,j}
      // f(X; M^{vv'}) = (1/n) Σᵢ₌₁ⁿ max_{j∈X} M^{vv'}_{i,j}
      g(s) = f(S ∪ {s}; M^{tv'}) + α · f(S ∪ {s}; M^{vv'})
    s* = argmax_s g(s)
    S = S ∪ {s*}
  返回 S  // 选中的 k 个 vision token 索引
  ```
  复杂度: O(kn)，对 m ≪ n 和 d 固定。对 2880 tokens 选 160 tokens 仅需 6.4ms on A6000，13.93 GFLOPs。
  
  **Step 6 — LLM 推理**：仅将选中的 k 个 vision token {v_s}_{s∈S} 与 text tokens 拼接后送入 LLM decoder。
  
  **与 baseline 对比示例**（以 LLaVA-1.5-7B, "Describe the cat in the image" 为例）：
  - Baseline: CLIP-ViT → 576 vision tokens → concat with ~10 text tokens → LLM (attention cost O(586²))
  - MMTok: CLIP-ViT → 576 vision tokens → greedy coverage selection (64 tokens) → concat with ~10 text tokens → LLM (attention cost O(74²))。保留 96.6% 原始性能（Avg. over 8 benchmarks），推理 token 数减少 88.9%。
