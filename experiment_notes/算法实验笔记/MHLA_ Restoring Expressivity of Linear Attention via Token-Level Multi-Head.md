## MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 Multi-Head Linear Attention (MHLA)，将 token 序列沿 token 维度划分为 M 个 non-overlapping blocks（"heads"），每个 block 计算局部 KV summary，再通过可学习的系数矩阵 Mc（Multi-Head Mixing）让每个 query block 对各 block 的 summary 进行加权混合，恢复 query-conditioned 的 token 级别选择性，同时保持 O(Nd²) 线性复杂度。
  - 实验比较：在 DeiT/VLT（图像分类）、DiT/DiG（类别到图像生成）、SANA（文本到图像生成）、Wan2.1（视频生成）、Transformer++（NLP）五个领域，将 MHLA 替代原始 attention 模块（self-attention 或 linear attention），对比准确率/FID/生成质量/perplexity 等指标。

- 硬件平台是什么，配置是什么。
  - NVIDIA H100 Tensor Core GPU（吞吐量测试使用）
  - 不同设备上的吞吐量对比（DiT-S/2 at 4096 resolution across different devices）
  - 训练硬件：论文未明确说明具体 GPU 型号和数量

- 模型是什么。数据集和bench分别是什么。
  - 图像分类模型：DeiT-T/S、VLT-T/S；数据集：ImageNet-1K
  - 图像生成模型：DiT-S/B/L/XL、DiG-S；数据集：ImageNet-1K (C2I)
  - 文本到图像生成模型：SANA-0.6B（从官方 checkpoint fine-tune）；数据集：31,292 张互联网图片
  - 视频生成模型：Wan2.1-1.3B（替换 FlashAttention 为 MHLA）；评测：VBench；序列长度 31,500 tokens（81 frames at 480×800）
  - NLP 模型：340M 参数语言模型；数据集：FineWeb-Edu 10B tokens 训练，SlimPajama 5B tokens；评测：MMLU, Commonsense Reasoning (WinoGrande, PIQA, ARC-c, ARC-e, OBQA, BoolQ), Wiki ppl, LMB ppl, LongBench

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://github.com/DAGroup-PKU/MHLA（MIT license）
  - 代码结构：五个子项目 mhla_image_classification、mhla_dit、mhla_nlp、mhla_videogen、mhla_sana
  - 基于 PyTorch 实现，依赖 flash-linear-attention、timm 等库
  - 算法 pipeline 张量计算流程：
    1. 输入 X ∈ R^(N×d)，线性投影得到 Q, K, V = XW_Q, XW_K, XW_V
    2. Kernelized 特征映射：Q̃ = φ(Q), K̃ = φ(K)（如 ReLU 或 elu+1）
    3. 将序列分为 M 个 blocks，每 block b 含 N_b 个 tokens
    4. 每个 block 计算局部 KV summary：S_b = Σ_{j∈b} K̃_j^T V_j ∈ R^(d×d)，z_b = Σ_{j∈b} K̃_j ∈ R^d
    5. Multi-Head Mixing：通过可学习系数矩阵 Mc ∈ R^(M×M)，query block i 的混合 summary：S̃_i = Σ_{b=1}^M m_{i,b} S_b，z̃_i = Σ_{b=1}^M m_{i,b} z_b
    6. 输出计算：o = (q̃^T S̃_i) / (q̃^T z̃_i)
    7. 初始化策略：m_{i,j}^(0) ∝ 1 - dist(i,j)/max_k(dist(i,k))（locality-biased），训练中 clip 到 (0,1) 确保非负
    8. 复杂度 O(Nd² + M²d²)，当 M² ≤ N 时主导项为 O(Nd²)
