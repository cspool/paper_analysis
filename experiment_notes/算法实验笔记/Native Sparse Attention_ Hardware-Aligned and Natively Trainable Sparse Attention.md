## Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  NSA（Native Sparse Attention）提出一种原生可训练的稀疏注意力机制，通过三条并行注意力路径（compression、selection、sliding window）替代 Full Attention 的密集计算。核心实现：(1) Token Compression：将 key/value 序列按块（block length l=32, stride d=16）通过可学习 MLP φ（含 intra-block position encoding）压缩为块级表示 $\tilde{K}_t^{\text{cmp}}, \tilde{V}_t^{\text{cmp}}$，捕获粗粒度全局语义；(2) Token Selection：利用压缩注意力的中间 attention score $\mathbf{p}_t^{\text{cmp}}$ 推导 selection block 的重要性分数，Top-n（n=16，含 1 个初始块和 2 个局部块）保留精细 token 块（block size l'=64），实现 blockwise 连续内存访问以利用 Tensor Core；(3) Sliding Window：独立窗口分支（w=512）显式处理局部上下文，三条分支输出通过可学习门控 $g_t^c = \text{Sigmoid}(\text{MLP}(\mathbf{q}_t))$ 融合；(4) 三路 attention 使用独立的 key/value 投影矩阵，防止局部短路学习。

  实验比较：(a) NSA vs Full Attention baseline on 通用 benchmarks（MMLU, MMLU-PRO, CMMLU, BBH, GSM8K, MATH, DROP, MBPP, HumanEval — 27B 模型, 270B tokens 预训练）；(b) NSA vs H2O/InfLLM/Quest/Exact-Top on LongBench（SQA, MQA, Synthetic, Code 子类）；(c) NSA vs Full Attention on Needle-in-a-Haystack（64k context）；(d) NSA-R vs Full Attention-R on AIME 数学推理（8k/16k 生成 token 限制，DeepSeek-R1 蒸馏 SFT）；(e) 预训练 loss 曲线对比（NSA 始终低于 Full Attention）；(f) 消融：token selection 策略对比（Key-Clustering, auxiliary loss-based, heuristic parameter-free vs NSA 的 compression-based 重要性分数推导）。

- 硬件平台是什么，配置是什么。
  8-GPU NVIDIA A100 系统。预训练：27B 总参数（3B active），GQA+MoE backbone，30 layers，hidden dimension 2560，64 attention heads，GQA groups=4（每 group 16 heads），d_q=d_k=192，d_v=128，MoE 72 routed experts + 2 shared experts，top-k=6。训练数据：270B tokens of 8k-length texts，后续用 YaRN 在 32k-length texts 上 continued training + SFT。NSA 超参：compression block size=32，sliding stride=16，selected block size=64，selected block count=16，sliding window=512。

- 模型是什么。数据集和bench分别是什么。
  模型：27B 参数 Transformer（GQA + DeepSeekMoE），3B active parameters。数据集：270B tokens 预训练语料（8k 长度），10B tokens 32k-length 数学推理链用于 SFT（蒸馏自 DeepSeek-R1）。Benchmarks：(a) 通用：MMLU (5-shot), MMLU-PRO (5-shot), CMMLU (5-shot), BBH (3-shot), GSM8K (8-shot), MATH (4-shot), DROP (1-shot F1), MBPP (3-shot Pass@1), HumanEval (0-shot Pass@1)；(b) 长上下文：LongBench（MFQA-en, MFQA-zh, Qasper, HPQ, 2Wiki, GovRpt, Dur, PassR-en, PassR-zh, LCC）；(c) Needle-in-a-Haystack (64k)；(d) 推理：AIME 24（16 samples, temperature 0.7, top-p 0.95）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文为 DeepSeek-AI 出品，论文中未提供显式 GitHub 链接但建议关注 DeepSeek 官方仓库。算法 pipeline 的核心张量计算如下：

  **Step 1 - Token Compression（粗粒度压缩）**：
  输入序列 X ∈ R^{t×d}，对 key 序列 k_{:t}，每 l=32 个连续 token 为一个 block，stride d=16：
  $\tilde{K}_t^{\text{cmp}} = \{\varphi(\mathbf{k}_{id+1:id+l}) \mid 0 \le i \le \lfloor\frac{t-l}{d}\rfloor\}$
  其中 φ 为含 intra-block position encoding 的 MLP，将每个 l×d_k block 映射为 1×d_k 压缩 key。同理得到 $\tilde{V}_t^{\text{cmp}}$。Block 数 ≈ t/16。

  **Step 2 - Blockwise Selection（基于压缩注意力的细粒度选择）**：
  先计算压缩注意力分数：$\mathbf{p}_t^{\text{cmp}} = \text{Softmax}(\mathbf{q}_t^T \tilde{K}_t^{\text{cmp}} / \sqrt{d_k}) \in \mathbb{R}^{\lfloor(t-l)/d\rfloor + 1}$
  将压缩分数按空间对应关系聚合为 selection block（l'=64）的重要性分数：
  若 d|l 且 d|l'：$\mathbf{p}_t^{\text{slc}}[j] = \sum_{m=0}^{l'/d-1}\sum_{n=0}^{l-1} \mathbf{p}_t^{\text{cmp}}[\frac{l'}{d}j - m - n]$
  GQA 场景下跨 head 聚合：$\mathbf{p}_t^{\text{slc'}} = \sum_{h=1}^{H} \mathbf{p}_t^{\text{slc},(h)}$
  取 Top-n（n=16）blocks：$\mathcal{I}_t = \{i \mid \operatorname{rank}(\mathbf{p}_t^{\text{slc'}}[i]) \le n\}$
  拼接选中 block 的原始 K, V：$\tilde{K}_t^{\text{slc}} = \text{Cat}[\{\mathbf{k}_{il'+1:(i+1)l'} \mid i \in \mathcal{I}_t\}] \in \mathbb{R}^{d_k \times nl'}$

  **Step 3 - Sliding Window（局部窗口）**：
  $\tilde{K}_t^{\text{win}} = \mathbf{k}_{t-w:t}$，$\tilde{V}_t^{\text{win}} = \mathbf{v}_{t-w:t}$，w=512。

  **Step 4 - Gated Fusion（门控融合）**：
  三条路径分别计算 attention 后加权融合：
  $\mathbf{o}_t^* = \sum_{c \in \{\text{cmp},\text{slc},\text{win}\}} g_t^c \cdot \text{Attn}(\mathbf{q}_t, \tilde{K}_t^c, \tilde{V}_t^c)$
  其中 $g_t^c = \text{Sigmoid}(\text{MLP}_g(\mathbf{q}_t))$，三路使用独立 K, V 投影。

  总稀疏度：$N_t = |\tilde{K}_t^{\text{cmp}}| + |\tilde{K}_t^{\text{slc}}| + |\tilde{K}_t^{\text{win}}| \approx t/16 + 1024 + 512 \ll t$（长序列下）。

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  见 kernel调度 分层。
