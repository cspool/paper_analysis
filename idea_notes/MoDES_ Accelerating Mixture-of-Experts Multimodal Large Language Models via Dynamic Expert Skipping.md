## MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

- baseline方法是什么？
  Baseline 为 LLM 领域的 expert skipping 方法（NAEE [42]、MC-MoE [22]、DiEP [6]），它们仅依赖 **intra-layer routing probabilities** 决定跳过哪些 expert，且为 **unimodal LLM** 设计。以 NAEE 为例说明全栈执行路径：
  - **算法层（Expert Skipping）**：输入 token 经 router 得 routing probs $\pi_1,...,\pi_M$ 和 top-k indices $\mathcal{S}^{(l)}$。NAEE 判断：若累积尾部概率 $\sum_{u=i}^k \pi_{\text{top-}u}^{(l)} < \beta^{(l)} \cdot \sum_{v=1}^k \pi_{\text{top-}v}^{(l)}$，则跳过 top-i 到 top-k 的 expert。**缺陷**：(1) 忽略了 layer-level 的全局贡献差异——浅层 expert 的错误会经后续层放大，应保守跳过，深层可激进跳过，但 NAEE 对所有层同等对待；(2) 忽略了 modality gap——vision token 的 expert 冗余度远高于 text token（FFN 对 vision token 的更新幅度小），应更激进跳过 vision expert，但 NAEE 对所有 token 一视同仁。直接应用这些方法到 MoE MLLM 在 83% skipping ratio 下导致 >10% 平均精度下降。
  - **系统框架层**：使用 HuggingFace Transformers 加载模型，标准 PyTorch 推理。baseline（DiEP 等）通过离线校准选择超参数 $\beta^{(l)}$，推理时无额外开销。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 PyTorch MoE kernel 实现。论文提到 DiEP 等 baseline 在相同 skipping ratio 下的 inference speedup 与 MoDES 类似（差别 <1%），因其推理开销可忽略。
  - **硬件架构层**：单张 H200 GPU 推理。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文 MoDES 方法通过三个组件逐层解决 Baseline 缺陷：
  1. **GMLG（解决全局贡献忽略）**：通过离线校准计算每层的全局重要性因子 $\alpha^{(l)}$（跳过该层所有 expert 后的 KL 散度），推理时 $s_i^{(l)} = \alpha^{(l)} \cdot \pi_i^{(l)}$。浅层 $\alpha^{(l)}$ 大 → $s_i^{(l)}$ 大 → 更难被跳过 → 保护关键浅层 expert。深层 $\alpha^{(l)}$ 小 → $s_i^{(l)}$ 小 → 更容易被跳过 → 激进去除深层冗余。
  2. **DMT（解决 modality gap 忽略）**：设置独立阈值 $\tau_t$（text）和 $\tau_v$（vision）。基于发现 vision token 与 FFN 权重的夹角更接近 $90^\circ$（更新幅度小），设置 $\tau_v > \tau_t$ 使 vision expert 被更激进跳过。如图 8 所示，实际 skipping ratio 在 vision token 上远高于 text token。
  3. **Frontier Search（解决阈值搜索效率）**：利用 $f$（KL 散度）和 $g$（skipping ratio）对阈值单调递增的性质，设计 $\mathcal{O}(ND)$ 算法替代 $\mathcal{O}(ND^2)$ exhaustive search，搜索时间从 >2 天降至 <2 小时（30B 模型）。

  MoDES 全栈执行路径（与 baseline 同框架对应）：
  - **算法层（Expert Skipping）**：输入 token x 进入第 l 个 MoE FFN → Router 输出 routing probs $\pi_i^{(l)}$ 和 top-k set $\mathcal{S}^{(l)}$ → **GMLG** 计算 $s_i^{(l)} = \widetilde{\alpha}^{(l)} \cdot \pi_i^{(l)}$（$\widetilde{\alpha}^{(l)}$ 离线预计算，推理无开销）→ **DMT** 根据 token modality 选择 $\tau_t$ 或 $\tau_v$，跳过 $s_i^{(l)} < \tau$ 的 expert → 仅保留的 expert 参与加权聚合 $\mathbf{y}^{(l+1)} = \sum_{i \in \text{kept}} \pi_i^{(l)} \cdot \text{Expert}_i^{(l)}(\mathbf{x}^{(l)})$。
  - **系统框架层**：基于 HuggingFace Transformers。离线阶段：用 GQA 1024 样本 calibration → 计算 $\alpha^{(l)}$ → Frontier Search 得 $(\tau_t^*, \tau_v^*)$。在线阶段：加载预计算的 $\widetilde{\alpha}^{(l)}$ 和阈值 pair，推理时动态跳过。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：编写自定义 CUDA kernel：(a) Router kernel 内嵌 DMT branch-free comparison + sentinel ID assignment（无额外 kernel launch）；(b) Dispatch/Gather 过滤 sentinel entries；(c) Group GEMM 统一 kernel launch 并发执行保留 expert，离线 profiling 确定最优 tile size。实现 prefilling 2.16× 加速，decoding 1.26× 加速（Qwen3-VL-MoE-30B-A3B-Instruct, 88% skip）。
  - **硬件架构层**：单张 H200 GPU。MoDES 的 overhead 仅来自 top-k 列表上的 element-wise 操作（batch-free comparison），对 warp divergence 影响极小。

- baseline方法是什么？
  Baseline 为 LoRA-MoE（如 MoLORA），m 个独立的 LoRA expert（每个 expert = A^i B^{iT}），router $\mathcal{R}(\mathbf{x})$ 对 m 个 expert 加权。以 MoLORA 16×4（4 experts, rank=4 per expert）为例说明全栈执行路径：
  - **算法层（PEFT 微调）**：预训练 LLM（Gemma 2B）冻结权重 W0，每层注入 LoRA-MoE 模块。输入 x → 每层的 m 个 expert 各自执行 $\mathbf{x}\mathbf{A}^i\mathbf{B}^{iT}$（各 expert 独立拥有一对 A^i, B^i 矩阵）→ router softmax ($\mathbf{x}\mathbf{W_R}$) 输出 m 维权重 → 加权求和 → 与 frozen output 相加。**缺陷**：每个 expert 拥有独立的 down-projection 矩阵 A^i，造成参数冗余——PCA 分析表明不同任务的 down-projection 向量高度聚类（task-agnostic），而 up-projection 向量分散（task-specific）。
  - **系统框架层**：标准 HuggingFace Transformers 或类似 LLM 训练框架加载预训练 Gemma 2B 权重，注入 PEFT adapter（仅训练 adapter 部分）。论文未明确说明具体框架。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。Gemma 2B 规模较小（2B），通常使用标准 PyTorch kernel 即可。
  - **硬件架构层**：论文未明确说明硬件平台。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文 MoDE 方法通过两项创新解决 LoRA-MoE 的冗余问题：
  1. **共享 down-projection 矩阵 A**：观察发现不同任务/experts 的 down-projection 向量在 PCA 空间中聚类（Figure 3），说明 down-projection 是 task-agnostic 的，无需每个 expert 独立学习。MoDE 让所有 expert 共享一个 A，将参数从 $m \cdot P \cdot r$ 降至 $P \cdot r$。仅此改进（LoRA-MoE-SD / MoLORA-SD）就用 36% 的参数实现了 0.88% ROUGE-L 提升。
  2. **原子 rank-one adapter + fine-grained routing**：LoRA-MoE-SD 虽省参数，但 router 只能提供 m 种选择（所有 r 个 rank 维度绑定在一起路由）。MoDE 将 LoRA 更新分解为 dyadic sum $\Delta\mathbf{W} = \sum_{j=1}^r (\mathbf{a}_j \otimes \mathbf{b}_j)$，对每个 rank j 独立设置 m 个 expert $\{\mathbf{b}_j^1, ..., \mathbf{b}_j^m\}$ 并通过 router $\mathcal{R}_j$ 独立选择，共 $m \times r$ 个 rank-one expert，可表达 $m^r$ 种组合（vs baseline 的 m 种）。这使 MoDE 能动态组合出针对不同输入的专用 up-projection 矩阵。

  MoDE 全栈执行路径（与 baseline 同框架对应）：
  - **算法层（PEFT 微调）**：输入 x 进入 Transformer layer → frozen 前向 $\mathbf{xW_0}$ → 共享 A down-project 得 $\mathbf{h} = \mathbf{xA} \in \mathbb{R}^{1\times r}$ → 对每个 rank j ∈ {1..r}，独立 router softmax(x·W_{R;j}) 产生 m 维权重 → 第 j 个 rank 的 dyadic 贡献为 $\mathbf{h}_j \sum_{i=1}^m \mathcal{R}_j^i(\mathbf{x}) \mathbf{b}_j^{iT}$（标量 h_j 乘以加权 up-projection 向量）→ r 个 rank 求和得到总 adapter 输出 → 与 frozen 输出相加。与 baseline 的关键差异：baseline 中整个 r 维的 up-projection B^i 被 router 绑定为一个整体选择；MoDE 允许"B 的第 1 列用 expert 1，第 2 列用 expert 3，第 3 列用 expert 2，第 4 列用 expert 1"这样的细粒度组合。
  - **系统框架层**：与 baseline 相同，标准 LLM 训练框架。论文未明确说明具体框架。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。MoDE 的 rank-one 操作可视为矩阵-向量乘，标准 PyTorch 即可高效执行。
  - **硬件架构层**：论文未明确说明硬件平台。

  **实验结果证明**：MoDE 16×4 达到 60.00 ROUGE-L（vs LoRA 64: 56.11, MoLORA 16×4: 57.77, MoLORA-SD 16×4: 58.28）。Task-level win rate 分析：MoDE vs LoRA 78%, vs MoLORA 73%, vs MoLORA-SD 68%。Iso-parametric 下最佳配置为 MoDE 12×16×8（ROUGE-L 60.94）。
