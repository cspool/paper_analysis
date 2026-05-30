## SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

- baseline方法是什么？
  Baseline是两类现有高效注意力方法：(1) **Sparse Attention方法**（VSA、VMoBa、SpargeAttn等）——通过mask跳过部分注意力分数的计算，仅保留top-k最重要的注意力权重。缺陷：实践中稀疏度很少超过90%，通常在40-60%（短序列）或80-85%（长序列100K-300K）。根本原因在于注意力权重分布中，约8.1%的权重大于平均值1/N（主导分布），而中间约47%的权重在1/(100N)到1/N之间——跳过这些"中间值"会导致显著误差（相对L1 error从3%跃升至33%），但保留它们又严重降低稀疏度。这形成了一个"稀疏度-准确率"的trade-off dilemma。(2) **Linear Attention方法**（SANA、DiG等）——通过解耦softmax将复杂度降至O(N)（如φ(Q)φ(K)^T替代softmax(QK^T)）。缺陷：在视频扩散模型中严重失效。根本原因在于full attention权重具有高rank（stable rank远大于d），而线性注意力本质上是rank≤d的低秩近似，无法准确逼近高秩的softmax注意力分布。

  全栈执行例子（以Wan2.1-1.3B视频生成，30K tokens，Sparse Only baseline [85% sparsity]为例）：
  - 算法层：Sparse Only保留top 15%注意力权重（按绝对值排序），mask掉85%的小权重。对每个Q block Q_i，仅对M[i,j]=1的K_j/V_j blocks执行完整FlashAttention计算。但85%稀疏度下：被保留的15%中大部分是"中间值"（在1/(100N)到1/N之间），这些值对最终输出的贡献不够大却被分配了完整的O(N²)计算资源（每个block 64×64 full GEMM + softmax normalization）；而被mask掉的85%中包含约45%极度小值（<1/(100N)），这些值的mask掉是正确的，但另外40%的mask掉导致不可忽略的信息损失——累积误差在逐denoising step中传播。
  - 系统框架层：PyTorch + 自定义sparse FlashAttention kernel。对每个attention层：计算full QK^T → 在Q block级别按行排序取top-k → 生成block mask → sparse FlashAttention执行masked计算。Mask生成需要额外的full QK^T计算或简化的pooling预测。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Sparse FlashAttention kernel在RTX 5090上执行。对N=30K、sparsity=85%：实际计算约15%的QK^T GEMM和PV MM，以及全部softmax normalization（因为需要正确归一化被保留的部分）。Kernel包含mask检查逻辑和稀疏迭代。
  - 硬件架构层：NVIDIA RTX 5090 GPU。Sparse attention的GPU利用率受限于非规则的mask pattern——被跳过的块产生warp divergence，且mask检查逻辑产生额外指令开销。

  Baseline核心缺陷：
  1. **稀疏度天花板**：注意力权重的幂律分布使85-90%成为实践中稀疏度的soft limit——超过此值，被mask的"中间值"累积误差使生成质量急剧下降（从<3% L1 error跃至>33%）。这限制了稀疏注意力能将计算量降到的最低水平。
  2. **线性注意力高秩不匹配**：在视频扩散模型中，full attention权重具有远高于d的stable rank，而线性注意力的表达能力上限为rank d。这一rank gap使线性注意力在视频生成场景中完全失效（Linear Only的VA=0.042 vs Full Attention=76.78）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出SLA（Sparse-Linear Attention），通过对注意力权重的三级分类和混合计算策略，从根本上打破稀疏注意力的"稀疏度天花板"：

  **(1) 三级分类替代二级分类**——解决"稀疏度天花板"：
  SLA的核心洞察：注意力权重可以被分解为两部分——少量大值的sparse component（高rank，约8%）+大量小值的low-rank component（极低rank，约92%）。这解释了为什么sparse only和linear only都失败：sparse attention需要处理所有非极小的权重（包括那些"中间值"），而linear attention不需要处理高rank的大值部分。SLA将注意力权重分为三级而非传统的二级（保留/跳过）：
  - Critical（top k_h=5%）：高rank的大值 → 执行完整的O(N²) FlashAttention
  - Marginal（中间~85%）：低rank的小值 → 执行O(N)线性注意力（仅需d×d矩阵加法，占full attention <0.5% cost）
  - Negligible（bottom k_l=10%）：极度小值 → 完全跳过
  三级分类的关键意义：**sparsity从85%（仅保留critical）跃升至95%（跳过negligible），而marginal块用几乎免费的线性注意力替代，避免了"中间值"的dilemma**。对比baseline Sparse Only（85%稀疏度=跳过85%），SLA在95%稀疏度（跳过10% negligible + 用O(N)处理85% marginal）下实际计算量约为Sparse Only的一半（因为linear attention几乎免费），同时质量更好。

  **(2) 可学习融合而非简单叠加**——解决"线性注意力不匹配"：
  SLA不是简单地将稀疏和线性注意力的输出相加（L+S baseline的VA=29.65 vs SLA=76.96），而是设计了两个关键机制：
  - 可学习投影Proj(O^l)：对线性注意力输出应用可学习线性变换R^d→R^d，减少softmax和线性注意力之间的分布不匹配。这使线性注意力从"直接近似"转变为"learnable compensation"。
  - Fine-tuning：通过少量fine-tuning steps（2000步，<0.1% pretraining cost），模型参数自适应学习如何利用线性注意力作为稀疏注意力的补充。Fine-tuning使模型学会"信任"线性注意力的补偿，而非仅依赖稀疏部分。

  **(3) 预计算和单kernel融合**——实现前三级的实际加速：
  SLA通过预计算h_j = φ(K_j)^T V_j和z_j = rowsum(φ(K_j)^T)，使marginal块的线性注意力计算降为单次矩阵加法（而非每次重新计算φ(Q)φ(K)^T V）。这确保了95%稀疏度下的理论加速能转化为实际wall-clock speedup——13.7× kernel speedup vs FlashAttention2，2.2× end-to-end speedup。

  全栈执行对比baseline（以Wan2.1-1.3B同一视频生成，SLA 95% sparsity为例）：
  - 算法层：同一DiT架构，但每个注意力层替换为SLA。预测压缩mask P_c → 三级分类(M_c=1/0/-1) → critical块用FlashAttention（5%块数）→ marginal块用线性注意力precomputed H_i/Z_i加法（85%块数）→ negligible跳过（10%块数）→ Proj(O^l)融合。SLA @ 95% sparsity (2.73T FLOPs) vs Full Attention (52.75T FLOPs) = 19.3× FLOP reduction，视频质量VA=76.96 vs Full=76.78（无退化）。
  - 系统框架层：PyTorch + 自定义fused SLA CUDA kernel。Fine-tuning on 20K private videos × 2000 steps × batch 64。Fine-tuning cost << 0.1% pretraining。
  - 编译框架层：论文未明确说明（直接CUDA kernel实现，无编译框架层）。
  - kernel调度层：单CUDA kernel融合三种计算：
    1. Critical块：OnlineSoftmax FlashAttention（与FlashAttention2相同的GEMM+softmax+GEMM pipeline）
    2. Marginal块：单次矩阵加法（H_i += h_j）和向量加法（Z_i += z_j），预计算保证极低开销
    3. Forward: 13.7× vs FlashAttention2 forward; Backward: 6.8× vs FlashAttention2 backward
    4. 额外效率优化：Lookup table（预处理稀疏mask非零位置）、Pre-aggregation（减法替代加法）、Method of Four Russians（分组预计算子集和）
  - 硬件架构层：NVIDIA RTX 5090。SLA kernel的GPU利用率特征：critical块期间Tensor Cores活跃（GEMM），marginal块期间仅CUDA Cores做加法（几乎瞬时），negligible块完全跳过。由于marginal块计算量极低（<0.5% full attention），GPU大部分时间在critical块的Tensor Core密集型计算上，有效利用GPU算力。端到端attention时间从97s降至11s（8.8× reduction），非attention部分（MLP/RMSNorm/Conv）不变，总体2.2× end-to-end speedup。

  设计思路核心：
  论文的根本洞察是**注意力权重矩阵的谱分解**——P = (P ⊙ M) + (P ⊙ (1-M))，其中sparse component P⊙M保持高rank（需要完整O(N²)计算），而low-rank component P⊙(1-M)可用rank-d线性注意力准确近似。这解释了为什么单独使用sparse attention或linear attention都无法成功——它们各自试图用单一机制处理具有本质不同数学结构的两个成分。SLA的统一框架通过三级分类实现"对的结构处理对的成分"：高rank部分用sparse attention保持精度，低rank部分用linear attention换取效率。Proj层和fine-tuning进一步解决了两个成分输出分布不匹配的问题，使融合后的输出与full attention保持一致。关键实验证据：Figure 3直观展示了去除top 8%值后的剩余矩阵stable rank从~150降至~20，实证验证了low-rank分解假设。
