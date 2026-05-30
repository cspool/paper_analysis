## LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

- baseline方法是什么？
  Baseline是vanilla Mamba（Gu & Dao, 2023），一种selective state space model (SSM)，通过time-variant的隐藏状态更新实现线性复杂度的序列建模。Mamba每个block的计算流程: X = σ(Conv1D(Linear₁(I))) → Y = SSM(X) → O = Linear₃(σ(Linear₂(I)) ⊙ Y)。SSM核心递归公式为 H_t = Ā_t ⊙ H_{t-1} + B̄_t ⊙ X_t, Y_t = C_t^T H_t，其中Ā_t = exp(Δ_t ⊙ A) ∈ (0,1)^{d_s×d_e}是隐藏状态衰减因子（A为负矩阵保证Ā_t<1），B̄_t = Δ_t ⊗ B_t决定当前token的更新量。此外对比DeciMamba（Ben-Kish et al., 2024），一种逐层token pruning方法，在更深层逐步减少序列长度。

  Baseline全栈执行例子（vanilla Mamba-1.4B推理时处理长序列，S=16000 tokens）：
  - 算法pipeline：输入序列I ∈ R^{S×d_m} → 逐层Mamba block处理（每层: Linear₁投影 → Conv1D(因果卷积, kernel=4) → SiLU激活 → SSM递归计算 [对每个token t: 输入X_t → 计算Δ_t=Softplus(X_t), B_t/C_t=Linear₄(X_t) → Ā_t=exp(Δ_t⊙A), B̄_t=Δ_t⊗B_t → H_t=Ā_t⊙H_{t-1}+B̄_t⊙X_t (H_t∈R^{d_s×d_e}, 固定大小隐藏状态) → Y_t=C_t^T H_t] → ⊙ SiLU(Linear₂(I)) → Linear₃输出投影 → residual）→ LM head → next token。每token O(1)计算，隐藏状态H∈R^{d_s×d_e}固定大小。但当S≫L（训练长度=2k）时，全局通道的累积衰减∏_{k=1}^S Ā_k因指数衰减趋向于零（Eq.12: exp((ΣΔ_k)⊙A) with A<0），导致隐藏状态H_t中早期token的信息完全丧失，全局通道的感受野无法扩展到全序列长度（图1b）。
  - 系统框架：PyTorch + HuggingFace Transformers（mamba-ssm库），直接加载官方预训练checkpoint。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用Mamba官方CUDA kernel的selective scan实现）。
  - 硬件架构：NVIDIA A5000/A100 GPU，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. **全局通道感受野无法泛化到更长序列**：通过per-channel attention map可视化（图1），论文发现Mamba隐藏状态通道可分为局部通道（感受野短于训练长度，仅关注临近上下文）和全局通道（感受野覆盖训练长度，捕获全局信息）。但当输入序列长度显著超过训练长度（如16k vs 2k），全局通道的感受野无法自适应扩展（图1b：全局通道(iv)/(v)在2k长度上的红色边框在16k长度下萎缩），导致它们失去了捕获全局信息的能力——这是Mamba长上下文性能差的关键瓶颈。
  2. **指数衰减导致隐藏状态记忆消失**：Mamba的Ā_t∈(0,1)使得每次更新都在衰减历史信息。累积衰减 ∏_{k=1}^S Ā_k = exp((Σ_{k=1}^S Δ_k) ⊙ A) 随S增大而指数级趋近于零（A为负矩阵）。当S≫L时，早期token对H_S的贡献几乎为零，即使在全局通道中也是如此。
  3. **DeciMamba的token pruning无差别对待所有通道**：DeciMamba对所有隐藏状态通道无差别地prune token，没有区分局部和全局通道的不同需求——局部通道本来就不需要处理长上下文（它们专门处理局部信息），强制在它们上面prune不如集中优化全局通道的感受野。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出LongMamba——一种training-free技术，分为两个步骤：(a) 通过训练长度上的累积衰减 ∏_{k=1}^L Ā_k > θ 来识别全局通道；(b) 对于全局通道，通过token filtering（跳过Δ_t低于阈值g(S)的token的隐藏状态更新）来扩大感受野，使筛选后的累积衰减与训练长度对齐：∏_{i=1}^S Ā'_i ≈ ∏_{i=1}^L Ā_i。

  论文方法全栈执行例子（LongMamba-enhanced Mamba推理时处理长序列，S=16000 tokens，训练长度L=2000）：
  - 算法pipeline：
    Step 1 (离线标定，仅运行一次): 从Pile采样5条序列(各2000 tokens) → 计算每个通道c在训练长度L上的累积衰减 decay_c = ∏_{k=1}^L Ā_k[c] → 若decay_c > θ则标记为全局通道 → 记录全局通道中各token的Δ_t分布 → Clamp极值到top C% → 数值求解每个S=1000,2000,...下的阈值g_c(S)使得∏_{i=1}^S Ā'_i(g)[c] ≈ decay_c
    Step 2 (推理，对每个token t):
    - 标准Mamba预处理：X_t → Conv1D → SiLU → Δ_t, B_t, C_t → Ā_t, B̄_t
    - 对每个通道c:
      if c是全局通道:
        查表得 g = g_c(round_to_nearest_1000(S))
        if Δ_t[c] < g:
          Ā'_t[c] = 1, B̄'_t[c] = 0  # 跳过该token：H_t[c] = H_{t-1}[c]
        else:
          Ā'_t[c] = Ā_t[c], B̄'_t[c] = B̄_t[c]  # 正常更新
      else:  # 局部通道保持原样
        Ā'_t[c] = Ā_t[c], B̄'_t[c] = B̄_t[c]
    - H_t = Ā'_t ⊙ H_{t-1} + B̄'_t ⊙ X_t
    - 标准Mamba输出：Y_t = C_t^T H_t → ⊙ gating → output
  - 系统框架：PyTorch + 修改的Mamba前向传播（在SSM核心循环中插入token filtering逻辑）。代码开源：https://github.com/GATECH-EIC/LongMamba。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。LongMamba在SSM的递归循环中插入了per-channel的条件判断，不影响底层scan kernel。
  - 硬件架构：NVIDIA A5000/A100 GPU，论文未涉及RTL/模拟器层面。延迟开销≤4.5%（表6-7）。

  关键设计选择映射到缺陷：
  - 缺陷1（全局通道感受野无法泛化）→ 通过累积衰减阈值分类识别全局通道，仅对这些通道施加token filtering。直觉上，全局通道的隐藏状态需要存储长期信息，而每个不重要token的贡献都会累积额外的衰减（Ā_t < 1），导致历史信息迅速消失。通过跳过不重要token（Δ_t<g → Ā'_t=1），有效减少了衰减累积次数，使隐藏状态能在更长序列上保持早期信息。图1可视化对比展示了LongMamba处理后全局通道的感受野可扩展到16k tokens。
  - 缺陷2（指数衰减导致记忆消失）→ LongMamba通过token filtering使"有效衰减步数"保持在≈L的规模，而非S。核心对齐公式 ∏_{i=1}^S Ā'_i ≈ ∏_{i=1}^L Ā_i 将OOD长序列输入统计量变换为ID样本统计量。具体地，通过设置Ā'_t=1（即不衰减也不更新）来"跳过"那些不重要token——这些token的比例约等于(S-L)/S，使得筛选后的累积衰减（仅计算被保留token的Ā_t）与训练时的衰减量相似。
  - 缺陷3（DeciMamba无差别pruning）→ LongMamba区分对待全局和局部通道。局部通道专门处理局部上下文，不需要长感受野——因此它们完全不施加token filtering，保持对局部上下文的完整建模能力。对比DeciMamba的逐层pruning对所有通道同等对待，LongMamba的差异化策略带来显著性能优势（LongBench-E上Mamba-1.4B: LongMamba 17.33% vs DeciMamba 13.38%，提升3.95个百分点）。

  标定机制的自适应能力：
  - g(S)查找表per-channel构建，使每个全局通道根据其自身的Δ_t分布获得个性化的过滤阈值
  - Δ_t可解释为token的"重要性度量"（Mamba中Δ_t越大，该token对隐藏状态的更新贡献越大）。因此过滤Δ_t<g的token等价于"只让重要token更新全局通道的隐藏状态"
  - 标定使用Pile训练集数据，建立的是"训练分布下Δ_t的统计量"，推理时用此统计量决定哪些token值得全局通道记住
  - 消融实验（表4）验证了标定序列选择的鲁棒性——10组不同随机种子采样的校准序列在LongBench-E上STD仅为0.42%，说明Δ_t分布在不同序列间高度一致

  模型间的策略差异（体现方法灵活性）：
  - Mamba-1.4B：θ=10⁻³⁰, C=20（非常极端的阈值——只有极少数通道是全局通道，且大量clamping）
  - Mamba2-1.3B：θ=5×10⁻², C=5（较宽松的阈值——较多通道被识别为全局通道，少量clamping）
  - Zamba2-1.2B：θ=10⁻⁵, C=5（中等阈值——因混合Transformer-SSM架构影响通道分布）
  这些差异表明LongMamba自动适应不同模型的内部通道统计特性，无需人工调整。
