## Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

- baseline方法是什么？
  **Vanilla Masked Diffusion Model (MDM) with τ-leaping 独立并行解码**：当前开源的Diffusion LLM（LLaDA、Dream）使用基于τ-leaping近似的掩码扩散模型进行序列生成。在推理时，模型从全[MASK]序列开始，通过多步迭代逐步将[MASK]token替换为真实token。默认最优策略是每步解码1个token（顺序解码），因为τ-leaping虽然允许一次解码多个token，但存在conditional independence assumption问题——多token从独立边际分布中采样，破坏了token间真实联合分布中的依赖关系（如"high card" vs "high house"的不合理组合）。同时，由于Diffusion LLM使用full bidirectional attention，无法像自回归模型那样使用KV Cache复用之前的attention计算结果，每步都需要对全序列重新计算attention。

  全栈执行例子（LLaDA-Instruct GSM8K 5-shot, gen_len=256, A100 GPU）：
  - **模型推理算法层**：MDM使用absorbing-state离散扩散（Equation 1: q_{t|0} = Cat(x_t^i; (1-t)δ_{x_0^i} + tδ[MASK])），loss为MDM ELBO（Equation 2）。推理时使用τ-leaping近似反向过程（Equation 3），每步选择置信度top-1的token解码（1 token/step），共需约256步完成生成长度256。
  - **系统框架层**：直接使用LLaDA官方inference脚本，不使用任何serving框架（无vLLM/TensorRT-LLM）。batch size=1单请求推理。
  - **编译框架层**：论文未明确说明。使用标准PyTorch forward pass，无自定义编译优化。
  - **kernel调度层**：标准PyTorch attention实现（full bidirectional attention），每步执行Q·K^T/V计算于全序列矩阵（尺寸(|p|+L)×d），无KV cache无法复用前缀attention结果。256 step各自独立执行全注意力计算。
  - **硬件架构层**：NVIDIA A100 80GB GPU，无自定义硬件。吞吐量约6.7 tok/s（GSM8K 5-shot gen_len=256）。
  
  Baseline缺陷：
  - (a) **无KV Cache导致重复全注意力计算**：每步都需对prompt + 已生成 + 未生成的全序列重新计算Q·K^T和softmax，计算量O(T·(|p|+L)²·d)，T为总解码步数。当prompt较长（如8-shot）和生成长度较大（512/1024）时开销巨大。
  - (b) **τ-leaping并行解码破坏token依赖**：多token同时从p(X|E)=Π_i p_j(X_{i_j}|E)独立采样，忽略了真实联合分布p(X|E)中包含的token间条件依赖。导致生成不合理的token组合，质量随每步并行token数增加而显著下降。
  - (c) **无动态并行控制机制**：LLaDA的baseline要么全顺序（1 token/step，慢但准确），要么固定top-K解码（K token/step，快但质量下降），缺少根据模型置信度自适应调节并行粒度的机制。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Fast-dLLM：通过Block-wise Approximate KV Cache + Confidence-Aware Parallel Decoding 两种互补技术，训练无关地加速Diffusion LLM推理**。核心设计：(i) 分块生成（block-wise generation）+ 近似KV Cache（PrefixCache/DualCache），利用相邻步KV激活的高余弦相似度实现cache复用；(ii) 置信度感知并行解码（threshold/factor策略），通过理论保证（Theorem 1）仅在高置信度时并行解码多token，平衡速度与质量。

  全栈执行例子（LLaDA-Instruct GSM8K 5-shot, gen_len=256, A100 GPU, Fast-dLLM PrefixCache+Threshold τ=0.9, B=32）：
  - **模型推理算法层**：同一LLaDA MDM模型结构和权重，不修改模型参数（训练无关）。流程变为：
    1. 首步全序列forward pass，缓存prefix（prompt部分）的K/V矩阵
    2. 将生成拆分为K=⌈256/32⌉=8个块，每块最多T步解码
    3. 块k内：复用prefix K/V cache，仅对块内token执行attention→置信度计算→阈值过滤→多token并行解码
    4. 块k完成后：全序列forward pass，更新prefix K/V cache（与标准forward融合，无额外开销）
    5. 总步数显著减少（平均约40步 vs baseline 256步），吞吐量达到54.4 tok/s（vs baseline 6.7 tok/s，8.1×加速）
  - **系统框架层**：Fast-dLLM v1代码在PyTorch层实现，修改LLaDA推理loop：添加cache管理（存储/复用prefix K/V或prefix+suffix K/V），替换逐token顺序解码为块内自适应并行解码。不依赖serving框架。
  - **编译框架层**：论文未明确说明。标准PyTorch eager execution，无编译优化修改。
  - **kernel调度层**：attention计算量从O(T·(|p|+L)²·d)降至近似O(|p|²·d + K·T'·(B²+|p|·B)·d)，其中B为块大小（32），T'为每块内步数（远小于原始T），K为块数。DualCache进一步消除suffix attention计算，仅保留B×B块内自注意力。
  - **硬件架构层**：同一NVIDIA A100 80GB GPU，无自定义硬件。Fast-dLLM吞吐量54.4 tok/s vs baseline 6.7 tok/s（8.1×加速，GSM8K gen_len=256），8-shot gen_len=1024时DualCache达27.6×加速（0.7→19.3 tok/s）。

  关键设计选择与baseline缺陷的对应：
  - **defect (a): 无KV Cache → 重复全注意力计算** → 方案：Block-wise Approximate KV Cache。利用观察——相邻步KV激活余弦相似度接近1（Figure 3），在块内复用prefix/suffix K/V。块完成时更新cache（与解码forward融合，无额外计算）。块大小32在速度-精度间取得最佳折中（Figure 4）。DualCache变体进一步缓存suffix（全[MASK]），消除交叉注意力计算。
  - **defect (b): τ-leaping独立采样破坏token依赖** → 方案：Confidence-Aware Parallel Decoding。理论分析（Theorem 1）证明：当每token置信度>1-ε且(n+1)ε≤1时，argmax的乘积边际分布等价于argmax的真实联合分布。基于此，设计threshold策略（仅解码c_i>τ的token）和factor策略（动态选择满足(n+1)(1-c^(n))<f的最大n个token），在高置信度时安全并行解码多token，低置信度时保守解码。
  - **defect (c): 无动态并行控制** → 方案：threshold和factor两种策略都根据当前步的模型置信度水平动态决定并行解码的token数量。Threshold策略自适应在1到B个token之间调节；Factor策略通过理论绑定量(n+1)(1-c^(n))进一步精确控制并行度。对比固定token-per-step baseline，动态策略在相同accuracy水平下decodes significantly fewer NFEs（Figure 5c, Figure 8c）。
  - **额外设计：理论与实践的桥梁**：Theorem 1不仅给出argmax等价条件，还给出L_p距离和KL散度的上界（D_TV < (3n-1)ε/2, D_KL < (n-1)[H_b(ε)+ε·ln(|V|-1)]），量化了乘积分布对真实联合分布的逼近程度，为实际部署中选择阈值/因子提供了理论基础。
  - **额外设计：Prefill长度和生成长度的加速放大**：由于cache复用与序列长度成正比，更长的prefill（8-shot vs 5-shot）和更长的生成长度（1024 vs 256）带来更大的加速比（DualCache: 27.6× at 8-shot gen_len=1024 vs 19.6× at 5-shot），使方法在few-shot和长文本生成场景中价值更大。
