## BLASST Dynamic BLocked Attention Sparsity via Softmax Thresholding

- baseline方法是什么？
  **Dense FlashAttention（FlashAttention-3/4）**：在GPU上通过tiled block-wise online softmax算法计算完整attention矩阵Attention(Q,K,V)=softmax(QK^T/√d_k)·V。所有QK block（T_r×T_c个）都完整计算softmax指数和PV矩阵乘法，所有Value block都从HBM加载。

  全栈执行例子（Llama-3.1-8B，prefill batch=1，128K sequence length，B200 GPU）：
  - **模型推理算法层**：标准scaled dot-product attention，Q/K/V线性投影后计算全量QK^T ∈ R^{128K×128K}，softmax归一化后乘以V。MHA: 32个head独立计算；GQA: 8个KV head共享。
  - **系统框架层**：TensorRT-LLM或FlashInfer调用FlashAttention-3 kernel。In-flight batching（concurrency 64）混合调度prefill和decode请求。用户不可见attention内部sparsity pattern，所有token平等对待。
  - **编译框架层**：论文未明确说明编译框架层修改。TensorRT-LLM的graph optimization pass选择FlashAttention kernel作为attention实现。
  - **kernel调度层**：FlashAttention-3 prefill kernel pipeline：BMM1(QK^T) → softmax(EX2+rowsum) → BMM2(PV)，在warpgroup级别重叠tensor core MMA和CUDA core softmax。每轮mainloop迭代处理一个KV block。Decode kernel：V load from HBM → BMM1(单query QK^T) → BMM2(PV)，memory-bound于HBM带宽。Kernel内部无sparsity判断——所有B_c个score都参与softmax和PV乘法。
  - **硬件架构层**：标准NVIDIA B200/H200 GPU。Tensor core执行FP16/BF16 MMA（QK^T和PV）；CUDA core执行MUFU.EX2指数、FMUL乘法、FADD加法（softmax）；HBM→SRAM的tiled加载（Q/K/V分块）。论文未明确说明硬件架构层自定义修改。

  Baseline缺陷：(1) 计算浪费：长序列中大量attention score接近零（attention分布稀疏），但仍消耗CUDA core和tensor core计算softmax和PV乘法。(2) 内存带宽浪费：decode阶段所有V block从HBM加载，但很多block的attention权重近零。(3) 无法利用attention稀疏性：现有sparse attention方法（MInference/XAttention等）依赖pre-computation、proxy scores或额外训练，引入overhead抵消理论加速。(4) 缺乏统一的prefill+decode方案：大多数方法仅优化单一阶段。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **BLASST（BLocked Attention Sparsity via Softmax Thresholding）——基于online softmax统计量的动态block级attention稀疏化**：在FlashAttention的tiled online softmax过程中，复用已计算的running maximum和block local maximum统计量，通过简单阈值比较（m̃_i^{(j)} - m_i^{(j)} < ln(λ)）识别可跳过的block，跳过其softmax计算、PV乘法和V加载。**零额外推理开销**——skip decision复用的统计量在FlashAttention中本来就要计算。

  全栈执行例子（同样Llama-3.1-8B，prefill batch=1，128K sequence length，B200 GPU，target sparsity=50%）：
  - **模型推理算法层**：attention计算逻辑变为——对每个QV block i和KV block j，(a) BMM1计算S_ij = Q_iK_j^T（不变），(b) rowmax(S_ij)得到local max（本来就计算），(c) 更新running max（本来就计算），(d) 阈值判断：if local_max - running_max < ln(λ) → skip，否则正常softmax+PV。通过跳过50%约block，减少50%的CUDA core EXP计算和tensor core MMA计算。**无需训练、无需pre-computation、无需proxy scores**。
  - **系统框架层**：BLASST已集成到TensorRT-LLM和FlashInfer的attention接口中，作为FlashAttention的drop-in replacement，仅增加一个scalar threshold参数λ。用户通过calibration确定λ = α·exp(β·S)/L，其中S为目标sparsity。In-flight batching正常工作，prefill和decode请求均可受益。端到端TTFT和TPOT均有1.1×加速（Qwen3-30B，LongBench V1）。
  - **编译框架层**：论文未明确说明编译框架层修改。BLASST kernel直接替换框架中的attention kernel调用，接口兼容。
  - **kernel调度层**：
    - Prefill kernel（compute-bound优化）：BMM1照常计算所有block → skip check（predicate+VOTE+ATOMIC，隐藏于BMM1后）→ 被跳过block省略EX2 softmax和BMM2(PV)，tensor core直接进入下一轮BMM1。Pipeline从18 time units压缩到14 time units（50% sparsity, 4轮迭代）。Speedup从1.25×（39% sparsity）到1.77×（94% sparsity）on B200。
    - Decode kernel（memory-bound优化，batched load scheduling）：连续发射多个K^TQ的BMM1 → 批量skip check → 仅发射通过检查的V_j loads → 再执行对应的BMM2。消除naive sequential pipeline的scoreboard dependency bubble。Speedup从1.18×（37% sparsity）到1.79×（92% sparsity）on B200。
  - **硬件架构层**：标准GPU硬件，无自定义硬件修改。skip check实现仅需warp-level VOTE指令 + 单线程ATOMIC to shared memory，均由现有GPU指令集原生支持。论文未明确说明硬件架构层自定义修改。

  关键设计选择与baseline缺陷的对应：
  - **defect: 计算浪费（大量近零attention scores仍被计算）** → 方案：复用online softmax已有的running maximum统计量进行block级阈值判断（m̃_i^{(j)} - m_i^{(j)} < ln(λ)），跳过被剪枝block的softmax和PV乘法。Skip decision仅添加predicate+VOTE+ATOMIC几条指令，被pipeline隐藏。
  - **defect: 内存带宽浪费（decode阶段加载全量V blocks）** → 方案：decode kernel的batched load scheduling——先背靠背计算多个K^TQ确定skip pattern，再仅加载需要的V blocks，直接按sparsity比例减少HBM traffic。
  - **defect: 现有sparse attention方法引入overhead（pre-computation/proxy scores/training/fine-tuning）** → 方案：完全training-free和pre-computation-free。所有sparsity decision基于online softmax内部已有统计量，是"免费"的byproduct。Table 1对比了所有方法的特性——BLASST是唯一同时支持prefill+decode加速且无需training和pre-computation的方法。
  - **defect: 缺乏统一的prefill+decode方案** → 方案：同一算法框架下设计了两套specialized kernel——prefill kernel优化compute-bound场景（跳过softmax+MMA），decode kernel优化memory-bound场景（跳过V loading+softmax）。两者共享相同的skip判断逻辑，使用同一套阈值校准参数。
  - **defect: 固定阈值在不同context length下sparsity不稳定** → 方案：校准pipeline（Algorithm 2）发现λ·L = α·exp(β·s)的指数关系，其中λ与L成反比。用户仅需指定目标sparsity S，kernel自动按context length适配阈值，sparsity偏差仅~1.2%（Table 6）。
  - **defect: 超高sparsity下accuracy退化** → 方案：sparsity-aware training（fine-tuning forward pass中应用BLASST），模型在训练中学到将重要信息集中在高attention score block中，accuracy退化降低至1.7×（Figure 6）。
