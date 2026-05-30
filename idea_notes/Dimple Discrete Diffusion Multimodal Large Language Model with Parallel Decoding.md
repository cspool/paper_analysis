## Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding

- baseline方法是什么？
  **自回归多模态大语言模型（AR MLLM）**，以LLaVA-NEXT为代表。训练使用causal attention mask + next-token prediction loss，逐token自回归生成。推理时每个forward step生成1个token，LLaVA-NEXT-7B训练数据约1.2M样本。

  全栈执行例子（LLaVA-NEXT回答"What is the common item in the two images?"）：
  - **模型推理算法层**：Vision encoder (CLIP ViT-L/14) 将两幅图像编码为visual tokens，经过projector映射到LLM embedding空间。Prompt tokens + visual tokens拼接为输入序列。使用causal attention：每个token只能attend到自身及之前的token。逐token自回归生成回答：先输出"In..."，再"the..."，依次生成直到[EOS]或max_length。total forward次数 = response token数（如64 tokens需要64次forward）。
  - **系统框架层**：自回归MLLM serving（如vLLM/SGLang）使用continuous batching + PagedAttention管理KV cache。每次forward迭代：所有batch中的请求各生成1个token → 更新KV cache → 检查是否[EOS] → 移出完成的请求。Prefill阶段一次处理prompt的KV cache计算。
  - **编译框架层**：论文未明确说明。通常使用PyTorch + FlashAttention-2进行高效attention计算。
  - **kernel调度层**：论文未明确说明。自回归decode阶段为memory-bound的GEMV操作（batch_size × 1 token），每次forward处理小矩阵向量乘，GPU利用率低。
  - **硬件架构层**：NVIDIA H100 GPU，无自定义硬件修改。

  Baseline缺陷：
  - (a) **逐token生成低效**：每个forward step仅生成1个token，response length 64需要64次forward，GPU在decode阶段利用率低（memory-bound GEMV）。
  - (b) **无法并行解码**：causal attention要求严格的左到右生成顺序，无法利用token间的独立性。
  - (c) **输出控制困难**：无法精确控制输出格式和长度。自回归模型依赖[EOS]终止，无法预先指定response长度；控制输出结构需要依赖instruction prompt或CoT的间接引导。
  - (d) **无法提前给出答案**：自回归模型必须按序生成全部tokens后才能到达最终答案，无法在处理中间推理步骤时提前给出结论。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Dimple：首个离散扩散多模态大语言模型（DMLLM）**。核心设计：(i) Autoregressive-then-Diffusion混合训练范式——先AR训练学习多模态能力，再Diffusion训练恢复并行解码；(ii) Confident Decoding——基于置信度阈值γ动态决定每步解码token数；(iii) Prefilling——复用prompt KV cache降低注意力复杂度；(iv) Structure Prior——预置特定位置token实现输出结构和长度精确控制。

  全栈执行例子（Dimple回答同样问题，使用Confident Decoding + Structure Prior）：
  - **模型推理算法层**：Vision encoder (Qwen2.5-VL ViT, 冻结) 编码图像 → 2层MLP projector → Dream DLM embedding空间。初始化：设定response_length=64，所有answer位置初始化为[MASK] token。Structure Prior预置特定位置："In the first image, there "、"In the second image, there "、"The common item in the two images is"（标记为已确定，不参与mask）。Forward #1（首个迭代）：bidirectional attention over全部L_prompt+L_answer个位置，所有未确定的[MASK]位置预测token分布。计算confidence c_t^(i) = max(softmax(logits)^(i))（使用pre-revision概率，不受temperature/top-p影响）。设置阈值γ：筛选c_t^(i)≥γ的位置，一次性更新多个高置信token（如同时解码9个token）。Fallback：若无位置超过阈值，随机选K个最不确定位置采样更新。Forward #2-#7：每次forward基于上一步结果，继续在未确定位置预测+筛选。第10步（共30步）：token "scissors"（最终答案）已完成解码——答案出现在完整response之前。最终30步完成（vs baseline 64步），实际迭代数仅为response_length的~1/2至1/3。
  - **系统框架层**：Prefilling实现：首次forward计算完整attention并保存prompt tokens的K/V。后续迭代复用保存的prompt K/V（仅计算answer部分的attention），复杂度从O((L_prompt+L_answer)²)降至O(L_answer²)。注意：由于DMLLM使用full bidirectional attention，answer token会attend到prompt tokens，prefilling理论上不是lossless（但实验证明性能下降仅平均0.8%）。生成不需要[EOS]终止——通过Padding token填充到预定义response_length。
  - **编译框架层**：论文未明确说明。使用PyTorch标准框架。
  - **kernel调度层**：论文未明确说明。Confident Decoding将多次memory-bound GEMV合并为更大的矩阵运算（每步同时处理多个token），提升了decode阶段的算术强度。
  - **硬件架构层**：NVIDIA H100 GPU集群（训练约100 GPU hours），单H100用于Prefilling消融实验。无自定义硬件修改。

  关键设计选择与baseline缺陷的对应：
  - **defect: 逐token生成低效 (a)** → 方案：Confident Decoding在高置信度时一次更新多个token。Table 5展示22个token仅需7次迭代（~1/3）；Table 6展示55个token需37次迭代。配合Prefilling，batch=32下TPS加速达7×（Table 3），batch=1下加速1.5×-2×。对比自回归需64步完成64-token response，Dimple仅需~30步。
  - **defect: 无法并行解码 (b)** → 方案：离散扩散模型使用full bidirectional attention，所有位置可同时attend和预测。训练时Phase II使用masked language modeling + absorbing-state diffusion，推理时所有[MASK]位置并行预测、基于置信度选择性更新。
  - **defect: 输出控制困难 (c)** → 方案：(i) Structure Prior允许预置任意位置的token，如"{date:"、"time:"、"}"定义JSON格式输出（Table 5）。(ii) Length Control——通过response_length参数精确控制输出长度，并在指定位置放置结束标记（如"Thus, the answer is \box{"在position[-12:-4]），模型自动调整推理跨度填满token预算（Table 6：length=16和32的两种配置）。
  - **defect: 无法提前给出答案 (d)** → 方案：扩散模型可在任意位置先解码出高置信token，不限于从左到右。Table 4展示正确答案"scissors"在第10步解码，而整个response在第30步才完成——答案先于完整推理步骤出现，这是自回归模型无法实现的。
  - **额外设计：训练不稳定性** → 方案：Autoregressive-then-Diffusion训练。纯扩散训练存在两个低效：(i) masked language modeling仅对masked token计算loss，监督信号覆盖率低于next-token prediction；(ii) 每个样本仅提供一个timestep的监督。AR-then-Diffusion先用AR训练建立多模态能力（更高监督信号利用率），再用Diffusion训练恢复并行解码。Table 2证明AT+DT在所有9个benchmark上优于纯DT，且缓解了Length Bias（纯DT在ChartQA上accuracy从42.7%→8.6%随response_length增加）。
