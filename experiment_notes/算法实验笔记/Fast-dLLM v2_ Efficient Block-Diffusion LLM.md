## Fast-dLLM v2: Efficient Block-Diffusion LLM

- 属于算法pipeline的实现是什么？实验比较什么？
  实现分为训练和推理两部分：(1) **Block Diffusion训练配方**——将预训练AR模型（Qwen2.5-Instruct）适配为block diffusion LLM，仅需~1B tokens微调（vs Dream的580B tokens）。核心设计包括：block-wise组织（block size=32，序列对齐到block整数倍并packing）、complementary masking（每个样本两个互补mask view，确保所有token既在masked又在unmasked上下文中被训练）、token shift（masked位置使用i-1位置的hidden state预测token i，保留AR模型的representation quality）、block-wise attention mask（M_BD block diagonal + M_OBC offset block causal + M_BC block causal，同时支持noised和clean上下文）。训练loss为masked-token-only cross-entropy。(2) **推理pipeline**——逐block自回归解码+block级KV cache复用+块内confidence-aware并行解码（来自Fast-dLLM v1）+ DualCache子块缓存。block size=32, sub-block size=8, 默认threshold=1（关闭并行解码），加速场景threshold=0.9。

  实验比较：(i) 1.5B和7B规模在HumanEval/MBPP/GSM8K/MATH/IFEval/MMLU/GPQA的benchmark性能 vs Qwen2.5-baseline、LLaMA-3.2、SmolLM-2、Dream、LLaDA系列（Table 1）；(ii) A100和H100上不同batch size的吞吐量 vs AR baseline（Figure 5）；(iii) 不同threshold下的accuracy-throughput trade-off（Figure 4, GSM8K）；(iv) 消融实验：naive token shift vs +pad vs +pad+CM（Table 2）；(v) 不同sub-block size的影响（Table 3）；(vi) 不同block size（与训练不一致）的影响（Table 4）；(vii) sub-block cache对accuracy和throughput的影响（Figure 6）。

- 硬件平台是什么，配置是什么。
  - 训练：64 × NVIDIA A100 GPU，DeepSpeed Zero-3，1.5B模型训练约8小时，7B模型约12小时
  - 推理评估：NVIDIA A100和H100 GPU，batch size 1-64，context length=2048，block size=32，sub-block size=8
  - throughput测量使用相同硬件对比AR baseline和diffusion方法

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen2.5-Instruct 1.5B和7B（预训练AR模型微调得到Fast-dLLM v2）
  - 训练数据：LLaMA-Nemotron post-training dataset子集（高质量instruction-following数据）
  - Benchmark：HumanEval/MBPP（代码生成，EvalPlus评估）、GSM8K/MATH（数学推理）、IFEval（指令遵循）、MMLU/GPQA（知识密集型QA）
  - 评估框架：LM-Eval（非代码任务）、EvalPlus（代码任务），零样本设定（GPQA使用5-shot），greedy decoding

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文声明Code and model will be publicly released，但未提供具体链接。v1版本已开源：https://github.com/NVlabs/Fast-dLLM。

  **训练算法pipeline（以block size=32, context length L=2048为例）：**
  ```
  输入: 预训练Qwen2.5-Instruct模型参数θ, 训练数据D, block size D=32, mask概率t~Uniform(0,1)
  1: 对每个训练样本：
  2:     将序列右填充至D的整数倍（填充[MASK]不计loss）
  3:     将多个样本拼接packing至长度L，自然对齐为B=L/D个block
  4:     对每个block b采样随机binary mask m_b ∈ {0,1}^D
  5:     生成互补mask m̄_b = 1 - m_b
  6:     两个view放入同一个batch
  7:     对每个masked位置i（m_i=1处）：
  8:         使用position i-1的hidden state预测token i（token shift）
  9:     attention mask = [[M_BD, M_OBC], [0, M_BC]]，同时处理noised x_t和clean x_0
  10:     loss: -Σ 1[x_t^i=[MASK]] · log p_θ(x_0^i | x_{<i}, x_{block(i)})
  11: 使用AdamW优化器，linear warmup 500 steps
  ```

  **推理pipeline（以threshold=0.9，block size=32，sub-block size=8为例）：**
  ```
  输入: prompt p, 目标生成长度L, block size B=32, sub-block size S=8
  1: 将L拆分为K=⌈L/B⌉个block，每个block再分sub-block
  2: 初始化x = [p; [MASK]×L]
  3: for k = 1 to K:                                        // 逐block自回归
  4:     for each sub-block in block k:
  5:         forward pass（仅当前block+prefix，复用block级KV cache）
  6:         计算当前sub-block内[MASK]位置的confidence
  7:         confidence > τ的token并行解码，其余保留[MASK]
  8:         使用DualCache复用sub-block级的prefix/suffix KV
  9:     end
  10:     block k完成，刷新block级KV cache（作为后续block的prefix）
  11: end
  12: return decoded tokens
  ```

  **张量计算关键设计——Attention Mask分解（M_full ∈ {0,1}^{2L×2L}）：**
  - M_BD (Block-diagonal): [M_BD]_{ij}=1 iff i,j同属一个block → 块内双向自注意力
  - M_OBC (Offset block-causal): [M_OBC]_{ij}=1 iff j所在block在i所在block之前 → noised token attends to clean prefix
  - M_BC (Block-causal): [M_BC]_{ij}=1 iff j在i相同或更早的block → clean token间的AR-like因果依赖
  - 使用PyTorch flex-attention实现高效结构化masking
