## Fast-dLLM v2: Efficient Block-Diffusion LLM

- baseline方法是什么？
  **标准自回归（AR）LLM推理**：以Qwen2.5-Instruct为backbone，使用causal attention mask + next-token prediction loss训练，推理时逐token自回归生成。每个forward step仅生成1个token，需要response_length次forward才能完成生成（如256 tokens需要256次forward）。自回归模型的sequential decoding限制了推理并行度和吞吐量。

  全栈执行例子（Qwen2.5-7B-Instruct，GSM8K 5-shot推理，A100 GPU，gen_len=256）：
  - **模型推理算法层**：Qwen2.5-7B-Instruct使用causal self-attention，每个token只能attend到自身及之前的token。逐token自回归生成：p(x_i | x_{<i})。训练loss为cross-entropy over next-token prediction。推理时从prompt开始，逐个生成token直到[EOS]或max_length。256 tokens需要256次sequential forward passes。
  - **系统框架层**：使用标准PyTorch推理，可搭配vLLM等serving框架使用continuous batching + PagedAttention管理KV cache。每次forward迭代生成1个token → 更新KV cache → 继续下一token。prefill阶段一次处理prompt的KV cache计算。论文未明确说明特定serving框架修改。
  - **编译框架层**：论文未明确说明。使用标准PyTorch/HuggingFace Transformers推理路径。
  - **kernel调度层**：Decode阶段为memory-bound的GEMV操作（batch_size × 1 token），每次forward处理小矩阵向量乘，GPU计算利用率低。KV cache存储在HBM中，每步加载完整cache参与attention计算。
  - **硬件架构层**：NVIDIA A100/H100 GPU，无自定义硬件修改。AR decode吞吐量约39.1 tok/s（GSM8K, A100）。

  Baseline缺陷：
  - (a) **Sequential decoding限制并行度**：逐token生成，256 tokens需256次forward，GPU在decode阶段利用率低（memory-bound GEMV）。
  - (b) **无法利用块内token间的双向依赖**：causal attention仅允许单向（left-to-right）attention，块内token无法互相condition以提升预测质量。
  - (c) **吞吐量扩展受限**：增加batch size虽能提升吞吐，但AR模型的每token延迟不变，总延迟与生成长度线性相关。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Fast-dLLM v2：将预训练AR模型适配为block diffusion LLM**。核心设计：(i) Block diffusion训练——序列拆分为block size=32的块，块内使用bidirectional attention + masked token prediction（diffusion），块间保持AR causal conditioning；(ii) Complementary masking + token shift——每个训练样本产生两个互补mask view，masked位置使用i-1位置的hidden state预测token i，保留AR模型的representation quality；(iii) Hierarchical caching推理——block级KV cache（跨block复用历史上下文）+ sub-block DualCache（块内高效并行refinement）+ confidence-aware parallel decoding。

  全栈执行例子（Fast-dLLM v2 7B，GSM8K 5-shot推理，A100 GPU，gen_len=256，threshold=0.9）：
  - **模型推理算法层**：序列被组织为K=⌈256/32⌉=8个block，每个block内通过masked diffusion并行refine（bidirectional attention），block间通过causal conditioning自回归生成。子块大小=8，每次forward在sub-block内基于confidence阈值0.9并行解码多个高置信token。训练时complementary masking保证所有token都接收masked和unmasked上下文的监督。Token shift通过position i-1的hidden state预测position i的token，使dLLM保持AR-like temporal representations。总解码步数远小于256（约40-80步），throughput从39.1→101.7 tok/s（2.6×加速）。
  - **系统框架层**：推理pipeline实现block-level KV cache管理：已解码block的K/V被缓存为read-only prefix，当前block仅需计算自身attention + 对prefix的cross-attention。无需serving框架修改，在PyTorch层实现block-wise decoding loop + cache管理。batch decoding通过右填充[MASK]对齐block边界，所有序列同步逐block推进。
  - **编译框架层**：论文未明确说明。使用PyTorch flex-attention实现自定义block-wise attention mask的高效计算。
  - **kernel调度层**：Block-level cache将attention计算量从O(T·(|p|+L)²·d)降至O(K·T'·(B²+|p|·B)·d)，其中B=block size，K=block数，T'为每块内步数（远小于原始T=256）。DualCache进一步缓存sub-block的prefix/suffix KV，将块内attention降至O(S²·d)（S=sub-block size=8）。这使得GPU compute利用率大幅提升，尤其在batch size较大时（compute-bound regime）。
  - **硬件架构层**：NVIDIA A100/H100 GPU，无自定义硬件修改。通过降低每token所需forward次数和复用KV cache减少总计算量，在A100上达2.5×加速（batch=1），H100上batch=64时达1.8×加速。

  关键设计选择与baseline缺陷的对应：
  - **defect: Sequential decoding限制并行度 (a)** → 方案：Block diffusion——序列分为block，块内bidirectional attention允许并行生成多token。配合confidence-aware parallel decoding（来自Fast-dLLM v1），threshold=0.9时仅轻微准确率下降即达2.6×加速（GSM8K）。块间仍保持AR因果依赖，保证全局语义连贯性。
  - **defect: 无法利用块内双向依赖 (b)** → 方案：Block-wise attention mask设计——M_BD（块内双向自注意力）使block内token可互相condition，M_OBC（offset block-causal）保持对历史clean context的单向访问，M_BC（block-causal）保持clean token间的AR-like progression。这种hybrid attention使模型在块内获得更丰富的context modeling能力，同时保留AR模型的预测质量。
  - **defect: 吞吐量扩展受限 (c)** → 方案：Block diffusion的并行特性使throughput随batch size增长优于AR模型（Figure 5）。H100上batch=64时diffusion比AR快1.8×，因为diffusion的forward pass计算更密集（多token并行），能更好利用H100的更高算力。Sub-block cache在compute-bound regime（batch=32）下提供额外加速。
  - **额外设计：数据效率** → 通过复用预训练AR模型的权重和AR-friendly的block-wise attention设计（接近原始causal attention结构），Fast-dLLM v2仅需~1B tokens微调（vs Dream的~580B tokens），实现500×数据减少。训练仅需64×A100约8-12小时，使block diffusion适配变得实际可行。
  - **额外设计：训练-推理一致性** → 通过引入sub-block解码策略（推理时使用sub-block size=8，训练时block size=32），在保持训练block结构与推理一致的前提下（避免Table 4中的mismatch性能退化），灵活控制推理粒度以优化accuracy-throughput trade-off。
