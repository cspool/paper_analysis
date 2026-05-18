## MFS: An Efficient Model Family Serving System for LLMs

- baseline方法是什么？
  Baseline是Orca serving system + 独立部署model family中多个模型（如分别部署Llama2-7B和Llama2-13B）。Orca的selective batching假设batch内所有请求共享同一模型结构，因此无法高效跨不同规模模型batching；独立部署小模型和大模型时，模型权重和KV cache都需分别保存，显存占用随模型数增加而上升；speculative sampling中draft model（7B）和target model（13B）参数/层结构不一致，小模型生成的token后大模型无法直接复用其KV cache，切换模型需重复prefill/decode计算。

  全栈执行例子（以Orca分别部署Llama2-7B + Llama2-13B + 独立speculative sampling为例）：
  - 算法层：两个独立LLM（Llama2-7B-chat、Llama2-13B-chat），各自独立推理，无参数/KV cache共享。Speculative sampling时小draft model生成候选token→大target model需从头prefill这些token→无法复用draft的KV cache。
  - 系统框架/Serving层：Orca selective batching按模型类型分组→7B请求batch和13B请求batch完全独立→GPU compute和显存无法跨模型共享。两份模型权重独立加载→显存占用=W(7B)+W(13B)。两份KV cache独立管理→跨模型不共享。每个模型各自维护waiting queue→各自调度各自batch。
  - 编译框架层：论文未明确说明（PyTorch默认编译路径）。
  - kernel调度层：论文未明确说明（使用Orca默认CUDA kernel，未提出新kernel）。
  - 硬件架构层：NVIDIA A100 GPU（2卡）/ NVIDIA 3090 GPU（8卡），无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出MFS，用Knowledge Precipitation将model family最大模型微调为嵌套multi-tier模型，配合group batching、attention fusion、shareable KV cache和tier-aware speculative sampling。

  **缺陷1：Orca selective batching假设batch内请求共享同一模型结构→无法跨不同规模模型高效batching→GPU compute underutilization**
  → MFS方案：Group batching。multi-tier模型使不同tier共享公共前缀层→tier-level scheduler将需要相同前缀层的不同tier请求组成group batch→公共层一起执行→GPU compute利用率提升。decode阶段通过attention fusion拼接QKV进一步提升GPU并行度（以少量冗余attention计算换并行度）。

  **缺陷2：独立部署多模型→模型权重和KV cache双重占用GPU显存→显存压力限制并发数**
  → MFS方案：Shareable KV cache + 嵌套模型权重共享。所有tier共享低层Transformer参数（单份权重替代多份独立模型），KV cache manager将公共前缀层的KV cache标记为multi-tier shareable→高tier可直接复用低tier KV cache→GPU memory footprint降低47.8%。

  **缺陷3：独立draft/target speculative sampling→小模型KV cache对大模型不兼容→target验证需重新计算前若干层prefill**
  → MFS方案：Tier-aware speculative sampling。draft和target是同一嵌套模型的不同tier→低tier产生的hidden states和KV cache对高tier完全兼容→高tier验证时直接继承低tier KV cache，无需重新prefill→GPU utilization从23.9%提升到59.8%。

  **缺陷4：直接构造multi-tier模型的baseline方法（strawman early-exit/PEFT/deep pruning）无法同时保证质量、层连续性和KV cache兼容性**
  → MFS方案：Knowledge Precipitation + layer-only tier split + step-by-step fine-tuning。(a) 只切layer不切head→保持attention中所有head一致性，确保跨tier KV cache可共享；(b) step-by-step逐层沉淀知识而非一次性joint training→避免多tier loss梯度冲突导致低tier性能不可控；(c) latency-aligned切分→使tier延迟匹配目标小模型用户体验。

  论文方法全栈执行例子（以Llama2-13B → 3-tier MFS + 2×A100 serving为例）：
  - 算法层：Knowledge Precipitation将Llama2-13B fine-tune为嵌套3-tier模型（tier-1=前18层/~3B等效, tier-2=前32层/~10B等效, tier-3=全部40层/13B）。每个tier有独立lm_head和training loss，低tier通过高tier梯度反向传播获得知识。Step-by-step fine-tuning：先train tier-3→加tier-2 head co-train→加tier-1 head co-train，逐层沉淀。
  - 系统框架/Serving层：front-end接收请求和所需tier→request pool→tier-level scheduler维护三个tier队列→group batching将需要相同前缀层的请求合并执行→attention fusion在decode阶段拼接QKV提升GPU并行度→shareable KV cache manager记录所有tier公共前缀层KV为可共享→tier切换时直接复用。可选speculative sampling：tier-1 draft若干token→tier-3 verify，复用tier-1 KV cache。
  - 编译框架层：论文未明确说明（使用PyTorch默认路径）。
  - kernel调度层：论文未明确说明（未提出新GPU kernel，attention fusion使用标准attention实现但拼接QKV输入）。
  - 硬件架构层：训练16×H800 SXM5 80GB + 400Gbps InfiniBand、推理2×A100/8×3090，无定制硬件。
