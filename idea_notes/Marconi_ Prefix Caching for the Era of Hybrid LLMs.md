## Marconi: Prefix Caching for the Era of Hybrid LLMs

- baseline方法是什么？
  Baseline是extended SGLang/vLLM prefix caching方案，针对Hybrid LLMs（Attention+SSM混合架构）采用**fine-grained checkpointing**策略。由于SSM层使用in-place recurrent state更新（无法像Attention KV cache那样通过切片回滚到任意前缀位置），baseline采用naive方案：每隔固定x个token保存一次SSM layer的完整recurrent state作为checkpoint，使用标准LRU eviction管理缓存容量。如页面25-28所示，该方案存在两个致命缺陷：
  - Catch 1 — cache entries are sparsely-hit：大量checkpoint位于无人复用的token位置，缓存命中率极低
  - Catch 2 — cache entries are huge：SSM state的固定大小虽不随序列长度增长，但比单token的KV cache大几个数量级（d_state × d_model × 4 bytes per layer），大量低价值checkpoint占满缓存，导致频繁thrashing和低命中率

  全栈执行例子（以vLLM+ extended serving Mamba2-Hybrid-7B，LMSys conversational workload）：
  - 算法层：Fine-grained checkpointing——每隔k token保存完整SSM states（d_state=128, d_model=4096 → ~2MB per layer checkpoint, 24 SSM layers → ~48MB per sequence checkpoint）。LRU eviction管理所有cached states（SSM states + KV caches）。
  - 系统框架层：vLLM+/SGLang+ serving框架。扩展vLLM原有prefix caching（仅支持KV cache）增加SSM state checkpoint save/restore逻辑。但admission策略简单粗暴——每个请求的所有state都被缓存（admit-all），eviction仅基于recency（LRU）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。Trace-based离线模拟不涉及实际GPU kernel调度。
  - 硬件架构层：论文未明确说明（Cloudlab CPU节点运行离线模拟）。

  Baseline缺陷：(1) Admission缺乏判断力——每个请求的SSM state都被无差别缓存，对sparsely-hit的checkpoint没有识别能力；(2) Eviction缺乏计算感知——LRU仅看recency，不考虑复用该state能节省多少计算量（FLOPs vs 内存的tradeoff），导致高计算价值的长前缀state可能被低计算价值的短前缀state挤出；(3) 混合架构下KV cache与SSM state统一管理的缺失——KV cache大小随序列长度线性增长而SSM state固定，两者有不同的memory-compute tradeoff特征，LRU无法感知这种差异。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Marconi——首个面向Hybrid LLMs的prefix caching系统，通过两个协同设计解决baseline缺陷：

  **(1) Judicious Admission（基于前缀复用模式分类）**：不再admit-all，而是通过radix tree bookkeeping将前缀复用分为两种模式——Purely Input（系统提示词、few-shot示例，被多请求共享的中间节点）和Input+Output（对话历史，从leaf node继续）。对Purely Input识别为高复用概率并缓存；对Input+Output仅缓存最后token的SSM state（因为对话总是从末尾继续）。限制每序列至多2个SSM state checkpoint，从源头上消除sparsely-hit checkpoints。

  **(2) FLOP-Aware Eviction（计算感知淘汰策略）**：定义FLOP Efficiency = Total FLOPs saved across all layers (Attention + SSM + MLP) / Memory bytes consumed by cached states。Utility Score = recency + α × flop_efficiency。关键洞察：SSM state大小固定（不随prefix length变化），但长前缀节省更多FLOPs（覆盖更多token的计算），因此长前缀的FLOP efficiency更高。Marconi优先保留长前缀的高FLOP efficiency entries，而非LRU-only仅基于时间。α参数由config_tuner自动根据workload模式调优。

  **(3) Unified Radix Tree管理**：KV caches（Attention层）和SSM states（SSM层）统一在单个radix tree中管理，因为所有layer states必须代表同一prefix才能被复用。避免disaggregated管理导致的prefix一致性问题和eviction策略不协调。

  全栈执行对比baseline（以Marconi serving Mamba2-Hybrid-7B，同一LMSys workload）：
  - 算法层：Judicious admission替代naive checkpointing——radix tree speculative insertion后仅缓存高复用概率节点（purely-input branches + leaf final states），每序列≤2个checkpoint。FLOP-aware eviction替代LRU——eviction决策综合考虑recency和该state节省的FLOPs/byte。Mamba2-Hybrid 7B (4Attn+24SSM+28MLP)中SSM layer的FLOP efficiency远高于Attention layer（因为SSM state固定大小但覆盖prefix全部token的计算节省），Marconi的eviction policy因此对SSM-heavy模型表现更优。
  - 系统框架层：Marconi核心逻辑在radix_cache_hybrid.py中实现，通过radix_cache_vllm.py适配到vLLM serving framework。与baseline vLLM+的admit-all+LRU不同，Marconi在每次请求到达时执行speculative admission判断，仅selectively缓存高价值state。config_tuner持续监控workload命中率反馈，动态调整α权重。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。离线trace-based模拟环境。
  - 硬件架构层：论文未明确说明具体GPU硬件。Trace-driven evaluation on CPU。

  结果：Marconi vs fine-grained checkpointing (naive admission) token hit rate提升4.5×–34.4×（取决于workload和SSM比例），P95 TTFT降低36.1%–71.1%。FLOP-aware eviction alone vs SGLang+ LRU提升19%–219% token hit rate。Marconi表现随longer contexts、higher SSM ratios、larger SSM state dimensions更好——这些趋势与Hybrid LLM架构发展方向（更高SSM比例、更大模型如Jamba 1.5 398B）一致。

  设计思路核心：Marconi的本质是将prefix caching的admission和eviction从"recency-only, admit-all"的粗粒度策略转变为"reuse-pattern-aware, FLOP-aware"的精细策略。关键洞察在于Hybrid LLMs中Attention和SSM层具有本质不同的memory-compute tradeoff特征（KV cache O(L) memory vs SSM state O(1) memory），前缀复用模式可以被分类为结构化的两种类型，以及radix tree的节点自然对应复用概率梯度（中间节点 → 高复用，叶子节点 → 仅末尾复用）。
