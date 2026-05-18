## Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

- baseline方法是什么？
  Baseline是四种SOTA MoE serving系统（均经论文作者改造以公平比较）：(1) MoE-Infinity：使用request-level Expert Activation Matrix记录每个request历史上激活的expert集合，同步进行expert prediction和prefetching，prefetch与inference forward不重叠；(2) DeepSpeed-Inference：expert-agnostic layer-wise parameter offloading，按层on-demand从CPU加载参数到GPU，无expert级别的prefetch/cache；(3) Mixtral-Offloading：layer-wise speculative expert prefetching，使用LRU cache管理GPU侧expert，预测粒度为layer级而非iteration级；(4) ProMoE：stride-based speculative prefetching + per-layer NN predictor，需要针对每个MoE模型训练predictor（论文因ProMoE未开源只做best-effort复现）。共同根因：baseline依赖request-level热度统计或固定规则（stride/LRU），忽略不同iteration、不同layer、不同prompt语义之间的细粒度差异，导致expert hit rate低或GPU cache被低价值expert占据。

  全栈执行例子（以MoE-Infinity + Mixtral-8x7B + RTX 3090 24GB为例）：
  - 算法层：Mixtral-8x7B MoE decoder-only模型，每层8 experts×top-2 routing。Gate network每个token选择2个expert。MoE-Infinity在request history中统计每个request过去激活的expert，构建Expert Activation Matrix。新请求到达时，基于历史矩阵做expert prediction→prefetch predicted experts→forward。
  - 系统框架/Serving层：MoE-Infinity codebase。Request到达→查询Expert Activation Matrix→predict requested experts→同步prefetch（blocking forward）→MoE layer gate network top-2选择→若命中GPU cache则直接计算，若miss则on-demand CPU→GPU loading。问题：request-level聚合冲淡iteration-level清晰模式（entropy analysis显示coarse-grained patterns比fine-grained更不可预测），同步prefetch增加延迟。
  - 编译框架层：论文未明确说明（PyTorch + CUDA默认编译路径）。
  - kernel调度层：MoE-Infinity在GPU侧管理expert cache，使用CUDA Runtime APIs做expert weight的CPU↔GPU传输。Expert loading通过PCIe 4.0（32GB/s），单次expert loading耗时与expert size成正比。
  - 硬件架构层：6×RTX 3090（24GB each, pairwise NVLinks, PCIe 4.0 32GB/s）+ AMD Threadripper 3955WX（32 cores, 480GB CPU memory）。Expert parallelism将不同expert分布到不同GPU。inactive expert占据GPU显存（72%-84% of parameters为inactive），造成数十GB显存浪费。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出FineMoE，核心是用iteration-level expert probability distribution（Expert Map）替代request-level expert activation count，结合semantic和trajectory similarity做细粒度expert prefetch/cache决策。

  **缺陷1：Request-level Expert Activation Matrix冲淡iteration-level pattern → expert prediction hit rate低**
  → FineMoE方案：Expert Map Store记录每个inference iteration、每个MoE layer上gate network对所有experts的概率分布。Entropy analysis显示fine-grained patterns比coarse-grained patterns更可预测（coarse-grained统计将早期decode iteration的清晰模式与后续iteration混成高熵统计）。概率分布不仅表示expert是否被选中，还表达gate network对expert的置信度→低相似度时多预取降低miss，高相似度时少预取节省显存。

  **缺陷2：固定规则（stride/LRU）忽略prompt语义和iteration间expert使用变化 → 预测与当前context脱节**
  → FineMoE方案：Dual-Mode Expert Map Search。前d层：semantic-based search——从embedding layer获取prompt semantic embedding，与历史embedding做cosine similarity取最相似历史iteration的前d层expert probability map。d层之后：trajectory-based search——收集当前iteration已过层的gate probability trajectory，与历史expert maps做cosine similarity。两种模式分别在prompt语义相似度和execution trajectory相似度上匹配，使预测适应当前上下文。

  **缺陷3：同步prefetch阻塞inference forward → 增加TTFT/TPOT**
  → FineMoE方案：Publisher-Subscriber异步路径。Context Collection→Expert Map Searcher→Prefetch Publisher→Cache Subscriber与inference forward完全解耦。forward执行时search和prefetch在后台异步进行，不阻塞compute。仅在cache miss时暂停普通prefetch task执行on-demand loading。

  **缺陷4：LRU eviction与MoE layer-wise顺序执行特征不匹配 → 驱逐即将使用的expert**
  → FineMoE方案：Probability-Aware Prefetch/Eviction Priority。Prefetch priority = p/(l-l_now)：概率越高、距离当前layer越近，越优先预取。Eviction priority = 1/(p×freq)：低概率、低访问频率expert优先驱逐。直接量化"未来哪些expert更可能且更快被用到"，而非LRU的时间局部性假设。

  **缺陷5：粗粒度预测无法在hit rate和GPU memory之间fine-tune → cache capacity利用不高效**
  → FineMoE方案：Similarity-Aware Dynamic Selection。delta = clip(1-score, 0, 1)，按概率从高到低选择最少expert使累计概率超过delta。低相似度时delta高→多预取→降低miss risk；高相似度时delta低→少预取→节省GPU cache。动态均衡hit rate和memory footprint。

  FineMoE全栈执行例子（以Mixtral-8x7B + LMSYS-Chat-1M + 6×RTX 3090为例）：
  - 算法层：Mixtral-8x7B MoE decoder模型不变（每层8 experts×top-2 routing）。控制面新增Expert Map Store（iteration-level expert probability map）和Similarity-Aware Selection（delta=clip(1-score,0,1)）。FineMoE不改变gate network或模型权重，仅在expert placement层面优化。
  - 系统框架/Serving层：基于HuggingFace Transformers + MoE-Infinity codebase改造。六组件协同：(1) Context Collection：从embedding layer提取semantic embedding（前d层），收集已过层gate probability作为trajectory；(2) Expert Map Searcher：订阅context，semantic search（前d层）+ trajectory search（d层之后），PyTorch native ops做pairwise cosine similarity；(3) Similarity-Aware Expert Selector：按similarity score算delta，选累计概率超delta的最少expert；(4) Expert Cache（C++/CUDA Runtime）：按p/(l-l_now)异步prefetch，按1/(p×freq) evict；(5) On-Demand Loading：cache miss时暂停普通prefetch，立即CPU→GPU加载缺失expert；(6) Map Updater：iteration结束写回新map，redundancy score去重。多GPU expert parallelism用hash map+round-robin分配。整体在6GB GPU cache下vs DeepSpeed-Inference/Mixtral-Offloading/ProMoE/MoE-Infinity的TPOT分别降低36%/25%/16%/29%。
  - 编译框架层：论文未明确说明（PyTorch + CUDA默认编译路径）。
  - kernel调度层：Expert Cache用CUDA Runtime APIs实现expert weight的CPU↔GPU传输和GPU cache管理。Prefetch通过异步task pool+scheduler执行，与inference forward CUDA stream重叠。On-demand loading使用高优先级task打断普通prefetch。Expert weight传输通过PCIe 4.0（32GB/s）。
  - 硬件架构层：6×RTX 3090（24GB, NVLinks, PCIe 4.0 32GB/s）+ AMD Threadripper 3955WX（32 cores, 480GB CPU memory）或A100 80GB。GPU显存作为高速expert cache，CPU memory作为大容量expert weight存储。Expert Map Store在CPU memory（1K maps <200MB CPU memory，32K maps仍低于200MB）。PCIe 4.0带宽32GB/s决定expert loading latency下限。
