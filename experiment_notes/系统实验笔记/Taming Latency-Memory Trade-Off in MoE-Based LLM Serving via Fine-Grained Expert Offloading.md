## Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

- 属于Serving调度的实现是什么？实验比较什么？
  论文提出FineMoE，一个细粒度(fine-grained) expert offloading系统，用于MoE-based LLM serving。核心Serving调度实现包含：(1) Expert Map Store：存储每个inference iteration、每个MoE layer上gate network对所有experts的概率分布（而非request-level的expert激活计数），用ndarray存储semantic embeddings和expert maps，容量默认1K maps（约200MB CPU memory）；(2) Expert Map Searcher：前d层使用semantic-based search（从embedding layer获取prompt semantic embedding，与历史embedding做cosine similarity取最相似历史iteration的前d层expert probability map），d层之后使用trajectory-based search（收集当前iteration已过层的gate probability trajectory，与历史expert maps做cosine similarity）；(3) Similarity-Aware Expert Selection：根据similarity score动态设置threshold delta=clip(1-score,0,1)，按概率从高到低选择最少expert使累计概率超过delta，且至少选择top-K所需数量；(4) Publisher-Subscriber异步路径：Context Collection→Expert Map Searcher→Prefetch Publisher→Cache Subscriber，将map search、prefetch和map update与inference forward解耦；(5) Expert Cache：C++/CUDA Runtime APIs实现prefetching/caching/offloading/on-demand loading，prefetch priority按p/(l-l_now)排序（概率越高、距离当前layer越近优先），eviction priority按1/(p×freq)（驱逐低概率、低访问频率expert）；(6) Expert Map去重：store满时用semantic和trajectory redundancy score去重。实验比较四个SOTA baseline：(a) MoE-Infinity（request-level Expert Activation Matrix + 同步prefetch）；(b) ProMoE（stride-based speculative prefetching + per-layer NN predictor，best-effort复现）；(c) Mixtral-Offloading（layer-wise speculative prefetching + LRU expert cache）；(d) DeepSpeed-Inference（expert-agnostic layer-wise parameter offloading，在MoE-Infinity codebase中实现并加入expert cache公平比较）。核心指标：TTFT、TPOT、expert hit rate、cache limit sensitivity、online end-to-end latency、overhead。

- 硬件平台是什么，配置是什么。
  主实验平台：6×NVIDIA GeForce RTX 3090（每张24GB GPU memory），GPU间pairwise NVLinks，CPU-GPU通过PCIe 4.0连接（带宽32GB/s），CPU为AMD Ryzen Threadripper PRO 3955WX（32 cores），480GB CPU memory。补充实验：NVIDIA A100（80GB HBM2e，2TB/s峰值带宽）。多GPU推理采用expert parallelism，expert通过hash map和round-robin分配到不同GPU。

- 开源Serving框架是什么。修改了什么。
  基于HuggingFace Transformers，复用/修改MoE-Infinity codebase构建FineMoE原型。Expert Map Store和Expert Map Searcher用Python/PyTorch/NumPy实现（ndarray存储，Tensor相似度计算）。Expert Cache在MoE-Infinity基础上用C++/CUDA Runtime APIs实现prefetching/caching/offloading/on-demand loading。新增GPU侧task pool与异步线程调度expert prefetching和on-demand loading。Baseline改造：MoE-Infinity评测前准备对应历史Expert Activation Matrix；ProMoE基于MoE-Infinity做best-effort prototype；DeepSpeed-Inference offloading逻辑在MoE-Infinity codebase中实现并加入expert cache公平比较。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  未找到FineMoE官方开源仓库。相关页面：https://2026.eurosys.org/papers.html（EuroSys 2026论文列表）；https://intellisys.haow.us/publications/（作者论文页仅提供PDF）。FineMoE复用/修改MoE-Infinity codebase，MoE-Infinity开源地址https://github.com/TorchMoE/MoE-Infinity。以Mixtral-8x7B + LMSYS-Chat-1M serving为例：
  1. 部署：启动6×RTX 3090 GPU集群，expert按hash map和round-robin分布到不同GPU。Expert Map Store预加载LMSYS-Chat-1M 70% prompts的semantic embedding和expert maps（1K maps, <200MB CPU memory）。Expert Cache初始化GPU cache limit（如6GB）。
  2. 请求到达：prompt "Summarize this paper" 到达。Prefill或第一个decode iteration开始前，Context Collection从embedding layer获取prompt semantic embedding。
  3. Expert Map Search（前d层）：Expert Map Searcher接收semantic embedding → 在Expert Map Store中计算cosine similarity → 找到最相似历史iteration → 获取其前d层expert probability map。
  4. Similarity-Aware Selection：similarity score s ∈ [0,1] → delta = clip(1-s, 0, 1)。若s低（相似度低），delta高，系统选择更多高概率expert预取以降低mis-prediction；若s高，delta低，只预取较少expert节省GPU cache。至少预取top-K所需数量。
  5. Trajectory-Based Search（d层之后）：当inference forward执行到d层后，收集已过层的gate probability distribution作为expert trajectory → 与历史expert maps做cosine similarity → 取对应目标层概率分布。
  6. Expert Prefetching：Expert Cache按p/(l-l_now)排序预取，把CPU memory中的expert weights异步搬到对应GPU。异步publisher-subscriber路径确保map search/prefetch与forward解耦。
  7. MoE Layer执行：gate network执行top-K选择。若所需expert已在GPU cache中→直接计算；若miss→FineMoE暂停普通prefetch task，立即从CPU到GPU on-demand loading缺失expert。
  8. Cache Eviction：cache超限时按1/(p×freq)驱逐低概率、低访问频率expert。
  9. Map Update：iteration结束后新产生的expert map和semantic/trajectory context写回Expert Map Store。Store满时用redundancy score去重保留覆盖性更好的maps。
  10. 效果：FineMoE相比DeepSpeed-Inference/Mixtral-Offloading/ProMoE/MoE-Infinity，平均TTFT分别降低74%/67%/56%/53%，平均TPOT分别降低46%/38%/27%/22%。Expert hit rate相比Mixtral-Offloading/ProMoE/MoE-Infinity分别提升14%/37%/68%。6GB cache limit下TPOT分别降低36%/25%/16%/29%。平均降低inference latency 47%，提升expert hit rate 39%。

