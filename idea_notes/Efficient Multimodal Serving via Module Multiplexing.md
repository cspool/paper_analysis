## Efficient Multimodal Serving via Module Multiplexing

- baseline方法是什么？
  Baseline是传统unimodal serving系统（Triton/ TF-Serving/ Clipper/ TurboTransformers）和Gpulet（spatio-temporal GPU sharing）。Triton将多模态模型当作monolithic computation graph，所有模块顺序执行，使用统一batch size和100% SM allocation。Gpulet将每个模块当作独立模型并发执行，但强制共享固定batch size且无模块间协同调度。

  全栈执行例子（以BLIP VQA + Triton + RTX 3090为例）：
  - 算法层：BLIP由三个模块组成：ViT-B/16 visual encoder (86M params)、BERT-based text encoder (110M params)、BERT-based text decoder。请求以统一batch size（如B=8）进入。
  - 系统框架/Serving层：Triton将BLIP作为单一模型图→一次接收B个请求→先对B个图像做preprocessing（resize/normalize/tensor conversion/H2D transfer，耗时可达visual encoder计算时间的40%）→visual encoder全部batch推理（GPU接近满利用率）→text encoder全部batch推理（GPU利用率骤降，因text encoder计算强度远低于visual encoder）→text decoder全部batch推理（GPU继续低利用率）。同一batch内所有请求必须等待最慢模块完成。如VQA中单图对应多个问题，每个问题独立通过全流程，visual encoder被重复计算。
  - 编译框架层：论文未明确说明（使用PyTorch默认编译路径）。
  - kernel调度层：论文未明确说明（使用vLLM默认CUDA kernel）。
  - 硬件架构层：NVIDIA RTX 3090/A100，无定制硬件。GPU active SM在超过50%时间低于10%（论文Figure 4）。

  Baseline缺陷总结：
  1) **Inference heterogeneity**：visual encoder latency可达text encoder的8×，text encoder在B≈8后饱和而text decoder到B>32仍受益，统一batch无法同时匹配各模块最优工作点。
  2) **Preprocessing heterogeneity**：图像预处理（loading/resize/normalize/H2D）可达visual encoder计算时间的40%，串行执行让GPU在预处理期间空闲。
  3) **Input imbalance / reuse缺失**：VQA2中单图至少对应3个问题（最多246个），每个请求重算visual encoder浪费compute和latency。
  4) **Gpulet虽支持spatio-temporal sharing**，但将模块视为独立竞争模型，无synergistic scheduling、无stage-level pipeline、无modal cache reuse。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出EEVEE，核心是module multiplexing：将多模态模型的modality-specific modules作为独立调度单元，配置独立batch size和SM allocation，在同一GPU上通过NVIDIA MPS并发执行。系统由offline scheduler（greedy search生成multiplexing strategies）和online controller（modal cache管理）两部分组成。

  **缺陷1→方法**：inference heterogeneity导致统一batch下GPU利用率波动剧烈
  → EEVEE方案：Module-level scheduling + balanced SM allocation。每个模块运行独立vLLM process，拥有各自最优batch size（text decoder可用大batch、visual encoder用小batch）；SM allocation按模块计算强度动态划分（从短latency模块转移SM到长latency模块直到stage内各模块latency趋近平衡）。这样让visual encoder获得充足SM的同时text decoder以更大batch充分占用剩余SM，填充原baseline中的GPU bubble。

  **缺陷2→方法**：preprocessing heterogeneity导致GPU等CPU
  → EEVEE方案：Stage-level parallelism。将预处理和推理视为独立阶段，visual encoder的新请求预处理可与text decoder的推理在同一GPU上并发执行。GPU不再因等待CPU完成预处理而闲置。

  **缺陷3→方法**：input imbalance导致visual encoder被重复计算
  → EEVEE方案：Modal cache control + request-aware reuse。对encoder-decoder模型（BLIP）缓存cross-attention消费的视觉token KV pairs，对decoder-only MLLM（LLaVA）缓存decoder消费的视觉token KV pairs。后续引用同一图像的问题通过64-bit hash查找缓存，跳过visual encoder重算。Cache支持compression（按attention score剪除低重要性token），在GPU memory紧张时优先使用compressed critical cache，full cache保留在host memory。

  **缺陷4→方法**：Gpulet类spatio-temporal sharing将模块视为独立竞争模型
  → EEVEE方案：Synergistic scheduling algorithm。将模块间的directed dependency、egress module batch multiplier、SLO latency约束和SM总量约束（per-stage SM总和=100%）建模为约束优化问题。Greedy search从monolithic顺序执行的可行策略初始化，逐步增加各模块batch size，每次将增量放在对batch latency负面影响最小的stage，直到违反SLO。这避免了随意并发导致的资源竞争退化（论文strawman实验：不合适策略使throughput从22.2降到14.1 req/s，合适策略可达34.4 req/s）。

  论文方法全栈执行例子（以BLIP VQA + EEVEE + RTX 3090为例）：
  - 算法层：BLIP模块不变（ViT-B/16 + BERT encoder + BERT decoder），但执行方式从整模型顺序变为模块级并发。Visual encoder batch size按复用需求设为较小值（如B_v=2），text decoder batch size可扩大（如B_t>2）。Modal cache compression按30% ratio剪除低attention视觉token，score仅轻微下降。
  - 系统框架/Serving层：三个独立vLLM process（visual encoder、text encoder、text decoder）通过NVIDIA MPS并发执行→offline scheduler根据SLO、模型结构和GPU硬件生成strategy（如visual encoder 70% SM，text decoder 30% SM）→controller管理GPU shared memory中的modal cache（64-bit hash索引、per-module hashmap、global LRU eviction）。用户请求包含图像+Q1：visual encoder处理图像→controller缓存visual tokens→text decoder生成A1；同一图像Q2到达→controller从cache加载visual tokens→跳过visual encoder→cache loading与Q1的text decode或新图像的encode overlap→生成A2。Stage-level parallelism使Q2的preprocess与Q1的inference重叠。
  - 编译框架层：论文未明确说明（使用PyTorch默认编译路径，vLLM后端）。
  - kernel调度层：论文未明确说明（未提出新GPU kernel，SM allocation通过CUDA MPS active thread percentage环境变量在CUDA初始化前设置）。
  - 硬件架构层：NVIDIA RTX 3090/A100，PCIe 3.0 32GB/s，使用NVIDIA MPS（非MIG）实现GPU spatial sharing。GPU active SM接近90%。
