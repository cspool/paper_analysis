PD-Multiplex、Bullet、Shift可以试试。

Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates：第一，提出 Hardwired LPU / HNLPU 这一完全硬连线 gpt-oss 120B FP4 模型的处理器范式，把权重参数物理嵌入计算结构，消除权重 fetch。

ZipServe：算法pipeline优化

现有 lossless model compression 能无损减少 LLM 权重存储，但很少真正加速 inference。原因不是压缩本身无用，而是传统熵编码和 GPU 执行模型不匹配：Huffman、ANS 这类 variable-length bitstream 需要数据相关的串行解析、查表和指针推进，会破坏 GPU warp 的 SIMT 并行。

DFVG：故事思路可抄（pipeline部分放到FPGA，单GPU不适合）

DFVG 提出一种 Draft-on-FPGA、Verify-on-GPU 的异构 speculative decoding 架构。

PAT：共享前缀，Decode Attn的运行时kernel调度。

BAT：GR Attn。

TetriServe：DiT Serving的更细调度粒度（round），因为运行时间可profile？

Laser：LLM iter、layer执行，学习。

MoDM：大小模型协作的DiT生图。

JanusQuant：2 bit推理的运行时开销。

3-bit LLM：3 bit的访问不对齐。

Sparse Transformer：MHA-mask-后续算子融合的编译框架，处理mask激活的后续算子低效。

MetaAttn：各种Attn变体的baseline。

ChituDiff：Diffusion编译。主要瓶颈来自三个层面。第一是计算与显存访问冗余：多个请求共享 prompt、LoRA、ControlNet 或中间结果时，baseline 仍可能重复执行 CLIP / U-Net / attention 等计算，或重复读取相同 K/V、权重和 conditioning。第二是调度和 batching 低效：按完全相同 shape 才 batching 会让许多请求无法合批；直接使用 ragged batch 又会为均匀 shape 请求引入额外索引和变换开销。第三是编译期组合爆炸：一个 diffusion pipeline 可有十几个甚至二十多个输入，如果为所有输入属性组合生成整图特化版本，dEngine 数量会指数增长。

MixFusion：不同分辨率切分相同patch并行。

ASM-SpMM：SpMM on Arm SME。

SpMV on TC：SpMV对齐TC。

VDHA：SpMSpV的GPU kernel

RoMeo：Text模型量化中的离群值处理，换到Video/Image呢？

AUM：CPU（+AU）的多任务调度。

GyRot：低比特量化的算法/硬件设计，多种方式需要同时支持！。

SLINFER：CPU+GPU，ServerlessLLM，调度设计，学习。

PIMPhony：PIM系统设计（学习）。

Adaptive Draft Sequence Length：SpD的PIM硬件系统设计。

bitdecoding：CC+TC作低比特量化计算和运行时。

AQPIM：激活在线量化。Product Quantization(PQ) 在线压缩 KV cache，把 KV 表示成 codebook 与 index，并在 PIM 内直接用 codebook/index 计算近似 attention，避免恢复完整 KV。

focus、VRex：Video LM

AgentBench：Agent Serving框架。（需要至少1张A100）

PASCAL：reasoning decoding？reasonging mode和一般LLM区别？

RPU - A Reasoning Processing Unit：新dataflow架构需要设计什么？

RoMe：内存控制器重构，Row粒度访问。

LeGO：单GPU同时部署SLM和图形渲染任务。

Inter-Chiplet Communication：chiplet通信。

LRM-GPU：多chiplet GPU？传统GPU？

Swift：学习SpMM的GPU kernel。

Uni-STC：四种Sp代数的统一硬件单元。

QuCo：ATT运行时配置的硬件模块。

μShare：GPU kernel运行时，提高SM不同资源使用率。

flashfuser：学习，利用DSMEM完成kernel fusion，而不需要GMEM。

VAR-Turbo：新pipeline和加速器。

AdaServe：多应用SLO要求的调度框架。

PiLLM：面向LLM Serving的token level资源管理机制，基于预测更激进分配内存同时减少OOM。

flexpipe：inflight pipeline refactoring 根据实时 CV、吞吐、延迟和队列状态选择候选 granularity，在 burst 时拆细 stage，在稳定时合并 stage。

tokenflow：交互式Streaming Serving的抢占式优化。

Fine-Grained Expert Offloading：MoE Serving，基于运行时信息预测的Expert调度，拆分Expert。

Adaptive KV Caching：Cache策略，空闲SM重算，故事思路。

MFS：model family改为嵌套model tier，新pipeline。

Module Multiplexing：多模态Serving，不同模块（VAE Encoder/Decoder、BackBone）的token走独立pipeline执行。

Test-Time Compute：TTC pipeline。

TailorLLM：大小模型的云-边协同。

TZ-LLM：权重加密infer。

AIMS：agent级联调用的云-端协同。

Computer-Use Agents：Agent友好的OS接口，学习。

FlashPS：图像编辑pipeline的Serving优化。

Cooperative Compilation and Scheduling：编译-调度协同优化，动态Tiling。

LATENT WAVELET DIFFUSION FOR ULTRA-HIGH-RESOLUTION IMAGE SYNTHESIS：新模型。

Spectral Regularization for Diffusion Models：新模型。
