## GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出GyRot量化框架，通过CoRFiG（Coarse Rotation, Fine Grouping）+ HAP（Harmonic-Aligned Permutation）+ 重公式化非对称量化+ceiling-based零點舍入，实现rotation与fine-grained group quantization的协同配合。核心算法设计：(1) CoRFiG：将rotation限制在coarse scope R=1024内局部执行，而非全局rotation，同时保持fine group size G=32，满足R=2^g·G关系。这样rotation在R范围内flatten分布、amortize outliers，同时group quantization保留local adaptation能力，解耦rotation scope与group granularity。(2) HAP：利用Hadamard矩阵的harmonic行结构（递归构造产生长度2^k的全+1或全-1向量），将全局选出的高magnitude outlier channel permute到harmonic rows上，使每个group内outlier乘以一致的符号（全+1或全-1），从而tighten per-group range并降低scale/zero-point精度需求。(3) 重公式化非对称量化：将传统先scale后bias的公式x̂=⌊x/s_x+z_x⌉改为先bias后scale的x̂=⌊(x+z_x)/s_x⌉，zero-point直接从unscaled domain计算z_x=−min(x_g)，避免小scale因子放大zero-point分布尾部。(4) Ceiling-based ZP rounding：用⌈·⌉替代传统round，保证z_Q≥z，消除zero-point量化误差导致的underflow clipping。最终实现INT8 scale factor和INT8 zero-point的fully integer dequantization，在W4A4KV4配置下保持competitive accuracy。实验比较：对比Tender (W8A8/W4A4)、MANT (W4A8/W4A4)、Quarot、SpinQuant、DuQuant、LightRot在WikiText-2 PPL、6个zero-shot task（PIQA/ARC-e/ARC-c/BoolQ/HellaSwag/WinoGrande）、MT-Bench conversational benchmark上的表现。GyRot-INT在W3A4极低bit配置下仍保持competitive PPL。

- 硬件平台是什么，配置是什么。
  算法accuracy评估在NVIDIA GPU上运行（论文未详述具体GPU型号，因accuracy evaluation不依赖特定硬件）。硬件实现评估：GyRot accelerator RTL以SystemVerilog实现，Samsung 28nm工艺，Synopsys Design Compiler综合，1GHz目标频率，片上SRAM由commercial memory compiler生成。对比baseline（Tender, MANT, LightRot）在iso-compute-area约束下综合评估。DRAM功耗使用Micron DRAM Power Calculator (DDR4 model)。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-1 (7B/13B)、LLaMA-2 (7B/13B)、LLaMA-3 (8B, 8B-Instruct)。数据集：WikiText-2（perplexity评估）、PIQA、ARC-easy、ARC-challenge、BoolQ、HellaSwag、WinoGrande（zero-shot task评估，通过LM-Evaluation-Harness框架）、MT-Bench（LLM-as-a-Judge对话质量评估，160 turns）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  论文未明确说明开源。GyRot算法pipeline：
  1. 离线阶段：
     a. 对每层weight矩阵应用CoRFiG：在R=1024 scope内执行Hadamard rotation（R=2^g·G, g=5, G=32）→局部flatten weight distribution
     b. HAP permute：识别全局high-magnitude outlier channels→permute到Hadamard矩阵的harmonic rows（长度2^k的全+1/全-1 vectors）→permutation可fuse进weight矩阵（permutation-invariant性质），无runtime overhead
     c. 对activation应用重公式化非对称量化：z_x=⌈−min(x_g)⌉（ceiling rounding避免underflow），s_x=(max(x_g)+z_x)/(2^b−1)，量化为INT4
     d. Weight对称量化（GPTQ）+ rotation后INT4量化，scale factor SW量化为INT8
  2. 在线推理（以LLaMA-3-8B W4A4KV4为例）：
     a. Activation经FVU做online Hadamard rotation（非线性层如embedding/SwiGLU后需要online rotation，因rotation-invariance不跨非线性层成立）
     b. INT4 activation × INT4 weight → 32-way dot product in PE → 13-bit partial sum
     c. Fully integer dequantization：先乘activation scale SX (INT8)→加zero-point项 ZX×WSUM（WSUM预计算）→乘weight scale SW (INT8)→32-bit integer accumulation
     d. 最终output转FP16写回buffer
  3. CoRFiG+HAP使INT8 SF/ZP即可保持accuracy，消除FP dequantization硬件开销
