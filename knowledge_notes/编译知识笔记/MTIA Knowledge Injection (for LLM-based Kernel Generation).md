## MTIA Knowledge Injection (for LLM-based Kernel Generation)

术语是什么？
MTIA Knowledge Injection解决LLM缺乏proprietary accelerator知识的根本挑战：Unlike widely-documented GPU architectures (NVIDIA CUDA, AMD ROCm)，MTIA的proprietary architecture和programming model mainly absent from public training corpora——pretrained LLMs不知道MTIA-specific hardware features、扩展的Triton language constructs和optimization patterns。Knowledge injection通过系统性地将MTIA domain expertise注入persistent knowledge base，使Deep Search Sub-Agent在runtime检索并inject MTIA-specific documentation到LLM context window，effectively teaching the model MTIA-specific programming idioms absent from pretraining data。

从编译框架角度拆解术语：
MTIA Triton Extensions的三类文档化内容：

**Hardware Feature Exposure**:
- SFU LUT operations: `tl.extra.libdevice.gelu(x)` → SFU LUT query (not mathematical approximation)
  文档化: exp, gelu, log, sigmoid, tanh → each mapping to dedicated SFU instructions
- Compilation options:
  `cb_multiplier` (int): 扩大Circular Buffer分配实现multi-operation concurrent execution
  `use_dual_core` (bool): distributor DMA→core A, vector→core B实现heterogeneous execution
  通过@triton.autotune静态探索: BLOCK_SIZE∈{32..1024}, cb_multiplier∈{1..8}

**Compute Helper Functions**:
- unary_elemwise_compute(op, x): 30+ ops → optimized vector instructions
- binary_elemwise_compute(op, x, y): tensor-tensor arithmetic/comparison/ML-specific
- binary_elemwise_const_compute(op, x, const): tensor-scalar operations

**Advanced Synchronization & Communication Primitives**:
- Cross-PE Broadcasting: tl.load(direction="down"/"right") + tl.consume()
- Cross-PE Reduction: tl.store(direction=...) sends results to neighboring PE
- Runtime Barriers: tl.pe_runtime_barrier() → libjit_fba_runtime_barrier()
- Explicit Tensor Copies: tl.copy() for producer-consumer sync between dual cores

术语一般如何实现？如何使用？
Without MTIA knowledge injection: LLM生成standard GPU-targeted Triton code → compilation failures或functionally incorrect on MTIA (wrong SFU usage, missing inter-PE sync, incorrect type system)。With injection: runtime context教育LLM → generation能leveraging hardware-specific features——SFU for activation functions, inter-PE communication for multi-PE kernels, dual-core synchronization for pipeline parallelism——生成接近hand-optimized performance的production-grade MTIA kernels。This approach generalizes to future accelerators: 只需inject对应文档到知识库即可，无需model retraining或fine-tuning。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
