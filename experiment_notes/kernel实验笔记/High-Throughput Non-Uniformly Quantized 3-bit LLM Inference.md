## High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Quantix fused dequantization-matmul CUDA kernel，将non-uniform 3-bit weight的dequantization与Tensor Core matrix multiplication融合为单一GPU kernel。核心kernel设计：(1) 输入为hardware-aligned bit-shuffled weights（1-bit packed W1'和2-bit packed W2'）、activations A和centroids C，输出Y=A×Dequant(W1',W2',C)；(2) inter-tile层：用cp.async (128-bit width)异步预取future K-tile的W1'/W2'/A到shared memory，与当前tile计算重叠；(3) intra-tile层：从shared memory load subtile到registers→CUDA cores执行in-register dequantization（1-bit+2-bit bit concatenation重建3-bit index→shift+mask按qi=(R>>3i)&0x7提取→centroid lookup用3-bit index查row-specific centroids得FP16 W†）→Tensor Cores执行MMA (A×W†)；(4) in-register dequantization避免了中间结果写回global memory，消除cache-unfriendly pointer chasing和额外指令开销；(5) 两层double buffering：Smem0/Smem1实现inter-tile overlap，Reg0/Reg1实现intra-tile overlap；(6) Split-K将K维切分为多个独立slices并行计算partial sums，最后lightweight reduction kernel合并；(7) 128-bit vectorized memory access（UINT4 reinterpret）使global→shared和shared→register均以单指令传输128-bit chunk。实验比较kernel-level speedup vs FP16 cuBLAS、SqueezeLLM、Any-Precision LLM、GPTQ，Ablation study量化in-register dequantization/pipelining/vectorization/Split-K贡献，GPU utilization profiling (NVIDIA Nsight)分析compute/memory/ALU/Tensor Core/cache utilization。

- 后端平台是什么，配置是什么。
  NVIDIA L40 GPU（主要kernel benchmark平台，面向LLM inference，Compute Capability 8.9）、NVIDIA A100 GPU（对比平台，更高memory bandwidth降低memory-efficient kernel的相对优势）。NVIDIA Nsight用于profiling。

- 评估性能的软件/脚本是什么。修改了什么。
  NVIDIA Nsight分析GPU utilization：compute/memory utilization、ALU/Tensor Core utilization、cache hit rate和throughput。kernel benchmark从LLaMA/OPT linear layers提取真实weight matrix shapes并测试batch size 1-512。Quantix fused kernel通过CUDA实现，输入为预处理的1-bit和2-bit packed weights（经由offline bit shuffling），kernel内部通过PTX-level instructions调用Tensor Core MMA（mma.m16n8k16等）。修改：论文fused kernel集成进HuggingFace Transformers替换SqueezeLLM默认backend；对uniform baselines (GPTQ/Marlin)使用AutoGPTQ library。kernel benchmark 100 warm-up + timed iterations。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/yuang-chen/Quantix-PPoPP26。Quantix fused kernel使用流程：
  1. 离线bit shuffling：对已有non-uniform quantized weights执行bit dividing+bit mapping，生成W1'/W2'和reordered centroids C
  2. Kernel编译：nvcc编译fused dequantization-matmul CUDA kernel为.so
  3. 在线调用：输入W1' (32×1-bit elements/32-bit word)、W2' (32×2-bit elements/64-bit word)、activations A (FP16)和centroids C (FP16 per row)
  4. Kernel执行流程：`cp.async` prefetch→shared memory staging→register load→in-register dequantization→Tensor Core MMA→output Y
  5. L40上3-bit Quantix平均speedup：4.82× over FP16 cuBLAS、3.93× over Any-Precision、46.07× over SqueezeLLM、10.25× over GPTQ
  6. Ablation：移除in-register dequantization性能降至~40%（最显著），禁用pipelining降至~41%，移除vectorization降至~86%，Split-K主要帮助小矩阵增加parallelism
  7. 2-bit/4-bit变体：2-bit Quantix平均5.45× over 16-bit baseline（up to 8.59×），4-bit Quantix因更多centroids和更高memory bandwidth需求通常比3-bit慢但精度更高

