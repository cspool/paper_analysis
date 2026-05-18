## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- baseline方法是什么？
  **模型scaling baseline**：直接增大模型参数（Qwen2.5 3B/7B base model, Llama3.2 3B base model）在手机上运行，但这带来更高的内存占用（3B model dmabuf ~2090MiB, total ~2.4GiB）、带宽需求和功耗。**小模型单路径baseline**：1B/1.5B模型用单路径autoregressive decode，成本低但数学推理准确率不足（Qwen2.5-1.5B MATH500 base accuracy ~50%）。**系统baseline**：llama.cpp OpenCL backend使用Snapdragon Adreno GPU的Q4_0 GEMM kernel，在decode batch size=1时更快，但大batch test-time scaling下无法利用Hexagon HMX矩阵tile的空置行。**QNN baseline**：Qualcomm闭源QNN框架仅支持per-tensor/per-channel粗粒度量化，Llama3.2-1B-Instruct的QNN W4A16在MATH500上准确率从AutoAWQ的15.9骤降至2.1、GSM8K从32.6骤降至3.4，精度损失使test-time scaling不可行。

  **全栈执行例子（baseline: llama.cpp OpenCL + Qwen2.5-1.5B单路径decode）**：
  - 算法层：Q4_0 group quantization (group size 32, conventional column-major layout)，无tile layout transformation，单token autoregressive decode via greedy sampling。
  - 系统框架/Serving层：llama.cpp main executable → OpenCL backend → Adreno GPU command queue。每个decode step：CPU准备activation (shape [1, hidden_dim]) → clEnqueueWriteBuffer → GPU OpenCL kernel执行GEMV（GEMM退化，32-wide warp仅1 lane有效计算）→ clEnqueueReadBuffer → CPU sample next token。
  - 编译框架层：论文未明确说明（OpenCL runtime编译，无自定义compiler pass）。
  - kernel调度层：Adreno GPU OpenCL Q4_0 matmul kernel：column-major weight layout → per-group dequantization (INT4→FP16 via unpack+convert) → GEMV inner product → 单token activation仅利用GPU warp的1/32 lane → Adreno GMEM/TCM未针对HMX tile优化。无FlashAttention on GPU（使用标准attention实现）。
  - 硬件架构层：Snapdragon Adreno GPU (OpenCL)，Hexagon NPU的HMX matrix unit在decode阶段闲置（非目标硬件）。CPU处理所有logits/lm_head、sampling。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **方法概述**：将"更多test-time compute"转化为decode batch parallelism，用Hexagon NPU HMX matrix unit原本空置的tile行做多候选生成(Best-of-N/Beam Search)，让1B/1.5B小模型+test-time scaling达到或超过3B/7B base模型的accuracy-latency/accuracy-energy Pareto frontier。核心技术创新：(1) Hardware-aware tile quantization：权重按HMX FP16 tile layout重排→在memory order上做group size 32量化→group coalescing适配HVX 1024-bit register；(2) LUT-based Softmax/dequantization：用HVX vgather/vlut16查表替代多项式exp和mask-unpack-convert。

  **缺陷→方法映射**：
  - 缺陷1：模型scaling增大memory/带宽/功耗→方法：不增大模型参数，用test-time scaling增加decode batch parallelism。Best-of-N中Qwen2.5-1.5B TTS结果超过3B base accuracy，Beam Search中Llama3.2-1B达到与3B相当效率。1.5B模型batch size=8 decode energy低于3B model batch size=1。
  - 缺陷2：QNN粗粒度量化破坏fine-grained quantization精度→方法：tile-group quantization保留group size 32的fine-grained量化精度，但将layout重塑为HMX tile format。Tile quantization group与conventional group的WinoGrande/MMLU/Wiki PPL差异远小于量化本身的精度损失。
  - 缺陷3：HVX通用计算能力弱（单thread FP16 GEMM仅33 GFLOPS vs HMX 12 TFLOPS），多项式exp和传统dequantization成为瓶颈→方法：LUT-based computation。Softmax用64KiB FP16 LUT + vgather替代exp2 Taylor展开，加速1.26-2.19× vs F32 exp。Dequantization用vlut16直接INT4→FP16 + LUT广播scale，相比conventional layout加速9.65-19.04×。
  - 缺陷4：GPU OpenCL backend在大batch下无法利用HMX tile空洞→方法：将decode batch映射到HMX 32×32 tile的多行。batch size从1增至16时decode throughput随batch增大显著提升（因HMX tile行利用率从1/32升至B/32），而HMX计算时间几乎不增。

  **论文方法全栈执行例子（Qwen2.5-1.5B Best-of-N B=8, OnePlus 12 Snapdragon 8 Gen 3）**：
  - 算法层：Best-of-N parallel sampling (B=8条候选路径) + Skywork-1.5B-PRM outcome reward scorer。Weight采用Q4_0 tile-group quantization: HuggingFace weight→HMX 32×32 tile layout重排（tile级column-major + tile内2-row permutation）→在memory order上group size 32量化（等于2×16 tile片段）→8 group coalesce为128-byte super-group→HVX vlut16查表INT4→FP16 dequantization→HMX FP16 tile-level inner product。
  - 系统框架/Serving层：llama.cpp CPU backend → rpcmem/dmabuf shared memory → FastRPC remote NPU session → Hexagon NPU operator library。CPU维护8条候选路径的KV cache和sampling状态，lm_head/logits在CPU侧计算。NPU侧thread pool轮询共享内存中的operator request。
  - 编译框架层：Hexagon SDK 6.0.0.2 LLVM toolchain编译，无QNN依赖，使用reverse-engineered FP16 HMX指令。离线模型转换脚本：HuggingFace→HMX layout GGUF→llama-quantize(Q4_0/Q8_0/F16混合)。
  - kernel调度层：每个transformer layer的linear层执行时→DMA搬weight tile和activation tile入TCM→HVX vlut16 dequantization+scale broadcast（产生HMX-compatible FP16 tiles）→HMX执行32×32 tile MM（8条候选路径占8 activation rows, 24 rows空置）→accumulate到FP32 internal accumulator→output tile写回TCM。Attention执行：FP16 FlashAttention tile-level Q/KV分块→HMX算QK^T和PV→HVX rowmax+rowsum+LUT_Exp (64KiB TCM LUT)→FP16 online softmax (critical accum in FP32)。DMA→TCM→HVX→HMX→TCM→DMA形成闭环比。
  - 硬件架构层：Snapdragon 8 Gen 3 Hexagon V75 NPU。HVX (1024-bit VRF, 4-6 units)负责dequant/LUT/scale/reduction，HMX (1-2 units, ~12 TFLOPS FP16)负责tile MM，TCM 8MiB承载LUT(64KiB)+weight tiles+activation tiles+KV cache tiles，DMA ~60GB/s搬移DDR↔TCM。CPU负责lm_head/sampling/KV cache管理。Total device power <5W (1.5B model decode)，dmabuf 1056MiB (1.5B, ctx=4096)，total memory ~1.3GiB。

  **关键trade-off**：依赖Qualcomm Hexagon SDK、FastRPC、reverse-engineered FP16 HMX指令，移植到其他NPU需重新适配tile layout和指令。Decode speed受runtime dequantization overhead限制。CPU logits计算在batch size=16时占比≥50%，削弱scaling收益。仅评估数学推理Best-of-N/Beam Search，对不适合并行采样的任务收益不成立。
