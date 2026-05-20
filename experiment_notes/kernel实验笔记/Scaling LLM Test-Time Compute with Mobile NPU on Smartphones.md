## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  在Qualcomm Hexagon NPU (HVX vector + HMX matrix混合架构)上实现两个核心kernel：(1) Hardware-aware tile-quantized dequantized GEMM kernel：离线将权重按HMX FP16 tile layout（tile级column-major，tile内2-row permutation）重排→在memory order上做group size 32的4-bit量化→8个group coalesce为128-byte super-group适配HVX 1024-bit vector register。运行时HVX用vlut16查表将INT4权重转为FP16并广播scale→HMX执行FP16 32×32 tile-level inner product→accumulate到FP32 accumulator。与传统layout相比加速9.65-19.04×，比仅用HMX layout版本加速1.82-3.45×，仅比no-dequantization上界慢27%。(2) LUT-based FP16 FlashAttention/Softmax kernel：利用safe softmax保证exp输入≤0，预计算64KiB FP16 exp LUT（32768个entry存于TCM）→通过vgather指令查表替代多项式exp2→减少VLIW顺序依赖。相比F32 exp加速1.26-2.19×，相比F16 exp加速最高1.60×。算子消融：对比F32 polynomial exp、F16 polynomial exp、no dequantization上界。Attention延迟分解显示Softmax占比随query length增加从39.2%(q=4)升至84.6%(q=32)。

- 后端平台是什么，配置是什么。
  Qualcomm Hexagon NPU (V73/V75/V79三代)：HVX vector units (1024-bit VRF, 4-6个, 单thread FP16 GEMM ~33 GFLOPS)，HMX matrix units (1-2个, FP16 GEMM ~12 TFLOPS)，1MiB L2 cache，8MiB TCM (software-managed)，DMA ~60GB/s，l2fetch 20-30GB/s。HVX scatter/gather操作和所有HMX指令仅可访问TCM。HMX FP16 tile为32×32=2KiB，每两行permuted。手机：OnePlus Ace3 (Snapdragon 8 Gen 2/V73)、OnePlus 12 (Snapdragon 8 Gen 3/V75)、OnePlus Ace5 Pro (Snapdragon 8 Elite/V79)。

- 评估性能的软件/脚本是什么。修改了什么。
  Hexagon NPU operator library (htp-ops-lib) + llama.cpp Hexagon NPU backend。约7K行C/C++和inline assembly，用Hexagon SDK 6.0.0.2 LLVM toolchain编译。修改：(1) GEMM kernel：从conventional column-major layout的scatter-based dequantization改为HMX tile layout + group coalesce的连续写入TCM方案；(2) Softmax kernel：从F32/F16 polynomial exp2改为64KiB LUT-based vgather exp，只存x≤0范围；(3) dequantization: vlut16直接INT4→FP16查表，并用LUT广播scale替代split-broadcast；(4) Attention：实现FP16 FlashAttention on NPU（Algorithm 1: tile-level Q/KV分块、FP16 online softmax、FP32 accumulation on critical ops）；(5) rpcmem/dmabuf共享内存通信替代FastRPC默认RPC降低延迟。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/haozixu/llama.cpp-npu，https://github.com/haozixu/htp-ops-lib。kernel使用例子（Qwen2.5-1.5B decode, batch size=8, OnePlus 12）：
  1. 初始化：CPU调用FastRPC启动Hexagon NPU remote session→NPU侧thread pool初始化→分配rpcmem/dmabuf共享内存区域（模型weight 1056MiB for 1.5B, 2090MiB for 3B under ctx=4096）。
  2. 线性层GEMM kernel：CPU将operator request写入共享内存（含activation pointer, weight pointer, dimensions）→cache flush→NPU thread轮询到请求→DMA将weight tile从DDR搬入TCM→HVX用vlut16查表将INT4→FP16（每个8-bit index映射为16-bit value, vlut16生成一对HVX register）→HVX用vlut16 LUT（scale of 4 groups as LUT content + constant indices）广播scale→HMX加载FP16 activation tile和weight tile→32×32 tile-level inner product→accumulator累加→output tile写回TCM→DMA搬回DDR。
  3. FlashAttention Softmax kernel：HMX执行QK^T tile multiplication（FP16 input, FP32 accumulate）→output S tile in FP16→HVX执行rowmax reduction→S - rowmax（safe softmax）→HVX vgather查64KiB exp LUT（input MSB忽略+left shift 1 bit作为byte offset, 一次vgather收集64个FP16）→HVX rowsum（FP32 accumulate）→HMX执行PV multiplication→online rescale + accumulate O。LUT存储在TCM固定64KiB区域（占总TCM 0.8%）。
  4. Test-time scaling集成：Best-of-N生成B个候选路径→CPU llama.cpp sample层维护B个序列的KV cache→每个transformer layer执行时B条路径的activation rows映射到HMX tile的B行→B=8时相比B=1无显著增加HMX延迟→lm_head/logits在CPU侧计算（B=16时CPU时间占比≥50%）。Beam Search在每step结束时由PRM skimmer评分并剪枝。

