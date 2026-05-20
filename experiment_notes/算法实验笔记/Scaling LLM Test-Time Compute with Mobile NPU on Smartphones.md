## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- 属于算法pipeline的实现是什么？实验比较什么？
  提出面向Qualcomm Hexagon NPU的端到端LLM test-time scaling推理系统。核心算法pipeline：(1) Hardware-aware fine-grained tile quantization：将权重按HMX FP16 tile layout（tile级column-major、tile内每两行permute）重排后，以2×16 tile片段为quantization group做group size 32的4-bit量化。量化后通过group coalescing将8个quantization group合并为一个128-byte super-group，使256个INT4值正好填满一个HVX 1024-bit vector register，scale连续存放。(2) LUT-based Softmax：利用safe softmax保证exp输入非正，预计算64KiB FP16 exp LUT（仅x≤0的32768个entry），通过vgather指令查表替代exp2多项式展开，消除VLIW顺序依赖。(3) LUT-based dequantization：用vlut16指令将INT4权重值直接映射为FP16，并用LUT广播多group scale，替代传统mask-unpack-convert流程。(4) Test-time scaling：将Best-of-N/Beam Search的多候选并行采样映射到decode batch，填充HMX 32×32 tile的空置行。实验比较：accuracy-latency Pareto frontier（Best-of-N/Beam Search vs base model scaling），算子消融（LUT Softmax vs F32/F16 polynomial exp，tile quantized GEMM vs conventional layout/HMX layout only/no dequantization upper bound），端到端decode/prefill throughput对比（Ours vs llama.cpp OpenCL backend vs QNN FP16）。

- 硬件平台是什么，配置是什么。
  NPU性能实验：OnePlus Ace3 (Snapdragon 8 Gen 2, Hexagon V73)、OnePlus 12 (Snapdragon 8 Gen 3, Hexagon V75)、OnePlus Ace5 Pro (Snapdragon 8 Elite, Hexagon V79)。部分准确率实验用NVIDIA RTX3090服务器。Hexagon NPU架构：6-8 scalar VLIW threads、4-6 HVX vector units (1024-bit registers)、1-2 HMX matrix units、1MiB L2 cache、8MiB TCM、DMA ~60GB/s、l2fetch 20-30GB/s。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-1.5B/3B/7B-Instruct、Llama3.2-1B/3B-Instruct。PRM scorer：Skywork-1.5B-PRM。数据集：MATH500、GSM8K（数学推理 pass@1）、WinoGrande、MMLU、Wikitext-2（PPL）。统一0-shot CoT prompt。量化方案：多数矩阵Q4_0 (~4.5 BPW)，FFN down矩阵Q8_0 (~8.5 BPW)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：主仓库 https://github.com/haozixu/llama.cpp-npu，算子库 https://github.com/haozixu/htp-ops-lib。算法pipeline：
  1. 离线阶段：HuggingFace weight→按HMX tile layout（32×32 tile column-major + tile内2-row permutation）重排→在memory order上按group size 32做4-bit group quantization（等于2×16 tile片段为单位量化）→8个group coalesce为super-group（256个INT4填满一个HVX register，scale连续存放）→输出HMX layout GGUF格式。
  2. 在线decode：CPU侧llama.cpp Hexagon NPU backend将activation/KV cache/operator request通过rpcmem/dmabuf共享内存写入→手动cache flush→FastRPC远程NPU session启动→NPU侧线程轮询共享内存接收请求。HVX用vlut16查表将INT4权重转FP16并广播scale，HMX执行FP16 tile-level inner product。进入attention时FlashAttention按Q/KV tile流式计算，HMX算QK和PV，HVX做rowmax/rowsum/LUT_Exp。
  3. Test-time scaling：Best-of-N保留多个候选路径→batch size=B时HMX的32×32 tile有B行有效计算（vs单路径decode仅1行）→利用原本空置的matrix compute→最终由PRM/ORM scorer选择最优输出。Beam Search在中间步骤用PRM打分剪枝低质量路径。lm_head和logits保留在CPU（因Hexagon NPU 32-bit虚拟地址空间限制），batch size=16时CPU logits计算占比接近或超过50%。
