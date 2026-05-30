## FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现FlashInfer v0.2：基于FlashAttention-2（Turing/Ampere/Ada架构）和FlashAttention-3（Hopper架构）算法的CUDA/CUTLASS可定制attention kernel模板系统。核心kernel调度设计包括：(i) **Block-Sparse Row (BSR) attention kernels**——将KV-cache统一为block-sparse矩阵格式，支持任意block size $(B_r, B_c)$，通过从分散的global memory加载sparse tile到contiguous shared memory再调用dense tensor core MMA（使用LDGSTS 128B异步拷贝指令），支持vector-sparse（fine-grained sparsity with $B_c$=1）；(ii) **多tile size microkernel选择**——FA2 kernel提供tile sizes $(1,16,32,64,128) \times (32,64,128)$，FA3提供64倍数row tile sizes对齐WGMMA要求，根据硬件资源和workload特征（平均query长度、GQA group size $g$）的heuristic自动选择最优tile size；(iii) **JIT编译的attention variant kernel生成**——通过CUDA代码字符串定义的variant functors（QueryTransform, KeyTransform, ValueTransform, OutputTransform, LogitsTransform, LogitsMask）填充CUDA模板，用PyTorch JIT compiler编译并注册为custom operator，支持fused RoPE、soft-cap、sliding window、FlashSigmoid（无softmax）等变体；(iv) **Load-balanced persistent kernel调度**（Algorithm 1）——CPU端planning阶段根据query/KV长度信息计算CTAs间workload分配（类似Stream-K但保证deterministic aggregation order），生成work queue和partial-to-final output index mapping，GPU端persistent attention/contraction kernel按plan执行，兼容CUDAGraph的static grid size要求。

  实验比较：(i) Kernel bandwidth和FLOPs utilization vs FlashAttention main branch（含FA2和FA3），decode和prefill两种模式，batch size 16，3种sequence length分布——constant (1024)、uniform (512-1024)、skewed (Zipf, avg 1024)，A100 40GB和H100 80GB，f16精度；(ii) Fused RoPE+attention kernel vs FlashAttention unfused kernel（RoPE + attention分离），kernel bandwidth utilization对比，Vicuna-13B Streaming-LLM on MT-Bench，recent window size变化（changing window sizes）。

- 后端平台是什么，配置是什么。
  - NVIDIA A100 40GB SXM（Ampere SM80, server-class）：Tensor Core FP16 GEMM throughput约312 TFLOPS
  - NVIDIA H100 80GB SXM（Hopper SM90A, server-class）：支持TMA、WGMMA异步指令，FP16 GEMM throughput约989 TFLOPS
  - CUDA 12.4 + PyTorch 2.4.0，存储和计算均使用FP16精度
  - FlashInfer kernel支持Turing (sm75)到Hopper (sm90a)全系列GPU架构：Ampere/Ada(sm89)使用FA2 algorithm + LDGSTS异步拷贝，Hopper使用FA3 algorithm + WGMMA + TMA（dense contiguous KV-cache）/ LDGSTS fallback（sparse non-affine KV-cache）

- 评估性能的软件/脚本是什么。修改了什么。
  基于FlashAttention-2（Dao, 2023）和FlashAttention-3（Shah et al., 2024）算法的CUDA/CUTLASS模板实现。修改内容包括：
  (1) **Sparse tile loading module**：非contiguous KV-cache地址通过BSR indices数组计算→使用`cp.async` (LDGSTS, 128B width)将分散global memory数据gather到contiguous shared memory→shared memory内数据变为dense tile，后续dense MMA路径与FlashAttention一致；
  (2) **Multi-tile-size microkernel generation**：在CUDA template中参数化query tile size（$T_q \in \{1,16,32,64,128\}$）和K/V tile size（$T_{kv} \in \{32,64,128\}$），compile-time resolve register和shared memory constraint，优先maximize SM occupancy。$T_q=1$使用CUDA core路径（因mma指令min row=16），$T_q \geq 16$使用Tensor Core路径；
  (3) **JIT compiler pipeline**：variant specification（CUDA code定义functors + additional tensors + data types）→ template population → PyTorch `torch.utils.cpp_extension.load_inline` 编译→ DLPack framework-agnostic interface注册custom operator；
  (4) **Load-balanced scheduler**：CPU端$\{l_{qo}(i), l_{kv}(i)\}$输入→计算max KV chunk size $L_{kv}$→split query tiles into KV chunks→sort by length descending→greedy min-cost CTA assignment (Algorithm 1)→输出plan info（CTA work queue + partial/final output index mapping）→async copy to GPU workspace buffer；
  (5) **Persistent kernel设计**：单persistent kernel合并attention+contraction两阶段，fixed grid size兼容CUDAGraph，workspace buffer fixed offset确保CUDAGraph capture指针不变。

  评估脚本：使用CUDA event timing测量kernel wall-clock time，bandwidth utilization = achieved bytes / peak bandwidth，FLOPs utilization = achieved FLOPs / peak FLOPs。Benchmark中decode核心理念——输出是$O(l_{qo})$密集的，FLOPs较低时bandwidth为主要限制。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/flashinfer-ai/flashinfer (Apache-2.0)。已集成进SGLang、vLLM、MLC-Engine等主流LLM serving框架。

  评估原理与流程（以H100 decode kernel, uniform seqlen 512-1024, batch=16, GQA group=4, head_dim=128, f16）：
  1. **Input准备**：Ragged tensors Q/O（shape `[total_tokens, nheads, head_dim]`，各请求token数通过`indptr`数组索引），KV-cache BSR matrix（shape `[total_blocks, B_r, nheads_kv, head_dim]`，`B_r=T_q`matched query tile，`B_c`=1 for page-level sparsity）。对应metadata：`kv_indptr`（row pointers for BSR）、`kv_indices`（column indices for non-zero blocks）。
  2. **Compile-time tile selection**：平均query长度（GQA fused）= $l_{qo} \times g$。若avg≥128选$T_q=128$，avg∈[64,128)选$T_q=64$，依此类推。Register constraint: $T_q \times T_{kv}$ 需与shared memory size constraint同时满足，求解max occupancy tile size。
  3. **CPU scheduler plan**（Algorithm 1）：给定$\{l_{qo}(i), l_{kv}(i)\}_{i=1}^{16}$，hyperparameters α=1, β=1（default cost function $cost(l_q,l_{kv})=\alpha l_q + \beta l_{kv}$）。Compute $L_{kv}= \sum \lceil l_{qo}(i)/T_q \rceil \cdot l_{kv}(i) / \#CTA$。每个query tile $(T_q)$ 的KV split为chunks of max $L_{kv}$，得work queue $W$。Sort $W$ descending→greedy assign to CTAs with min current cost→输出plan info（CTA-wise chunk assignment + partial output aggregation mapping）。
  4. **Persistent attention kernel**：Grid size = compiled constant。每个CTA读取plan info中自己的work queue→逐chunk处理：(a) Load sparse KV tile: compute block indices from BSR metadata→`cp.async` LDGSTS从HBM搬移到SMEM；(b) Load Q tile: dense affine addressing→LDGSTS；(c) $S_{ij}=Q_iK_j^T$ by WGMMA (Hopper) 或 HMMA (Ampere)；(d) Online softmax: rowmax (CUDA core REDUX)→exp (MUFU.EX2)→rowsum→rescale running O and l→$\tilde{P}_{ij}V_j$ by WGMMA/HMMA→accumulate partial O and l。
  5. **Persistent contraction kernel**（与attention合并入同一persistent kernel）：各CTA的attention partial outputs（Attention State: `(O_partial, LSE_partial)`）按plan info index mapping进行$\oplus$ composition（attention compose operator, equation in Section 2.2）→final O。$\oplus$操作：$O_{final} = (\exp(LSE_1)O_1 + \exp(LSE_2)O_2)/(\exp(LSE_1)+\exp(LSE_2))$，$LSE_{final}=\log(\exp(LSE_1)+\exp(LSE_2))$。
  6. **Performance measurement**：CUDA event timing测kernel wall-clock time。Bandwidth = (Q size + KV size + O size read/written bytes + partial O bytes) / time。FLOPs = $(2 \times total\_kv\_tokens \times head\_dim \times nheads_{qo})$ (QK^T + PV GEMM, 2× for MAC) / time。
  7. **Output**：Figure 8——decode kernel bandwidth utilization（FlashInfer显著高于FlashAttention on uniform and skewed distributions因load-balanced scheduler + multi-tile-size选择）；prefill kernel FLOPs utilization（similar but FlashInfer stable across distributions）。Fused RoPE+attention kernel: Figure 9——FlashInfer fused kernel bandwidth util vs FlashAttention unfused kernel (1.6-3.7× higher)。
