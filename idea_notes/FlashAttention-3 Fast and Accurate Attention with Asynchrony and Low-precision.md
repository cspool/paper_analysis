## FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision

- baseline方法是什么？
  **FlashAttention-2 (Dao, 2023)**：同步执行的tiled block-wise exact attention kernel。核心设计：(i) 将Q沿seqlen维度分块以增加并行度（vs FlashAttention-1仅并行化batch和heads）；(ii) 内循环沿KV blocks迭代，每一步串行执行QK^T GEMM → wait → softmax → PV GEMM → wait；(iii) 所有warps统一角色，同时执行数据搬运和计算；(iv) 仅支持FP16/BF16精度；(v) 基于Ampere架构设计，未利用Hopper特有的TMA、WGMMA异步、FP8 tensor core、setmaxnreg等能力。
  
  全栈执行例子（H100 GPU, BF16 forward, N=8192, d=128, 16 heads）：
  - **模型推理算法层**：标准scaled dot-product attention $\mathbf{O} = \text{softmax}(\mathbf{QK}^T/\sqrt{d})\mathbf{V}$，通过tiled block-wise online softmax实现exact computation。FlashAttention-2并行化策略：outer loop over Q blocks（$T_r$路并行，不同CTA处理不同Q tiles），inner loop over KV blocks（sequential per CTA）。
  - **系统框架层**：PyTorch集成，`flash_attn_func(q,k,v)`作为drop-in replacement。HuggingFace Transformers、GPT-NeoX等框架通过替换attention模块调用。框架对底层kernel执行无细粒度控制——kernel内同步/异步调度对框架透明。
  - **编译框架层**：论文未明确说明。CUDA C++手写kernel（基于CUTLASS或自编），非编译器自动生成。FlashAttention-2 in Triton版本利用Triton compiler自动tile和调度，但未使用Hopper-specific指令（TMA/WGMMA异步）。
  - **kernel调度层（核心缺陷）**：单个同步CUDA kernel。内循环迭代j：（1）从HBM加载$\mathbf{K}_j$, $\mathbf{V}_j$到SMEM；（2）Tensor core MMA: $\mathbf{S}_{ij}=\mathbf{Q}_i\mathbf{K}_j^T$ → warp-level同步等待 → （3）CUDA core softmax: rowmax + EX2 + rowsum + rescale → 等待$\mathbf{V}_j$加载完成 → （4）Tensor core MMA: $\mathbf{O}_i += \tilde{\mathbf{P}}_{ij}\mathbf{V}_j$ → 同步等待。问题：(a) Tensor core算完BMM1后进入idle等待softmax完成（softmax throughput仅3.9 TFLOPs/s vs matmul 989 TFLOPs/s）；(b) 数据搬运与计算串行——HBM→SMEM的加载不能与当前迭代的GEMM重叠；(c) 所有warps统一角色，register分配不优化；(d) H100仅35%利用率的根源：同步模型使tensor core大量空闲等待non-matmul操作和memory操作。
  - **硬件架构层**：NVIDIA H100 SXM5 GPU (Hopper)。FP16 Tensor Core: 989 TFLOPs/s。TMA硬件单元支持异步HBM↔SMEM拷贝但未使用。WGMMA指令支持异步tensor core操作但未使用。FP8 Tensor Core（2× throughput）未使用。setmaxnreg动态register分配未使用。FlashAttention-2仅在H100上达到~350 TFLOPs/s（35% utilization），而optimized GEMM达到800+ TFLOPs/s（80-85%）。

  Baseline缺陷：
  - (a) **同步模型导致tensor core利用率低**：FlashAttention-2的内循环是同步的——BMM1完成后需等待softmax（3.9 TFLOPs/s特殊函数）完成才能发射BMM2，tensor core在此期间空闲。FP16 head_dim=128时，matmul FLOPs:exponential FLOPs比=512:1，但exponential throughput比matmul低256×，exponential可占用50% cycle time。
  - (b) **数据搬运与计算无重叠**：HBM→SMEM的K/V加载与tensor core计算串行，TMA硬件单元的异步能力未利用。
  - (c) **Register分配不优化**：所有warps均分register，数据搬运仅需1 thread/warp但持有满额register，tensor core计算需要大量register但受限。
  - (d) **未利用FP8低精度**：Hopper FP8 tensor core提供2× throughput，但FlashAttention-2不支持。直接使用FP8 per-tensor量化导致高数值误差（RMSE 2.4× worse），尤其在有outlier features的LLM中。
  - (e) **Kernel launch overhead**：非persistent kernel，每次launch的prologue（Q加载）和epilogue（O写回）期间tensor core空闲。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashAttention-3：asynchronous warp-specialized attention kernel exploiting Hopper hardware features**。三个核心技术创新：

  **(1) Producer-Consumer asynchrony via warp-specialization + pingpong scheduling**
  解决缺陷(a)(b)(c)：将CTA内warps划分为producer（仅发射TMA）和consumer（仅发射WGMMA+softmax）。Producer通过TMA异步加载Q/K/V到circular SMEM buffer，不阻塞consumer的GEMM执行。两个consumer warpgroups通过bar.sync实现pingpong——当warpgroup 1执行softmax时，warpgroup 2执行GEMM，tensor core持续被占用。setmaxnreg让consumer获更多register（用于WGMMA），producer释放register（TMA只需1 thread）。
  
  **(2) Intra-warpgroup GEMM-softmax overlapping (2-stage pipeline)**
  解决缺陷(a)：通过寄存器缓冲$\mathbf{S}_{\text{next}}$打破迭代间依赖。迭代j：发射WGMMA(QK^T) of iter j+1（异步，不等待）→ 发射WGMMA(PV) of iter j（异步，不等待）→ 等待WGMMA(QK^T)完成 → softmax on iter j+1（与WGMMA(PV)重叠）→ 等待WGMMA(PV)完成 → rescale $\mathbf{O}_i$。SASS分析验证：softmax指令被compiler重排到第一个WGMMA之前，第一个WGMMA与softmax的FP32→FP16转换交错执行。
  
  **(3) Hardware-accelerated FP8 with block quantization + incoherent processing**
  解决缺陷(d)：Q/K量化为FP8 e4m3 with per-block scaling。Incoherent processing：Q和K先乘随机正交矩阵M（Hadamard × random sign diagonal），$\mathbf{M}\mathbf{M}^\top = I$不改变attention结果，但将outlier"摊平"到所有维度——每个$(\mathbf{QM})$和$(\mathbf{KM})$的元素是Q/K元素的随机线性组合，消除孤立大值对量化的影响。Block quantization fused with rotary embedding（memory-bound操作，零开销）。解决FP8 WGMMA的layout constraints：k-major Q/K（TMA load不变），in-kernel V transpose via LDSM/STSM + byte_perm，FP32→FP8 register exchange via byte_perm + shfl_sync。

  全栈执行例子（H100 GPU, BF16/FP8 forward, N=8192, d=128, 16 heads）：
  - **模型推理算法层**：Attention数学定义不变($\mathbf{O}=\text{softmax}(\mathbf{QK}^T/\sqrt{d})\mathbf{V}$)，计算重组为warp-specialized异步流水线。FP8 variant：$\mathbf{O} = \text{softmax}((\mathbf{QM})(\mathbf{KM})^\top/\sqrt{d})\mathbf{V}$ with per-block quantization of QM, KM, V。Algorithm 1+2提供完整的CTA-view和consumer warpgroup-view伪代码。
  - **系统框架层**：与FlashAttention-2相同的PyTorch接口（`flash_attn_func(q,k,v)`），drop-in replacement。计划集成到PyTorch core。开源：https://github.com/Dao-AILab/flash-attention。
  - **编译框架层**：论文未明确说明。使用CUTLASS primitives（WGMMA, TMA, setmaxnreg, pipeline barriers）手写CUDA C++ kernel。NVCC compiler重排指令以实现overlap——SASS分析验证compiler正确调度了WGMMA与softmax的交错执行。
  - **kernel调度层（关键创新）**：
    - **Warp-specialization**：CTA = 1 producer warp（仅TMA loads）+ 2 consumer warpgroups（各2 warps, WGMMA+softmax）。Producer使用s-stage circular SMEM buffer pingpong调度——异步加载Q/K/V tiles，commit到pipeline barrier通知consumer。Consumer warpgroups交替执行：warpgroup 1 GEMMs while warpgroup 2 softmax，反之亦然。
    - **2-stage pipelining per consumer warpgroup**：prologue: WGMMA(QK₀, sync) → softmax → mainloop: for j=1..T_c-1: WGMMA(QKⱼ, async) → WGMMA(PVⱼ₋₁, async) → wait QK → softmax on Sⱼ → wait PV → rescale O → copy S_next→S_cur。Tensor core和CUDA core通过异步执行实现了GEMM与softmax的重叠。
    - **FP8 support**：per-block quantize Q,K,V → in-kernel V transpose (LDSM→byte_perm→STSM) → SS-WGMMA FP8 QK^T → softmax → RS-WGMMA FP8 PV with register layout conversion (byte_perm→shfl_sync→byte_perm)。
    - **Persistent kernel**：132 threadblocks（=132 SMs），每个处理多个Q tiles，重叠后一个tile的prologue与当前tile的epilogue。
    - **Inference优化**：split-KV (Flash-Decoding) + GQA packing + PagedAttention with TMA block table。
  - **硬件架构层**：NVIDIA H100 SXM5 (Hopper)。TMA硬件单元：异步HBM↔SMEM拷贝，producer warp独占使用。FP8 Tensor Core：2× BF16 throughput。WGMMA异步指令：warpgroup-level tensor core操作，可异步发射不阻塞CUDA core执行。setmaxnreg：动态register重分配。bar.sync：warpgroup间同步屏障。达到：BF16 forward 840 TFLOPs/s (85% utilization, 2.4× vs FlashAttention-2的35%)，FP8 forward 1.3 PFLOPs/s，FP8 RMSE 2.6× better than per-tensor quantization baseline。
