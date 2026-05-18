## JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

- baseline方法是什么？
  Baseline方法分为两类：(1) **效率导向2-bit量化系统（Atom/QServe）**：采用per-token group quantization等硬件友好策略，QServe使用类似SmoothQuant的per-channel smoothing并通过离线校准将smoothing factor融入前层权重以降低runtime开销。全栈执行例子：算法层per-token group quantization→系统框架层离线校准smoothing factor融合到权重→编译框架层论文未明确说明→kernel调度层标准量化kernel，row-major访存→硬件架构层A100 GPU。缺陷：2-bit下每组仅4个量化值，K cache中outlier channel放大同组误差（Atom MSE 1.0352 vs QServe 0.5552 vs ideal 0.3734）；离线校准的静态smoothing factor无法适应不同请求和序列长度的per-channel absmax波动（可超4×），导致2-bit精度崩塌（Llama2-7B perplexity: Atom-2bit 103.05, QServe-2bit 11.36 vs FP16 5.47）。(2) **准确率导向2-bit量化方法（SKVQ/KVQuant/KIVI）**：通过channel reordering、dense-and-sparse quantization、recent token reservation维持2-bit准确率。全栈执行例子（以SKVQ attention layer为例）：算法层channel reordering + sliding-window recent token reservation + outlier detection/extraction→系统框架层Transformers/PyTorch→编译框架层论文未明确说明→kernel调度层多步分立kernel：retain recent FP16 tokens (tensor concatenation)→detect/extract outliers→quantize剩余KV→separate dequantization kernel→attention kernel→硬件架构层A100 GPU。缺陷：outlier handling overhead（prefill ~20%）、caching overhead from tensor concatenation（prefill ~20% / decode ~15%）、separate dequantization kernel overhead（prefill ~45% / decode ~80%），三项合计prefill阶段超85% runtime，decoding阶段超97% runtime，端到端延迟甚至高于FP16。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**JanusQuant**，通过RtSmooth runtime smoothing算法联合FAVP fast absmax positioning、memory-efficient ring buffer token cache和mixed-precision attention kernel三项系统优化的算法-系统协同设计。

  论文方法全栈执行例子（以Llama2-7B、128K context、单张A100-40GB decoding为例）：
  - **算法层**：RtSmooth对K cache执行per-token smoothing transformation + per-channel group quantization，对V cache执行per-token group quantization。每个token的smoothing factor = max(|K_i|)^0.5在运行时动态计算，缩小group内值域→降低quantization scale→降低error upper bound（ε_smooth ≤ s_smooth/2 < s/2）。Decoding中每g=32步将FP16 buffered recent tokens量化。无需channel reordering或sparse outlier path，保留positional alignment。
  - **系统框架层**：约2500行Python实现，基于PyTorch和Transformers。Custom attention module继承PyTorch nn.Module API，通过Pybind和FlashInfer调用CUDA kernels。支持Llama/Mistral/Vicuna/Qwen模型族。Pre-allocated ring buffer token cache替代sliding-window concatenation。
  - **编译框架层**：论文未明确说明。CUDA kernels编译为standalone .so。
  - **kernel调度层**：约3500行CUDA/C++。三类kernel：(1) Fused smoothing+quantization kernel：FAVP限制absmax计算到离线校准的稀疏channel集（<2% channels），将smoothing transformation、参数计算、KV cache INT2 packing融合，避免4.43× naive runtime smoothing overhead。(2) Ring buffer token cache kernel：预分配buffer+指针管理，以分段量化替代频繁tensor concatenation。(3) Mixed-precision attention kernel：将INT2 dequantization fused into attention，高效unpacking (lop3/or/sub, 3指令处理2值) + unified parameter block layout (memory transactions 20→8)。Task parallelism and async execution重叠计算与访存。
  - **硬件架构层**：NVIDIA A100-PCIE-40GB（单卡），PyTorch 2.4.0 + CUDA 12.6。

  对应解决Baseline缺陷：
  **(1) Atom/QServe 2-bit精度崩塌**：RtSmooth在运行时动态计算per-token smoothing factor适应不同请求和序列长度的KV cache分布变化，而非依赖静态离线校准→Llama2-7B perplexity 5.80 vs QServe-2bit 11.36，LongBench 8任务平均分保留99% FP16 accuracy。
  **(2) SKVQ/KVQuant outlier handling overhead (~20%)**：RtSmooth不需要检测和抽取outlier到单独稀疏路径，而是通过smoothing将outlier影响均化到group内→消除explicit outlier detection/extraction开销。FAVP将absmax扫描限制在<2% channels，进一步降低runtime smoothing成本。
  **(3) SKVQ/KIVI recent token reservation overhead (prefill ~20% / decode ~15%)**：Ring buffer预分配+指针切换替代tensor concatenation，避免decoding中每次token追加触发内存reallocation和data copy。缓存容量n*g token时每次decoding至少保留(n-1)*g个FP16 recent token。128K context下额外FP16 token仅占0.05%总token数。
  **(4) SKVQ/KVQuant separate dequantization overhead (~45%-80%)**：Mixed-precision attention kernel将dequantization融合进attention，消除独立dequantization kernel launch和全局内存往返。INT2-to-FP16高效unpacking降低compute intensity避免kernel成为compute-bound，unified parameter block减少memory transactions。Decoding 100 tokens总延迟：5.64× over FA2, 5.84× over KIVI, 4.45× over QServe, 2.50× over DuoAttention。KV cache memory 5.30× reduction over FP16。
  **(5) 整体trade-off**：JanusQuant接受额外smoothing factor存储（avg bit-width 3.008 vs 3.000 for KIVI/SKVQ）换取端到端加速，与KVQuant non-uniform方案（avg 2.320 bits）相比在压缩率上不占优但在实际decoding速度上显著领先。
