## JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现三类自定义CUDA kernel：(1) Fused smoothing + quantization kernel：整合smoothing transformation、scale/zero point计算与参数重排、KV cache INT2打包。FAVP技术将absmax计算限制在离线校准的稀疏channel集（<2% channels），使quantization kernel避免4.43× naive runtime smoothing overhead（64K seq len下，对比无smoothing baseline仅增加约1×开销）。(2) Memory-efficient ring buffer token cache kernel：预分配ring buffer替代KIVI/SKVQ的sliding-window tensor concatenation，以指针切换和分段量化避免decoding中频繁内存分配与拷贝。(3) Mixed-precision attention kernel：将INT2 dequantization与attention融合为单一kernel。包含两项优化：① INT2-to-FP16高效unpacking：利用FP16在[1024,2047]区间共享exponent=1024的特性，将2-bit值放入mantissa再bitwise OR设置exponent (R2=R2|0x64006400)，再减去1024得到FP16值，每条指令处理两个值（仅需lop3/or/sub三条指令 vs naive每值≥4条指令）。② Unified parameter block layout：将scale、zero point、smoothing factor等四类参数按thread block访问pattern合并对齐，减少memory transactions（示例从20次降至8次）。实验比较kernel runtime：对比SKVQ、KIVI、QServe、FA2 attention kernel。在128K seq、hidden size 4096、32 KV heads下，JanusQuant kernel speedup 6.17× over KIVI、1.69× over QServe、平均1.99× over FA2。Breakdown实验评估FAVP对quantization kernel的改善（64K seq下将4.43× runtime smoothing overhead降至接近无smoothing baseline）和unpacking/parameter reorg对attention kernel的改善（平均1.99×和3.05× over naive mixed-precision baseline）。

- 后端平台是什么，配置是什么。
  NVIDIA A100-PCIE-40GB（单卡）。CUDA 12.6。kernel编译为standalone shared library (.so)，通过Pybind和FlashInfer包装为Python extension。

- 评估性能的软件/脚本是什么。修改了什么。
  Nsight Compute用于Roofline分析（识别attention kernel在2-bit dequantization fused后从memory-bound变为compute-bound）。Kernel-level实验100 warm-up + 10000 runs取平均。量化kernel breakdown分析absmax calculation占quantization kernel >80% overhead（Figure 15a）。端到端实验10 warm-up + 100 runs取平均。修改：开发约3500行CUDA/C++和2500行Python，CUDA kernels包括token cache处理、fused smoothing+quantization、fused dequantization-attention三类。Python侧提供继承PyTorch nn.Module API的custom attention module，通过Pybind和FlashInfer调用CUDA kernels，支持Llama/Mistral/Vicuna/Qwen模型族。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  论文承诺artifact will be released但截至分析无法确认公开仓库。Kernel使用流程：
  1. FAVP离线校准：部署前在calibration dataset上运行一次（数分钟），为每个attention layer记录absmax频繁channel集（超过90%层仅需<2% channels）。
  2. Decoding时，fused quantization kernel执行：读取FAVP记录的sparse channel indices→仅扫描这些channel计算per-token absmax→smoothing factor = max(|K_i|)^0.5→smoothing transformation→per-channel/per-token group quantization→INT2 packing→追加到quantized KV cache。
  3. Mixed-precision attention kernel执行：每thread block加载unified parameter block→unpack INT2 values via bitwise ops (lop3/or/sub)→dequantize→与FP16 recent KV一起参与attention compute。Kernel利用task parallelism（不同thread block处理quantized/FP16 segments）和asynchronous execution重叠计算与访存。
  4. Kernel可编译为.so并通过Python extension调用，兼容PyTorch/Transformers serving框架。

