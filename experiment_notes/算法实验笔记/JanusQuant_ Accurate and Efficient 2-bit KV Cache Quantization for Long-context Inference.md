## JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出RtSmooth，一种面向2-bit KV cache量化的运行时per-token smoothing算法。核心设计：(1) K cache使用per-token smoothing transformation + per-channel group quantization，V cache使用per-token group quantization；(2) 运行时动态计算每个token的smoothing factor = max(|K_i|)^0.5，缩小group内值域范围，降低quantization error上界（ε_smooth(gp) ≤ s_smooth(gp)/2 < s_gp/2）；(3) FAVP (Fast Absmax Value Positioning)离线校准识别每层最可能持有absmax的稀疏channel集（超过90%层涉及<2% channel），运行时仅扫描这些channel计算smoothing factor；(4) decoding期间新生成KV token先保留FP16，每g=32步量化缓冲token。实验比较accuracy (perplexity on WikiText2/C4, LongBench 8 tasks)和efficiency (kernel runtime, serving latency, end-to-end GPU time, KV cache memory usage)。对比baselines包括FP16 FA2、RTN (Round-To-Nearest per-token group quantization)、Atom (2-bit/4-bit)、QServe (2-bit/4-bit)、SKVQ、KVQuant、KIVI (2-bit)、DuoAttention (KV selection)。

- 硬件平台是什么，配置是什么。
  单机4×NVIDIA A100-PCIE-40GB GPU，效率实验（端到端+kernel-level）均使用单张A100。软件栈：PyTorch 2.4.0 + CUDA 12.6。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-30B、Llama2-7B/13B、Llama3-8B、Mistral-7B、Vicuna-7B/13B、Qwen-2.5-32B。Llama-30B/Llama2/Vicuna使用torch.float16，Llama3/Mistral/Qwen使用torch.bfloat16。数据集：WikiText2（perplexity，sequence length 2048）、C4、LongBench（8个long-context multi-task：LCC, TriviaQA, RepoBench-P, QMSum, SAMSum, MultiNews, Qasper, TREC）。校准使用WikiText2 training set中128个8K-length样本。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  论文承诺artifact will be released as open-source repository，但截至分析无法确认公开官方代码仓库。算法pipeline：
  1. 离线校准阶段：用FAVP在128个WikiText2 8K样本上为每层记录最可能持有absmax的稀疏channel集（<2% of total channels），校准仅需数分钟。
  2. Prefill阶段：处理输入prompt生成KV cache，使用full-precision KV token参与attention保证accuracy。
  3. Decoding阶段：新生成token t的K_t、V_t先写入预分配ring buffer保留FP16精度→当buffer中旧segment达到g=32个token时，对K cache执行RtSmooth量化（FAVP快速定位absmax channels→计算smoothing factor max(|K_i|)^0.5→per-token smoothing→per-channel group quantization打包为INT2），对V cache执行per-token group quantization→量化后segment追加到低精度KV cache。
  4. Attention计算：mixed-precision kernel同时处理2-bit quantized KV和FP16 recent KV，在同一kernel内完成INT2-to-FP16 unpacking + dequantization + attention。
  5. Ring buffer容量为用户可配的g整数倍，默认2g；容量为n*g时，每次decoding至少保留(n-1)*g个最近FP16 token。
