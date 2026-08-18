## GGML（开源 GPU 加速张量库）与 ggml_mul_mat kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GGML 是开源的机器学习张量库（ggml-org/ggml，C/C++ 实现），llama.cpp 的底层算子库，面向本地 LLM 推理，提供量化张量与 GPU kernel（CUDA/Metal/Vulkan/SYCL/WebGPU/CPU 多后端）。其核心算子 ggml_mul_mat（含 ggml_mul_mat_vec 变体）执行量化矩阵乘：对 4-bit 类型（Q4_0/Q4_1/Q4_K）按块解量化后做 GEMM/GEMV；Web 证据显示现代版本按 batch 在 MMVQ（逐行 GEMV，低延迟）与 MMQ（MFMA-tiled GEMM，高通量）间分派。PRowhammer（ISCA'26）把它作为 LLM 攻击目标：llama.cpp 高层函数大量调用 ggml_mul_mat；压缩 nv_fatbin 14MB；单 bit-flip 使 Llama-2-7B/Mistral-7B/Falcon-7B（4-bit 量化）在 Google Natural Questions 100 问上的平均 BERTScore 从 0.58–0.62 跌到 0.25–0.30（输出 # 串或跨语言乱码；部分位产生语法连贯但语义错误的文本）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
运转链：llama.cpp 前向（QA 任务）→ 高层函数调 ggml_mul_mat（每 token 的矩阵乘，如 M=1 解码或大 M 预填充）→ CUDA 后端按量化类型选 mul_mat 实现（4-bit Q4 在 GPU 上 dequant+GEMM）→ SASS 从 GGML 的 .nv_fatbin 动态链接执行。PRowhammer 的利用：以 ggml_mul_mat 构造 wrapper 做 bit-flip profiling（RTX A6000/5060/4090 分别得 33/55/64 个可利用翻转）→ 因该 kernel 被模型反复复用，翻转位跨模型转移显著 → 攻击阶段对真实模型应用 → 解码输出被破坏。flip 模拟开销 500–700ms/次，50000 次 trial 中 cuBLASLt 无可利用位、需剪枝策略（GGML 41–99 个可利用位/10000 trial）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：开源（https://github.com/ggml-org/ggml），llama.cpp（https://github.com/ggml-org/llama.cpp）调用；GGML 共享库含多架构 SASS 的压缩 nv_fatbin。使用：正常使用 `llama-cli -p "..."` 跑量化模型推理；攻击侧用 run_profile_ggml.sh 跑五阶段 profiling（bitflip_data_ggml.csv 第一列索引对应 outs_ggml/stdout/out_err_<index>.log 的 corrupted 输出），再对真实模型施加单 bit-flip 验证 BERTScore 退化。攻击例子（Listing 3/4）：正确输出 "Google is a multinational technology company" → 翻转后 "Unterscheidung sehialog Dhorn Jurivers H"（乱码）；或 "The dog's name on Tom and Jerry is Spike." → "In the Tom and Jerry cartoon series, the dog's name is Momo."（连贯但错误）。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU
