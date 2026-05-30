## GGUF (GPT-Generated Unified Format)

术语解释
GGUF 是 llama.cpp 项目定义的一种二进制模型文件格式，用于高效存储和加载量化后的 LLM/SLM 权重，支持在 CPU 和边缘设备上进行低内存推理。GGUF 是 GGML 格式的后继者，支持多种量化精度（2-bit 至 8-bit）和多种模型架构。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GGUF 格式设计用于 llama.cpp 推理引擎的核心场景：在消费级硬件（CPU、Apple Silicon、手机）上以最少的内存运行 LLM。文件结构包含：
1. **Header**：魔数（GGUF）、版本号、tensor 数量、metadata 键值对数量。
2. **Metadata 键值对**：模型架构名（如 "llama"）、上下文长度、hidden size、层数、attention heads 数、词汇量、tokenizer 信息、量化格式等。
3. **Tensor 信息**：每个 tensor 的名称、维度、offset、量化类型。
4. **Tensor 数据**：实际的量化权重矩阵，按 offset 顺序排列。

支持的量化类型：Q2_K, Q3_K_S/M/L, Q4_0, Q4_K_S/M, Q5_0, Q5_K_S/M, Q6_K, Q8_0, F16, F32 等。其中 K-quant（k-量化）是对各层按重要性分配不同精度的混合精度量化方案。

从系统架构角度拆解术语，给出在系统架构中运转流程的具体例子。
MobiLlama 使用 GGUF 格式部署到边缘设备的流程：
```
# 1. 模型转换（离线）
MobiLlama PyTorch checkpoint (FP16)
  → convert.py (llama.cpp 工具)
  → MobiLlama-0.5B-Q4_K_M.gguf (4-bit 量化, ~400MB)

# 2. 边缘设备部署（i7 CPU 示例）
llama.cpp 加载 GGUF 文件:
  1. 解析 header → 获取模型架构参数 (hidden=2048, layers=22, heads=32)
  2. 解析 metadata → 获取量化格式 (Q4_K_M), tokenizer (32000 vocab)
  3. 读取 tensor infos → 分配内存 buffer (约 799MB 总内存)
  4. mmap() 映射 GGUF 文件 → 零拷贝加载权重
  5. 推理循环:
     - 读取输入 token
     - 按层顺序:
       a. 从 mmapped 内存读取量化权重 block
       b. Dequantize (Q4_K_M → FP16/F32)
       c. 执行 MatMul (CPU SIMD 加速，如 AVX2/NEON)
       d. 释放 dequantized buffer
     - 采样输出 token → 重复

# 3. 关键指标 (MobiLlama 0.5B Q4_K_M, i7 CPU)
Avg Tokens/Sec: 36.32
Avg Memory: 799 MB
Battery: 4.86 mAH/1k tokens
CPU Utilization: 24.64%
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
转换流程：使用 llama.cpp 提供的 `convert_hf_to_gguf.py` 将 HuggingFace 格式的模型权重转换为 GGUF 格式。支持的量化工具：`llama-quantize`（将 FP16 GGUF 量化为 Q4/Q5/Q8 等格式）。

部署优势：(1) 单文件分发——模型架构、权重、tokenizer 全部打包；(2) mmap 零拷贝——加载速度极快；(3) 混合精度 K-quant——自动按层重要性分配精度；(4) CPU 优化——利用 AVX2/AVX512/NEON SIMD 指令集。

在 MobiLlama 中的使用：论文展示 GGUF 4-bit 格式在 i7 CPU（36.32 tok/s）和 Snapdragon-685 手机（7.02 tok/s）上的性能。GGUF 的混合精度量化使 0.5B 模型可在仅有 770MB RAM 的智能手机上运行。

涉及论文标题：
- MobiLlama Small Language Model tailored for edge devices

---
