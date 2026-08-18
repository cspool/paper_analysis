## Concrete 工具链与 FHELinAlg dialect（Concrete Compiler / Concrete-ML / Concrete Optimizer / TFHE-rs）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是 Zama 公司维护的开源 TFHE 全栈工具链（https://github.com/zama-ai）：TFHE-rs 是 Rust 实现的 TFHE 库（Boolean 与整数密文，FlashTFHE 的 CPU 测量基线，commit 4c9b081）；Concrete Compiler 是 MLIR 之上的 TFHE 编译器（把高层 IR 编译为 FHE 等价运算，定义 FHELinAlg dialect——带精确类型与参数集的 TFHE 操作方言，v2.7.0、commit b7793aeb）；Concrete Optimizer 是安全/性能参数搜索器（commit 1da7347，联合 Lattice Estimator 风格安全估计与噪声分析）；Concrete-ML 是 PyTorch/scikit-learn 模型到 FHE 的编译器前端（PTQ 量化 + 自动编译，v1.6.1、commit 8681124）。整链作用：数据科学家用 Python/ML 框架写模型 → 自动生成带安全参数的 TFHE 程序，是 multi-bit TFHE 实际落地的标准软件栈。
- 在 FlashTFHE 中该工具链是编译器的基座与 workload 生成器：编译器"integrates directly with the Concrete toolchain, inheriting its parameter selection, quantization flow, and established software ecosystem"，输入即 Concrete 的 MLIR FHELinAlg dialect。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 完整流程（以 quantized GPT-2 decoder layer 为例）：PyTorch 预训练 GPT-2 → Concrete-ML 做 7-bit 量化/6-bit rounding 并编译（期间 Concrete Optimizer 搜索满足 128-bit 安全与 p_err≤2^-14 的参数，产出 Table II 的 n/l_b/N/k 与量化尺度）→ Concrete Compiler lower 到 MLIR FHELinAlg dialect（每条 TFHE 操作带类型与参数集）→ FlashTFHE 编译器消费该 dialect：展开 batched 操作（matmul、multi-LUT）为 per-ciphertext primitives → 依赖分析 → KS-dedup/ACC-dedup 去重 → adaptive batching 与 lane masking → 调度为匹配 temporal-reuse 执行模型的指令流 → BRU/LPU 执行。CPU/GPU 对比基线同样用该工具链实跑（TFHE-rs 跑 CPU，Concrete 工具链跑 GPU）。
- FHELinAlg dialect 的角色：比 MLIR 通用算术方言更接近 FHE 语义——直接携带 LWE/GLWE 类型、参数集与 PBS/LUT 操作，让后端编译器无须再反向推断密码参数，是"类型化方言承载领域语义"的典型 MLIR 用法。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MLIR（https://mlir.llvm.org/）+ LLVM 17（commit f5aec278e8df）之上的多级 lowering 管线；profiling 用 MLIR 与 Concrete Compiler 生成的 IR（论文 Section III-A）。使用：论文所有 7 个 workload（CNN-20/50、KNN、XGBoost、Decision Tree、GPT-2 单头/12 头）均由该链生成；参数搜索结果决定硬件 workload（n 737–1070、l_b 3–8、N 2048–65536、k=1、位宽 3–9）。注意：论文未明确说明是否修改 Concrete Compiler 本体，只说明集成其参数选择与量化流；FlashTFHE 编译器的去重/自适应批处理 pass 是其自研部分（见 KS-dedup/ACC-dedup 条目）。
- MNEMOS 补充视角（ISCA'26，Concrete 作为 GPU PBS 的编译/执行基座）：MNEMOS 构建于 ZAMA 开源栈之上——TFHE-rs v0.11.2（Rust，GPU PBS kernel）、Concrete-Python v2.10.0（TFHE 编译器，为 CNN 应用生成 Para-A~D 参数集并负责 PBS 任务划分）、Concrete-ML v1.9.0（加密 CNN 推理的 CUDA backend）。对编译/执行栈的两处修改：(1) 修改 Concrete 后端的任务划分策略——默认按内部启发式在 CPU/GPU 间分配 PBS 任务，改为全部 PBS 操作独占 offload 到 GPU（保证公平对比）；(2) 修改 Concrete-ML 的 CUDA backend 使所有操作（含 PBS）全部在 GPU 执行，得到全 GPU baseline。MNEMOS 的创新主要在 CUDA kernel 层（BSK 分块、Tensor Core FFT、跨迭代融合），未修改 Concrete 编译器本体；baseline 的"Rotation→FFT 融合为 kernel 1、其余为 kernel 2"的双 kernel 融合策略被 MNEMOS 的跨迭代融合取代。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
