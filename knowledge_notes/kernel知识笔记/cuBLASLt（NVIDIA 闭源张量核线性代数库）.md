## cuBLASLt（NVIDIA 闭源张量核线性代数库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
cuBLASLt 是 NVIDIA 的轻量级（Lt）BLAS 库，内含针对张量核（Tensor Core）优化的 GEMM 等 kernel 集合，被 PyTorch、TensorFlow、TensorRT-LLM、Triton Inference Server 等广泛调用（cuBLASLt 由 cuBLAS 内部调用）。Web 证据：cuBLASLt 用 heuristics（cublasLtMatmulAlgoGetHeuristic，按问题规模/GPU 配置/数据类型选最合适 matmul kernel）做运行时 kernel 选择；CUDA 13.x 新增 CUBLAS_GEMM_AUTOTUNE 实测选优。PRowhammer（ISCA'26）把它作为攻击目标：(1) 闭源、GPU kernel 压缩存储（nv_fatbin 压缩后 255MB，整库 335MB）；(2) 含 sm_86 的 3508 个 kernel，但给定矩阵形状/输出维度只调 1–2 个；(3) 分类模型末层线性层调用 cuBLASLt kernel——单 bit-flip 在 cuBLASLt 中即把 ResNet-18/34/50、VGG-16 在 MNIST/FMNIST/CIFAR-10/ImageNet 上的准确率打到随机猜测（ImageNet 最坏 0%，RPL 84.95–100%）；218（MNIST/FMNIST/CIFAR-10）+93（ImageNet）个可利用翻转位，且同一翻转位跨模型/数据集转移有效。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
cuBLASLt 在推理中的调度链：ML 框架（PyTorch）的线性层 → cuBLAS/cuBLASLt API → heuristics 按 (M,N,K,数据类型,GPU) 选 kernel（如 ImageNet 1000 类输出 vs CIFAR-10 10 类输出各对应不同 kernel）→ 运行时从 .nv_fatbin 动态链接 SASS → GPU Tensor Core 执行。PRowhammer 的利用链（black-box）：(1) profiling 模型（单线性层、输出维度=目标类别数）定位实际调用 kernel——输出 10 类的 MNIST/FMNIST/CIFAR-10 共用一个 kernel，输出 1000 类的 ImageNet 用另一个；输入维度未知时扫 2–10000 验证 kernel 集合稳定；(2) 剪枝定位：把 nv_fatbin 二分（n=2）分段、每段全 bit 翻转后执行 kernel 与 golden 比对，保留崩溃/改输出的有用段直到 1KB 阈值，再随机抽 10000 bit 逐个模拟，得可利用 bit（mnist/fmnist/cifar10 用偏移 0x95c787a 的 bit 4；imagenet 用偏移 0xc56745c 的 bit 8）；(3) 攻击者 Rowhammer 翻转该 bit → 受害者末层 GEMM 输出被破坏 → 分类概率乱序。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NVIDIA 闭源共享库 libcublasLt.so（Linux 位于 /usr/local/lib/libcublasLt.so.12），内含压缩 nv_fatbin（NVCC --compress-mode 闭源压缩算法）；artifact 用 get_golden_lib.sh 拷贝库、run_profile_cublas.sh 跑五阶段 profiling 管线（kernel_locater → choose_target_region → run_flipper_watchdog → segregate → extract_useful_flips）输出 bitflip_data.csv。使用：正常推理时框架透明调用；攻击侧把它作为"单一共享点"——一个库被破坏影响所有依赖它的模型（含 Triton Inference Server、TensorRT-LLM 的生产 LLM serving 后端）。限制：kernel 选择随库版本/GPU 架构/autotune 变化，profiling 需对每个 (库版本, 架构) 对重复；PRowhammer 对防御的启示：库代码路径需完整性校验（压缩/解压 ECC/CRC、dispatch 前哈希）。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU
