## nv_fatbin（NVIDIA fat binary 段）与 cuobjdump

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
nv_fatbin 是 NVIDIA 可执行文件/共享库中承载 GPU 二进制代码的专用段（fat binary 容器），内含一个或多个 cubin/ELF，每个含某 GPU 架构（sm_XX）的 SASS（以及可选的 PTX），支持一个库同时打包多架构代码、运行时按 GPU 选择。Web 证据：CUDA 编译链为 nvcc（CUDA C++→PTX）→ ptxas（PTX→SASS）→ cubin/ELF → fatbin 容器；cuobjdump（CUDA Binary Utilities）可 --dump-sass/--dump-ptx/--list-elf 反汇编查看，nvcc 的 --compress-mode 会对 fatbin 内容压缩。PRowhammer（ISCA'26）关键点：主机（CPU）代码与 SASS 严格分离，SASS 只位于 .nv_fatbin 段；该段被 mmap(MAP_PRIVATE)+PROT_READ|PROT_EXEC 映射进 hDRAM、只读页被 OS 去重——攻击者可映射同一物理页并对其做 Rowhammer 位翻转。NVIDIA 对 .nv_fatbin 使用闭源压缩算法（NVCC --compress-mode 默认启用），因此攻击者面对的是压缩字节流，无法直接识别 SASS 或区分不同架构代码。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在编译/链接流程中的角色：nvcc 编译多个 -gencode 目标（如 compute_86/sm_86、compute_89/sm_89）→ 每目标生成 cubin/ELF 嵌入 nv_fatbin → 链接进 .so/可执行文件 → 运行时 CUDA 驱动从 fatbin 选匹配 SASS（无匹配则 JIT 编译 PTX）。PRowhammer 的攻击流程（不逆向压缩算法）：(1) 观察——压缩码无可见模式、无法定位 kernel 或架构分界（cuobjdump 只能看解压后代码）；(2) 可行性——自编译 CustomLib（默认压缩，nv_fatbin 21KB）：随机单 bit-flip+执行，跨三架构崩溃率 8.13–11.16%、可利用率 0.21–0.25%；(3) 剪枝（应对 255MB cuBLASLt / 14MB GGML）：把 nv_fatbin 均分 n=2 段、每段全 bit 翻转后执行目标 kernel 与 golden 比对，输出正确丢弃、崩溃/改输出则递归二分至 1KB 阈值，再随机抽 10000 bit → cuBLASLt 3–83、GGML 41–99 个可利用位，≤90 分钟；(4) 验证——cuobjdump + diff 确认翻转后 SASS 仍合法。核心洞察：单 bit-flip 在压缩码中经解压产生 2–5 个（最多 25 个）改义但合法的 SASS 指令（Fig. 5），因此不必理解压缩格式即可实现语义篡改。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：nv_fatbin 由 fatbinary 工具（nvcc 内部）生成，压缩段存于 .nv_fatbin section；共享库（cuBLASLt、GGML）以压缩形式分发。使用：正常场景透明；攻击/逆向场景用 cuobjdump（--dump-sass/--dump-ptx/--list-text/--gpu-architecture sm_86 等）查看解压后代码。artifact 用法：profilename 用 get_golden_lib.sh 拷贝 libcublasLt.so.12 → run_profile_cublas.sh/run_profile_ggml.sh 跑五阶段管线（kernel_locater → choose_target_region → run_flipper_watchdog → segregate → extract_useful_flips）→ bitflip_data.csv；用 cuobjdump+diff 验证 SASS 合法性；对压缩码的 bit 偏移（如 cuBLASLt 偏移 0x95c787a 的 bit 4、0xc56745c 的 bit 8）在 hDRAM 中施加 Rowhammer。注意：由于压缩、多架构捆绑与库版本差异，profiling 需对每个 (库版本, GPU 架构) 对重复执行。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU
