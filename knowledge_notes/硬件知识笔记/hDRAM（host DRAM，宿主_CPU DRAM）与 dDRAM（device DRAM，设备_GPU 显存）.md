## hDRAM（host DRAM，宿主/CPU DRAM）与 dDRAM（device DRAM，设备/GPU 显存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
hDRAM 指 CPU 侧的系统内存（如 DDR3/DDR4/DDR5 DIMM），是主机物理内存；dDRAM 指 GPU（或其他设备）自带、专用、通常更高带宽的显存（如 GDDR/HBM，NVIDIA 离散 GPU 的 device memory）。在 CPU+GPU 异构系统中，GPU 的一切内容（要执行的数据与 kernel 代码）都先经 CUDA 软件栈加载进 hDRAM，再通过 cudaMemcpy/动态链接传输到 GPU；即使离散 GPU 有独立 dDRAM，GPU 计算仍依赖 CPU 与 hDRAM。PRowhammer（ISCA'26，IIT Bombay）正是利用 GPU 对 hDRAM 的这一依赖：GPU 共享库（cuBLASLt、GGML）的 .nv_fatbin 段以 mmap(MAP_PRIVATE)+PROT_READ|PROT_EXEC 映射进 hDRAM 并被 OS 页去重，攻击者在 hDRAM 中对其做 Rowhammer 位翻转，bit-flip 随 kernel 传输传播到 GPU 计算。关键区分：GPUHammer 直接攻击 GPU 自己的 dDRAM（GDDR6），而 PRowhammer 攻击 hDRAM——二者攻击面正交。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在异构内存层次中，hDRAM 是 CPU 可见的系统内存（经内存控制器，带宽较低、延迟较高），dDRAM 是 GPU 侧专用内存（经 NVLink/PCIe 连接，带宽高）。运转流程（CUDA runtime API 模板）：host 进程 malloc() 把数据放 hDRAM → cudaMalloc() 在 dDRAM 分配 → cudaMemcpy(cudaMemcpyHostToDevice) 把数据从 hDRAM 拷到 dDRAM → kernel 启动（kernel 代码本身也先被加载器映射进 hDRAM，可由 /proc/PID/maps 确认，再动态链接传输到 GPU）→ GPU 从 dDRAM 取数执行。PRowhammer 的利用链：GPU 共享库被 mmap 进 hDRAM → 只读 .nv_fatbin 代码页被 OS 去重、攻击者与受害者共享同一物理页 → 攻击者 Rowhammer 翻转该页 → victim 动态链接时把被篡改的 kernel 从 hDRAM 经 PCIe 传到 GPU → GPU SM 按被改 SASS 执行 → 计算输出被破坏。评估平台（表 II）：平台 A 为 Intel i7-4790 (Haswell)+8GB Kingston DDR3 1600MT/s，平台 B 为 Intel i7-8700 (Coffee Lake)+8GB Corsair DDR4 2400MT/s，GPU 为 NVIDIA RTX 4090/RTX A6000/RTX 5060，Ubuntu 20.04.6 + CUDA 12.8。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：hDRAM 即普通 CPU 系统内存（DIMM），由 OS 页分配器管理；dDRAM 是 GPU 板上显存，由 GPU 驱动管理（cudaMalloc/cudaFree）。使用方式：程序员通过 CUDA runtime API 显式管理 hDRAM（malloc/cudaMalloc/cudaMemcpy）与 dDRAM 间的数据/代码流转（Listing 1）；GPU 共享库由软件栈用 mmap 映射进 hDRAM，kernel 动态链接按需送 GPU。安全含义（论文核心）：即使进程/地址空间隔离存在，CPU 与 GPU 经共享 hDRAM 发生架构耦合，hDRAM 被攻破即可影响 GPU 计算——这要求对异构系统采取整体安全视角（secure hDRAM 与 dDRAM 同时保证）。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU
