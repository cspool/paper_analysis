## SASS（Streaming Assembler，NVIDIA GPU 机器汇编）与 PTX

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SASS 是 NVIDIA GPU 的实际机器码（每 GPU 微架构专属的 ISA 编码，如 sm_86 Ampere、sm_89 Ada、sm_120 Blackwell），PTX 是设备无关的虚拟 ISA（Parallel Thread Execution）。编译链：nvcc 把 CUDA C++ 编译为 PTX → ptxas 把 PTX 汇编为 SASS → SASS 打包进 cubin/ELF 并嵌入 nv_fatbin；运行时驱动选择与 GPU 匹配的 SASS（无匹配才 JIT 编译 PTX）。Web 证据：cuobjdump（CUDA Binary Utilities）可 --dump-sass/--dump-ptx 反汇编 fatbin 内的 SASS/PTX。PRowhammer（ISCA'26）观察 O2：NVIDIA GPU 共享库（cuBLASLt、GGML）主要含 SASS；单 bit-flip 可把 SASS 指令变成"不同但合法"的指令（表 I 四类：寄存器变化、opcode 变化、offset 变化、指令变化，如 MOV→MOV 换寄存器、FFMA→FSET.F.FTZ.AND、LDS [R17+0x140]→[R17+0x148]、SHL→LOP3.LUT），避免崩溃并改变 kernel 语义——这是精度降级攻击可行的关键。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SASS 在 kernel 执行中的角色：GPU 驱动按 GPU 架构从 .nv_fatbin 选 SASS → 动态链接 → 传至 GPU → SM 前端取指、按 warp 调度执行。PRowhammer 的指令翻转例子（表 I，RTX 4090，64-bit 机器码）：FFMA R11,R22,R11,R8（0x5980040000b7160b）单 bit-flip → FSET.F.FTZ.AND R11,R22,R11,!P0（0x5880040000b7160b）——opcode 从浮点乘加变比较集位，同一 kernel 后续计算语义全变；SHL R15,R3,0x6 → LOP3.LUT R15,R3,0x6,R0,0x48；LDS.U.32 R23,[R17+0x140] → [R17+0x148]（访存偏移 +8）。压缩码中单 bit-flip 经解压常产生 2–5 个（最多 25 个）改义但合法的指令（Fig. 5），部分 kernel 崩溃、部分（cuBLASLt 3–83、GGML 41–99 个可利用位）不崩溃只改输出。验证工具：cuobjdump 反汇编 + diff 比较翻转前后 SASS 是否仍合法（artifact 用 cuobjdump 检查合法性）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SASS 由 ptxas 生成、存于 fatbin；用户/攻击者用 cuobjdump --dump-sass 或 nvdisasm 观察。使用：正常场景下 SASS 对用户透明；PRowhammer 场景下攻击者把 SASS 当作攻击面——因共享库闭源且压缩，攻击者不做反编译，而是"翻转 bit → 执行 kernel → 观察输出"黑盒验证（500–700ms/次，自定义 CustomLib 100ms/次），配合剪枝定位可利用 bit；再对解压后 SASS 用 cuobjdump 确认指令合法性与改义类别。定位 kernel：profiling 模型（单线性层）确定目标模型调用的 cuBLASLt kernel（sm_86 共 3508 个，实际只调 1–2 个）。注意点：SASS 因架构而异，profiling 需对每个 (库版本, GPU 架构) 组合重复；RTX 4090（Ada sm_89）与 RTX A6000（Ampere sm_86）用同一 cuBLASLt 库代码，翻转位可跨模型/数据集转移。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU
