## 转置可读 Matrix SRAM（transpose-on-read，行/列访问无冲突的片上存储）

术语解释
支持"转置读"的片上 SRAM：同一数据可按行或按列读取，两种访问都不引入额外周期、bank 冲突或显式数据搬运，专门服务 QK^T 型 GEMM。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
自回归推理中显式转置大 tile 开销大（面积/能量/延迟），而把 K^T 预存 HBM 不可行（decode 时新 K 向量须追加进 KV cache），因此转置必须"在读侧"完成。PLENA 的 Matrix SRAM 把每个逻辑行分布到多个 sub-SRAM bank、同行元素存于不同 bank 的不同地址（图 9）：行访问与列访问各自命中不同 bank，转置/非转置读取可并行进行、无 bank 冲突，不损失带宽。该 SRAM 还直接存储 MX 格式数据（元素与 scale 分离布局），并配硬件预取引擎。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
FlashAttention 的 QK^T 流程：K 从 HBM 以 MX 格式加载进 Matrix SRAM → M 类 GEMM 指令以"转置读"模式取 K 边（K 的列成为 GEMM 的归约边）→ Q 边从 Vector SRAM 流入扁平阵列 → S=QK^T 的 tile 直接产出于片上（配合 online softmax 与 PV，全部中间量留片内）→ 新 K 向量每步追加进 KV cache（按行写），无需先物化 K^T。对照：GPU 上用共享内存 swizzle/padding 消除转置 bank 冲突（CUTLASS/Triton 的 bank-conflict-free layout pass），PLENA 把同样的思想固化为 SRAM 的 bank 寻址组织并暴露转置/非转置读取模式位给 ISA；FireFly-T 用 banked memory + 循环 bank 移位做无冲突数据流变换，属同类技术。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：sub-bank 间地址重映射（旋转/swizzle），使行/列访问各命中不同 bank；SRAM 提供转置/非转置访问模式位，由 ISA GEMM 指令选择；权重与 KV 张量常驻此 SRAM（矩阵单元专用存储），Vector SRAM 存激活。使用：attention 的 QK^T（转置读 K）、FFN 权重流（正常读）——省去转置引擎或双缓冲转置的硬件与能耗开销；对"按行追加写、按列读取"类数据（KV cache）具有普遍适用性。

涉及论文标题：
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
