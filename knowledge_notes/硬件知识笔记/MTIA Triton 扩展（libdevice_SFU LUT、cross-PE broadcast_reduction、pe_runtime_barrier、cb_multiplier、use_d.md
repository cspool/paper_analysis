## MTIA Triton 扩展（libdevice/SFU LUT、cross-PE broadcast/reduction、pe_runtime_barrier、cb_multiplier、use_dual_core）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Triton 起源于 GPU 语言，Triton-MTIA 扩展基础语言暴露 MTIA 硬件专属特性（论文 3.2.3 的 MTIA Knowledge Injection 内容），分四类：(1) 硬件特性暴露——libdevice API 把 Triton 操作映射到硬件原语：`tl.extra.libdevice.gelu(x)` 编译为 SFU LUT 查询（查表而非数学近似，性能高但可能有精度代价），已文档化 exp/gelu/log/sigmoid/tanh；编译选项 `cb_multiplier`（整数，把 Circular Buffer 分配按倍数扩大，允许多操作并发执行）与 `use_dual_core`（布尔，把操作分布到 core A/core B：core A 执行 DMA、core B 执行向量指令，异质执行提升吞吐），可用 `@triton.autotune` 静态探索（BLOCK_SIZE∈{32,1024}、cb_multiplier∈{1,8}，key=["N"]）。(2) 计算 helper 函数——unary_elemwise_compute(op,x)（30+ 操作：数学函数/激活/逻辑）、binary_elemwise_compute(op,x,y)（含 gelu_backward_tanh、log_sigmoid_backward）、binary_elemwise_const_compute(op,x,const)，编译为优化向量指令。(3) 自定义类型系统——TensorView（shape/stride/addressing 元数据）、CoreID（PE 标识与 chip 拓扑）、ExecutionGrid（kernel launch 配置），经 @core.struct_type 定义。(4) 高级同步/通信原语——cross-PE broadcasting（tl.load 的 direction 属性 "down"/"right" 流式广播到相邻 PE + tl.consume() 读丢弃保证所有 PE 执行相同 load 序列）、cross-PE reduction（tl.store 的 direction 把结果直接发相邻 PE，支持行/列归约）、`tl.pe_runtime_barrier()`（全 PE 运行时同步，映射 libjit_fba_runtime_barrier()，消除 kernel 拆分）、`tl.copy()`（显式深拷贝，编译器检测数据竞争、拷贝不足则编译失败）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 这些扩展让 LLM 生成的 Triton kernel 直接驱动 MTIA 的硬件单元：例子（跨 PE 行归约）——①各 PE 计算局部部分和 → ②`tl.store(out, val, direction=...)` 把结果直接发到相邻 PE → ③`tl.pe_runtime_barrier()` 保证所有 PE 完成后再进入下一阶段（避免 kernel 拆分与显式全局同步）→ ④RE 累加。例子（dual-core 优化）——kernel launch 时指定 `use_dual_core=True`：core A 发起 DMA（tl.load 的 async 拷贝），core B 并行执行向量指令（计算），配合 `cb_multiplier` 扩大 circular buffer 使多个操作同时驻留片上，实现 DMA-计算重叠；`emit_cxx=True` 暴露生成 C++（`__mtia_rvv_init256_fp16`、`__mtia_adjust_cb_read/write_pointer`、`__mtia_is_core_b` 等）供 replay_cpp 免重编译调试。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：文档化在 KernelEvolve 持久知识库的 hardware/mtia/ 子树（架构总览、语言扩展、优化模式、完整代码示例），deep search sub-agent 按瓶颈检索注入 LLM 上下文（编译错误引用未定义 MTIA 原语 → 检索语言扩展文档；SFU 利用率低 → 检索 libdevice 映射表）。效果：无注入时 LLM 生成 GPU 语义 Triton 在 MTIA 编译失败/功能错误；注入后生成利用 SFU、inter-PE 通信、dual-core 的生产级 kernel。局限：需避免 MTIA 编译不支持的构造（如循环内 tl.where），论文中 MBDT kernel 改用直接 boolean→int 转换。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
