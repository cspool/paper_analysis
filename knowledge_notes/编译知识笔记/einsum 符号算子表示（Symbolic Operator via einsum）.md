## einsum 符号算子表示（Symbolic Operator via einsum）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- einsum（Einstein summation）是一种紧凑的张量乘法记法，用 `input_subs -> output_subs` 的维度下标规则描述任意矩阵/张量运算，统一表达保留维、归约维与共享维。STAGE 用它作为符号张量图的算子表达：如 `y = einsum[bm, mn -> bn](x, w)`，x 形状 [b,m]、w 形状 [m,n]、输出 y 形状 [b,n]；SSM 建模示例 `dt1[B/p1,S,R] = AllReduce(einsum[bsd,de->bse](x, wdt1))`、`dA[B/p1,S,D/p2,P] = einsum[dp,bsd->bsdp](A, dt)` 等。vault 证据：paper_secs/knowledge_notes 中无专门笔记条目（no note evidence）；证据主要来自本论文 IV.-STAGE-SYMBOLIC-TENSOR-GRAPH-GENERATOR.md 与 H.-Discussion（SSM/表 X）。
- 从编译框架角度拆解：einsum 让"算子语义"与"分布策略"正交——下标表达计算语义，张量形状中的符号（/dp、/tp、@1 等）表达并行分布；编译器可据此判断哪些维度被切分/归约，进而决定需要什么集合通信。对任意张量计算（Transformer、MoE、SSM、DLRM）都能用统一下标规则描述，是 STAGE 覆盖"任意 tensor 图 workload"的基础。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现层面即 STAGE 内部把每个算子表示为 (op_type, einsum 规则, 输入张量形状) 的节点；图实例化时按下标规则与具体数值传播形状并计算 FLOPs/内存/通信量。用户自定义算子时按 `output = op[einsum_subs](inputs)` 格式扩展模板即可。einsum 本身是 numpy/torch 标准 API（torch.einsum），STAGE 借鉴其记法做符号化。

涉及论文标题：
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
