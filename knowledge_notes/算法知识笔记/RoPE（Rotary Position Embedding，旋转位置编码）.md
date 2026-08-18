## RoPE（Rotary Position Embedding，旋转位置编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RoPE 按绝对位置对 Q/K 向量的相邻两维做二维旋转（角度 θ_i 与位置 m 成正比），使注意力分数只依赖相对位置，支持训练长度外的外推。计算等价于复数乘法：对每对相邻元素 (x_{2i}, x_{2i+1}) 乘旋转矩阵，即乘 cos/sin 系数 + 相邻元素交换与取反。向量化 SIMD 阵列（DRAM-PIM 行粒度操作）做这种标量级邻居交换很别扭：CompAir 指出传统 DRAM-PIM 的 RoPE 要把数据搬回 CXL 控制器的 CPU 做 swap 与奇位取反（行粒度搬运），长上下文下开销大——这是"NoC 在途数据重排"的直接动机之一。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CompAir 的 RoPE 流程：① NoC_Exchange(R-, SrcRow, DstRow, 1, 2)——在 NoC 内完成相邻元素对调与奇数位取反（4 router 五阶段交换、ArgReg 作缓冲，swap 目标为 (x+Offset)%Group）；② DRAM-PIM 以 EWMUL（元素乘）乘预存的 cos/sin 系数。伪代码：
```
for pair (x_{2i}, x_{2i+1}):
    y_{2i}   = x_{2i} * cos(m*θ_i) - x_{2i+1} * sin(m*θ_i)
    y_{2i+1} = x_{2i} * sin(m*θ_i) + x_{2i+1} * cos(m*θ_i)
# NoC_Exchange 完成"对调+取反"（旋转的符号部分），EWMUL 完成系数乘
```
数据流：DRAM row → NoC Exchange（在途重排）→ EWMUL → 写回 DRAM row，全程不离开 PIM 设备。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：预计算 cos/sin 表、旋转只作用于 head 维（可选部分维度）；GPU 上以融合 kernel 实现（避免物化中间张量）；PIM 上需要专门的数据重排机制（NoC in-transit exchange）或专用 NLU/CPU 往返。使用方式：Llama/Qwen 系标准位置编码（与 GQA/MLA 组合）；硬件侧按"标量重排 vs 向量乘法"拆解——重排走 NoC、乘法走 bank 内 EWMUL。

P3-LLM 补充视角（ISCA'26，NPU-PIM 边缘 LLM 推理）：RoPE 决定 KV cache 量化的位置选择——profiling 发现 RoPE 旋转对 key cache 分布的影响取决于模型最大序列长度：Llama-2（4K 序列）旋转角大，post-RoPE key cache 的结构化 outlier 被打散（不利于按通道量化），故采用 pre-RoPE 量化 + 每轮 decode 在 NPU 在线对 key 做 RoPE（元素级操作、开销可忽略），此时量化 key 缺位置信息、Q·K^T 留在 NPU 高精度执行；Llama-3/Mistral（128K 序列）在典型 4K context 下旋转角极小、post-RoPE 分布几乎不变（保留结构化 outlier 便于动态平滑），故采用 post-RoPE 量化，量化 key 可直接与 query 相乘、Q·K^T 可 offload 到低精度 PIM PCU。

Tetris 补充视角（ISCA'26，RoPE scaling 作为上下文扩展手段）：Tetris 评估 LLaMA3-8B/70B 的 context-extended 变体（RoPE scaling，即位置插值类方法）以支持其工作负载中的 190K+ 上下文窗口；RoPE 在此处的作用是让预训练于较短上下文（8K）的模型在推理时外推到长上下文，使 serving 论文能在 A100 集群上以真实长上下文负载（Short/Medium/Long 三条 trace，最长 190K）评估 CDSP 的 SP 调度收益。注意与 serving 调度正交：RoPE scaling 只改位置编码参数，不改变 KV cache 布局或并行策略，因此可与 CDSP/ring attention 自由组合。
涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
