## Dynamic Per-Expert Execution Strategy for MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic Per-Expert Execution Strategy 是 Fiddler 论文提出的 MoE 推理运行时调度算法（Algorithm 1）。每个 MoE 层的每个 expert 在运行时根据其接收到的输入 token 数量 s 独立决定执行后端（CPU 还是 GPU）。决策基于三个 latency model 函数：(1) `cpu_lat(s)` — CPU 执行 expert FFN 的延迟，随 s 线性增长；(2) `gpu_lat(s)` — GPU 执行 expert FFN 的延迟，近乎恒定（不依赖 s）；(3) `trans_lat()` — PCIe 传输一个 expert 权重的延迟，恒定。

决策逻辑：
```
if is_at_gpu(layer, expert):
    execute_on_gpu_directly()                    // Strategy (a)
elif cpu_lat(s) > gpu_lat(s) + trans_lat():
    copy_weight_to_gpu(); execute_on_gpu()       // Strategy (b)
else:
    copy_activation_to_cpu(); execute_on_cpu();  // Strategy (c)
    copy_output_to_gpu()
```

核心洞察：当 s 较小时，cpu_lat(s) < gpu_lat(0) + trans_lat()，CPU 执行（strategy c）更优——因为 activation 拷贝量（s × d_model）远小于 weight 拷贝量（d_model × d_intermediate × 3 matrices）；当 s 较大时，CPU 计算时间主导，GPU+transfer（strategy b）更优。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Fiddler Algorithm 1 在每个 MoE 层的执行流程：

```
Input: layer i, inp_size[0..ne-1]  // 每个 expert 的输入 token 数

for j = 0 to ne-1:
    s = inp_size[j]
    if s == 0: continue             // 无 token 路由到此 expert
    
    if is_at_gpu(i, j):
        // Strategy (a): GPU hit — 零额外开销
        out += GPU_FFN(activation_gpu, W_gpu[i][j])
    
    elif cpu_lat(s) > gpu_lat(0) + trans_lat():
        // Strategy (b): GPU miss, large s — 传权重
        // Latency = trans_lat + gpu_const
        W_gpu = cudaMemcpyAsync(W_cpu[i][j] → GPU_buf, PCIe)
        out += GPU_FFN(activation_gpu, W_gpu)
    
    else:
        // Strategy (c): GPU miss, small s — 传 activation
        // Latency = cpu_slope × s (activation copy <1%)
        act_cpu = cudaMemcpyAsync(activation_gpu → CPU_buf, PCIe)
        out_cpu = CPU_AVX512_FFN(act_cpu, W_cpu[i][j])
        out += cudaMemcpyAsync(out_cpu → GPU, PCIe)
```

三种策略在不同场景下的主导地位：
| Scenario | s typical | Dominant Strategy | Reason |
|----------|-----------|-------------------|--------|
| Single-batch decode | 1 | (a) or (c) | GPU hit or small-s CPU avoid weight transfer |
| Long prefill | 512-4096 | (a) or (b) | Large s makes CPU computation prohibitive |
| Beam search (width=16) | 16 | (b) | Moderate s × beam width, CPU linear cost dominates |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 在 PyTorch forward path 中插入 decision logic，每个 expert 独立调用对应后端
- Latency model 参数在初始化阶段通过 microbenchmark 一次性校准（32 层 × 多次 input size 平均）
- GPU latency 建模为常数（实测 batch=1 时因 PyTorch 单 batch 不同实现有 ~10% 差异，但在含 weight transfer 的总延迟中可忽略）
- CPU latency 建模为 linear(s)（实测 R² > 0.99），activation copy 占总延迟 <1% 可忽略
- 阈值条件 `cpu_lat(s) > gpu_lat(s) + trans_lat()` 即 `cpu_slope × s > gpu_const + trans_lat`，可解出切换点 s_threshold

涉及论文标题：
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
