## TritonBench（kernel 正确性与性能评估框架 / BenchmarkOperator 注册机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TritonBench 是 Meta 开源的 PyTorch 算子集合，用于评估 Triton 及其与 PyTorch 集成的性能（https://github.com/meta-pytorch/tritonbench，BSD 3-Clause；镜像 pytorch-labs/tritonbench）。核心是 `BenchmarkOperator` 基类（tritonbench/utils/triton_op.py）：注册模式比较同一算子的不同实现，支持 FWD/BWD/FWD_BWD/FWD_NO_GRAD 模式，采集 latency(p50/p10/p90)、tflops、speedup、accuracy、walltime、compile_time、GPU/CPU 峰值内存、hw_roofline、kernel_source_hash、determinism、cosine_similarity、SNR 等指标，集成 NSys/NCU/Kineto profiling。KernelEvolve 用它验证生成 kernel 正确性（数值 vs PyTorch 参考，torch.allclose 容差）+ 测 speedup。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 作为 KernelEvolve 评估流水线的"正确性+速度闸门"运转：kernel_evolve CLI 把 LLM 生成的标准双实现接口（PytorchModel 基线 + TritonModel 优化 kernel + get_inputs 输入生成）确定性转成 TritonBench Operator 文件（附录 A conv1d 例子：Operator(BenchmarkOperator) 包装两模型，get_input_iter() 喂输入，run() 以 mode=fwd、metrics=latency,speedup,accuracy、precision=fp16、atol=1e-4、rtol=5e-4 执行）→ TritonBench 对每个输入尺寸输出 userbenchmark_dict（tritonbench_conv1d[x_...-triton_kernel]_latency/speedup/accuracy）→ 校验 accuracy==PASS 后按 avg_speedup 截断到 [0.1,10] 输出 FITNESS_SCORE → 该分数即图搜索节点 fitness。注册装饰器 @register_benchmark(baseline=True/False) 区分 PyTorch 基线 vs Triton 实现。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：`git clone https://github.com/meta-pytorch/tritonbench.git && cd tritonbench && git submodule update --init --recursive && python install.py && python run.py --op gemm`；自定义算子需实现 get_input_iter()、get_bwd_fn()、get_x_val()。子模块集成 generative-recommenders、Liger-Kernel、tilelang、flash-attention、FBGEMM、ThunderKittens、AITer(HIP) 等 kernel 源。在 KernelEvolve 中评估代码由"确定性代码生成器"生成（非 LLM 生成），保证评估脚本跨变体一致、可复现；解释器环境预部署工具链使单次评估从 ≥10 分钟编译降到秒级。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
