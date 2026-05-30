# 6 Evaluation

This section evaluates MPK by answering three key questions. First, § [6.3](#page-9-0) compares MPK's mega-kernel execution model with state-of-the-art kernel-per-operator systems. Second, § [6.5](#page-11-0) examines MPK's scalability and efficiency in multi-GPU execution for large-scale DNN workloads. Finally, § [6.6](#page-11-1) analyzes how the individual optimizations in MPK contribute to the overall performance improvement.

We focus our evaluation on LLM serving for two reasons. First, LLM serving has several heavily optimized kernel-peroperator baselines, including SGLang and vLLM [\[23,](#page-13-8) [38\]](#page-14-3); thus, comparing against them provides a stringent benchmark that highlights the benefits of MPK's mega-kernel approach. Second, LLM serving naturally exhibits dynamic execution

behavior, as each serving iteration can vary significantly in batch size, sequence length, and the balance between prefill and decode phases, creating heterogeneous workloads that stress both the compiler and the runtime. This variability makes LLM serving a representative and challenging workload for evaluating MPK. We note that the MPK compiler and runtime are model-agnostic and can easily support arbitrary DNN architectures.

