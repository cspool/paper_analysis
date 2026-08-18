## GPU 性能预测（Grey-box 解析-学习混合建模）与执行效率（Execution Efficiency）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPU 性能预测是用模型在运行前估算 kernel/推理的耗时，服务于硬件选型与系统探索。三范式各有短板：cycle-accurate 模拟器（Accel-Sim、MGPUSim、AMALI、LLMCompass）保真但慢且不可移植；解析模型（Roofline、GPUMeCH、GCOM）快但精度受限、依赖硬件专属 microbenchmark；数据驱动（Habitat、Neusight）学 tile 级延迟但把 tile 当原子、假设 SM 均匀、静态 wave 假设、无法刻画 fused kernel 与跨 SM 负载不均。PIPEWEAVE 提出 grey-box 混合：解析模型产出 pipeline 级 demand/理论周期特征（知识驱动），轻量 MLP 学习跨 pipeline 交互与资源争用（数据驱动）。其训练目标是**执行效率**（execution efficiency）η = 理论执行时间 / 实测延迟，MLP 输出层用 Sigmoid 限到 [0,1]，最终延迟 = 理论时间 / η——这让模型学的是"离理论多近"而非绝对延迟，天然具备跨硬件泛化性。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# PIPEWEAVE 预测单 kernel 延迟的完整流程（kernel 输入 → 性能输出）
features = []
for each task τi in KernelDecomposer(kernel, X, S):     # 拆 task
    for p in [Tensor, FMA, XU]:                          # math pipeline
        N_ops,p = α·tile_M·tile_N·tile_K (Tensor) 或源码解析的 EW ops
        C_p = N_ops,p / Th_p                              # 理论周期
    B_i = sum(data loaded by τi from memory hierarchy)    # MIO 需求
    features.append({N_ops,p, C_p, B_i, C_mem})
# task → SM → GPU 三级聚合，得到 Table IV 特征向量
y_hat = MLP(features)                       # 预测执行效率 η ∈ [0,1]
latency = T_theoretical / y_hat             # 最终延迟（µs）
```
MLP 为 3 隐层（256/128/64，ReLU+BN+Dropout 0.1），per-kernel 训练，MAPE loss + AdamW + early stopping。效果：seen GPU kernel MAPE 6.1%、unseen 11.4%（Neusight 42.6%/45.1%），E2E 8.5%/10.7%，比 AMALI/LLMCompass 快 3–7 个数量级且 MAPE 6.4% vs 28.3%/29.7%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源 artifact 在 https://github.com/zksainx/pipeweave（Apache-2.0），依赖仅 torch/numpy/pandas/sklearn/joblib。使用：`pip install torch numpy pandas scikit-learn joblib` → `python3 train_mlp.py` 训练各 kernel 的 MLP（输入 dataset/ 的 profiled CSV + hardware/ 的 GPU 规格）→ `python3 aggregator.py --workload workload/<model>_<bench>_<kernels>_tp<k>_pp<k>.json --hardware <GPU> --model_dir mlp_models --dataset_dir dataset --hardware_dir hardware --output e2e/...json` 预测 E2E 延迟 → `compare_pred_real.py`/`compare_vllm_pred_real.py` 与 Roofline/Habitat/Neusight/vLLM 实测对比 MAPE。ground-truth 用 PyTorch Profiler 在物理 GPU 上实测（5 warmup + 10 次取平均）。局限：per-kernel 建模（新 kernel 需新 Decomposer+训练）、假设 E2E 串行无重叠、通信 kernel 用 Random Forest 简化建模。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction
