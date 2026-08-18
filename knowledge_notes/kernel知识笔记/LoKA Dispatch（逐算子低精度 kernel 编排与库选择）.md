## LoKA Dispatch（逐算子低精度 kernel 编排与库选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LoKA Dispatch 是 LoKA 三大组件之一，实现"逐算子 kernel 编排"原则：把每个 GEMM 当作独立优化问题，从多个低精度库（TorchAO、DeepGEMM、FBGEMM）及其 recipe（tensorwise/rowwise/blockwise、快速累加、前向/反向不同 datatype 等）的候选实现中，选"满足精度约束下吞吐最高"的 kernel。依据：没有任何单一库/recipe 在所有 shape 与硬件上都最优（实测单库统一策略最好仅 1.08×，混合策略 1.12×）。
- 选择算法：候选先按 LoKA Probe 的 MERE 分析过滤——期望误差低于保守阈值（典型 MERE<0.2）且 Probe 测得加速比 >1.05× 才入选，再从过滤集选实测吞吐最高者。实现为自定义 autograd function（通用适配器）：模型初始化变换 pass 把目标线性层替换为 LoKA-aware wrapper（语义与标准 PyTorch Linear 一致），前向/反向分别路由、可各用不同最优实现（前向/反向 shape/layout/datatype 常不同）。
- 动态性取舍：推理时 kernel 选择完全静态确定（分布漂移由在线连续训练自然处理）；训练时动态切换仅作为大幅分布漂移（如节假日用户行为突变）的安全阀——频繁切换有重编译税，不值得。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 调度流程（一次前向）：输入 (x, W) 到达 LoKA-aware linear wrapper → 查算子映射表（由 Probe 离线生成的 (shape, 库, recipe) 最优映射）→ 路由到该 GEMM 的指定实现（例：该 (2048,123200)@(123200,1024) GEMM 走 DeepGEMM blockwise，另一 (2048,256)@(256,768) 走 TorchAO rowwise）→ 前向 kernel 执行；反向 input-grad GEMM 查另一映射（如混合 recipe RW GW HP 的 backward 分支）→ 反向 kernel 执行。
- 约束优化形式：对每个 GEMM g：候选集 C_g={(lib,recipe)}；过滤 F_g={c∈C_g | MERE_c<thresh 且 speedup_c>1.05×}；选择 c*=argmax_{c∈F_g} throughput(c)。表 VI 结果：TorchAO TW 1.05×、RW 1.01×、混合 1.08×、DeepGEMM BW 0.85×、FBGEMM RW 0.98×、LoKA Dispatch 1.12×（compute-only，Wukong）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PyTorch 内自定义 autograd.Function + 模型初始化时线性层替换 pass；依赖三个低精度库的公开 API；与 torch.compile 协作时引入新 kernel 需手动干预集成（论文 limitation）。使用：训练/推理框架透明接入（wrapper 保持标准 Linear 语义），配合 torch.compile 达最佳性能。作用：把跨库跨 recipe 的逐算子 kernel 选择自动化，避免对数百模块手工调优，比任何统一低精度策略更快；局限：引入新低精度 kernel 时需人工集成。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
