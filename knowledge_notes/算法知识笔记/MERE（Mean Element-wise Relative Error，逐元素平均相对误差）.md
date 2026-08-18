## MERE（Mean Element-wise Relative Error，逐元素平均相对误差）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MERE 是 LoKA 用于量化"低精度 kernel 相对高精度参考的输出误差"的统计指标：MERE(out, ref)=Σ_mΣ_n |(out_{m,n}−ref_{m,n})/ref_{m,n}|，逐元素相对误差求和（论文公式中未除元素数，可理解为对 M×N 输出逐元素相对误差累加）。它度量低精度执行相对 TF32 全精度结果的平均元素级偏差，是判断"该层能否安全用 FP8"的核心依据。
- 关键性质：MERE 对输入分布极度敏感——标准正态输入的 MERE 会系统性低估真实误差（FBGEMM/TorchAO/DeepGEMM 在 LRM 学习分布下 MERE 几何均值比正态输入高 15%，数值如 BF16 0.03/0.04、TorchAO RW 0.47/0.53、DeepGEMM BW 0.49/0.56、FBGEMM RW 0.48/0.52）。因此 MERE 必须配合"真实分布采样"使用才有意义。
- 附带价值：LoKA Probe 用学习分布测 MERE 时发现 FBGEMM 生产 benchmark 的 faulty test code——随机输入下正确/错误实现的 MERE 几乎相同（0.42 vs 0.42），而用 Probe 输入时相差 47×（17.04 vs 0.37），促使与开发者修复。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算流程：① 对每层从学习分布采样 50–100 对输入-权重（激活 T'=1_Bμᵀ+ZL_Σᵀ，权重 W'=M+L_UZL_Vᵀ）→ ② 分别跑低精度 kernel 与 TF32 参考得到 out 与 ref → ③ 按公式累加逐元素相对误差 → ④ 对多层求几何均值得整体 MERE → ⑤ 与阈值比较（LoKA Dispatch 用 MERE<0.2 作为入选候选 kernel 的精度门槛，配合 speedup>1.05×）。
- 张量计算例子：设 ref 为 (2,2) 张量 [[1.0,2.0],[4.0,8.0]]，FP8 out 为 [[1.02,1.96],[4.10,7.80]]，则 MERE=0.02+0.02+0.025+0.025=0.09。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：作为量化误差度量在低精度库测试（DeepGEMM/TorchAO/FBGEMM 自带数值测试对比精度）、以及 LoKA Probe 离线 benchmark 中使用。使用场景：跨库跨 recipe 的低精度 kernel 精度筛选、量化方案选择、检测 kernel 实现缺陷。局限：论文明确指出基于误差的 probing（含 MERE）无法推理误差在网络中的传播——各算子误差可能端到端抵消，MERE 会保守禁用本可低精度的层，错过机会。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
