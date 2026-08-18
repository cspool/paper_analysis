## Spatial-aware Expert Allocation（空间感知专家分配 / 层内低贡献专家剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
STEP 的离线优化：利用专家贡献的空间不均（层内不同专家对输出的贡献差异大），用校准数据集收集每层 top-k 路由权重分布，按归一化权重阈值 θ 识别并剪除持续低贡献的专家，使每层动态激活的 routed 数 k_l 下降。例（论文 IV-B）：某层 top-4 权重 0.62/0.21/0.13/0.04 → 分配 3 个专家（剪 1 个）；另一层 0.72/0.18/0.08/0.02 → 分配 2 个（剪 2 个）；剩余专家权重在计算时重新归一化以保持输出一致。θ=0.2 的示例阈值下，0.03–0.05 区间使平均每层 routed 数降 1–2。默认 θ：Mixtral 0.25、Qwen 0.13、DeepSeek 0.07（表 I）。它对应 T_load 公式（Eq.1）中减小 k_l 的杠杆，同时降低计算量与 PCIe 传输量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 离线校准（per layer l）
scores = collect_topk_scores(calib_set, layer=l)   # 校准数据集前向收集
w_norm = scores / sum(scores)                        # 归一化（跨层可比）
k_l = count(w_norm > θ)                              # 剪掉 ≤θ 的低贡献专家
# 推理时该层：
topk_idx = topk(gate(x), k_l)                        # 用减小的 k_l
w_i = softmax(gather(gate(x), topk_idx))             # 剩余权重重新归一化
y_routed = Σ w_i * E_{idx_i}(x)                      # 输出一致性保持
```
Annotations：θ=归一化权重阈值、k_l=层 l 的有效 routed 数、Avg. #Experts=层间平均（Table II-IV 扫 2→1.75/1.5、4→3/2.5/2、6→5/4/3）。为什么逐层而非全局固定：不同层权重集中度不同（有的层集中在少数专家、有的均匀分布），统一 top-k 对"集中层"浪费最大；消融（Fig.15/16）显示 STEP 自适应分配在低预算（平均 2-3 专家）下远优于固定专家数分配，因为它把算力集中到对精度关键的层、减少不敏感层的专家数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：离线阶段在部署前对校准集做一次前向收集每层 top-k 分数分布，按 θ 生成每层 k_l 配置；在线阶段各层按配置的 k_l 运行（θ 扫描可得到不同 Avg. #Experts 工作点）。效果：Mixtral 平均专家 2→1.75 时 MMLU 77.3→77.0、Arc-e 75.8→75.4（几乎无损），Qwen 4→3 时 70.6→70.2；单独启用该组件即贡献 1.46× 加速（消融 Fig.13），且在 prefill 阶段收益最大（低 CER 25% 下减少冗余计算是关键）。与压缩类方法（MoE-I2 剪枝+低秩分解）的区别：STEP 是推理期按贡献选择性激活，不改变模型权重；两者正交可叠加（MoE-I2+STEP decode 24.1 tok/s vs 单用 17.4/18.5）。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
