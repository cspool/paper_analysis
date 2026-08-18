## 激活 Outlier 与 Outlier-aware 量化（动态 top-k 保留 + 误差补偿）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
激活 outlier 指 LLM 中间激活中少数元素/通道幅度系统性远超其余（某些通道大 20-100×、集中在 ~0.1% 通道）的现象，它扩大量化范围、降低 inlier 的有效位分辨率，是激活量化的主要误差源。处理策略谱系：(1) 混合精度保留（Atom/LLM.int8()：outlier 保持 FP16/INT8）；(2) 等价变换迁移（SmoothQuant 缩放到权重）；(3) 旋转消除（QuaRot Hadamard）；(4) 动态 top-k 检测（KVQuant、OASIS）。OASIS（§II-C/§III-A）选动态识别每 token top-0.5% 最大 + bottom-0.5% 最小激活保留 FP16：因为 offline/online 的 upper outlier 阈值 RMSE 高达 0.32-0.38（图3），静态 outlier channel 识别不准确；动态检测精度更高（KVQuant 结论）。OASIS-S 变体复用离线校准阈值（省 Orizuru 硬件但精度略低，W4A4 下 PPL 高 0.05）。outlier 数量敏感性（论文 Fig.15）：0.5%→1% 吞吐几乎无损（主分支主导），1%→10% outlier 分支成新瓶颈。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# OASIS outlier-aware 量化流程（每 token 激活 x∈R^N）
mask = topk(|x|, k=0.005N) ∪ bottomk(|x|, k=0.005N)   # Orizuru 检测
x_in = quantize(x[¬mask], A_c)                        # NU4 量化 inlier
x_out = x[mask]                                       # FP16 保留 outlier
# 计算：Y = LUT_GEMM(x_in, W) + Σ_out (x_out - C_A[idx]) * W_deq   # 主分支+补偿分支
```
对比常规动态检测（图4a）：先扫描全向量分 inlier/outlier 再分别 GEMM，检测在关键路径；OASIS 的 look-ahead 双分支把检测并行化（见下条）。tie 处理：FP16 激活存在相等值（约 2% token），确定性选左孩子保证每 token 恰输出 k 个 max + k 个 min。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现即 top-k 排序/选择（PyTorch topk、分块扫描）；硬件实现 OASIS 用 Orizuru 双折叠二叉树引擎（见知识库硬件架构条目，1.5N+2k·log2(N) 次比较 vs SpAtten 引擎 6N）。部署要点：outlier 比例是精度-吞吐旋钮（论文在 0.5%-10% 扫描）；outlier 通道需在权重侧取对应通道做反量化补偿，每 cycle 只取一个通道（论文 §III-C）避免稀疏 GEMM 与多 MAC 开销。评测（论文表 III/IV）：OASIS W4A4 相对 FP16 平均 accuracy drop 2.05%（PPL）/1.94%（zero-shot avg），优于 Atom/QuaRot/SmoothQuant。

涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration
