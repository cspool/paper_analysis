## Look-ahead 计算与误差补偿（outlier 关键路径隐藏）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Look-ahead 计算与误差补偿是 OASIS 把动态 outlier 检测从 GEMM 关键路径上移除的方案（§III-C，图4b/图7）：主分支（main branch）先对"整个激活向量（含 outlier）"做聚类量化并执行 WAQ LUT-GEMM，暂时忽略 outlier 的量化误差（look-ahead）；outlier 分支（outlier branch）并行地由 Orizuru 检测 outlier、按通道索引取权重反量化、计算残差 (x_out − C_A[idx]) 并乘加生成误差补偿项；最终 Y = Y*（look-ahead）+ Y'（补偿），与"先检测再分 inlier/outlier 各做 GEMM"的常规动态检测数学等价，但检测延迟被并行隐藏。论文消融：相对常规设计（OASIS-C）吞吐高 16%（W4A4）/18%（W4A3）于 LLaMA-2-7B。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 主分支（并行）：全激活聚类 + look-ahead LUT-GEMM
A_idx = cluster(x)                 # 全部激活聚类（含 outlier，暂时容忍其误差）
Y* = LUT_GEMM(A_idx, W_idx)        # WAQ LUT-GEMM
# outlier 分支（并行）：
for i in 1..k:
    (v_i, ch_i) = Orizuru_pop(x)   # 每 cycle 顺序输出一个 outlier 及其通道索引
    w_i = dequant(idx_W[ch_i], C_W)   # 取该通道权重索引反量化
    r_i = v_i - C_A[idx_A[ch_i]]      # 残差 = 原始 FP16 - 量化值
    Y' += r_i * w_i                   # 误差补偿（每 cycle 1 个 MAC 通道）
Y = Y* + Y'
```
硬件关键点：outlier 分支每 cycle 只处理一个通道 → 无需稀疏 GEMM 表示、MAC 单元数少（每 PE Line 8 个 FP16 MAC）；Memory Controller 对双分支流水调度（论文 Fig.14：1-4096-4096 W4A4 1% outlier 各步骤 cycle 数，outlier 分支约快 33%，先完成并写 Output Buffer 等主分支）。Memory/energy 分解（Fig.18）：Weight Index Buffer 占内存流量 76.0%、LUT 占 19.2%；能耗主要来自归约 33.1% 与分支合并 22.1%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
算法层面即"先算近似值、再并行修正"的分解计算模式，与 FlashAttention 在线 softmax 修正、GPTQ 误差补偿同属一类"先 look-ahead 后补偿"思想，但 OASIS 把它用在量化 outlier 上并做成硬件双分支。实现依赖：(1) Orizuru 实时检测引擎（见硬件架构条目）；(2) 主/outlier 分支延迟匹配（论文按 1% outlier 调硬件配置使双分支延迟相当）；(3) outlier 比例作为旋钮——≤1% 时不构成瓶颈，>1% 时 outlier 分支主导端到端延迟。该模式对"动态稀疏/异常值检测开销大"的推理加速器设计有普适参考价值。OASIS 无公开代码。

涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration
