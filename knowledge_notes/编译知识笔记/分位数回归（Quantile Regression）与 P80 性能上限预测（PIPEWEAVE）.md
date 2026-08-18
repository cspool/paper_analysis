## 分位数回归（Quantile Regression）与 P80 性能上限预测（PIPEWEAVE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
分位数回归（Quantile Regression，Koenker & Bassett 1978）是最小化分位数损失（pinball loss）来预测响应变量条件分位数的回归方法，而非条件均值（MSE/MAPE 回归）。PIPEWEAVE 用它把"预测平均性能"升级为"预测潜在性能上限（Potential Performance Ceiling）"：用与普通 MLP 完全相同的特征集与目标（执行效率 η），但训练目标换成 P80 分位损失，让模型拟合性能数据的前 20%（top 20% 高效配置）而系统性过滤掉低效的 80%。选 P80 而非 P90 是因为 P80 对极端离群值与测量噪声更稳健。这是 PIPEWEAVE "beyond simulation" 的核心——回答"某个 kernel 配置在某 GPU 上到底还能快到多少"这个平均预测回答不了的问题。

从编译框架角度拆解术语，比如术语所在编译框架中如何发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
作为自动调优/优化指导的闭环（PIPEWEAVE Section VII，针对 SGLang Fused MoE Triton kernel）：
```
# ① 训练 P80 模型（train_mlp_quantile.py，同一特征集）
model_p80 = train(MLP, loss=quantile_loss(τ=0.80),  # 分位数损失
                  data={(features_k, η_k) for k in kernel samples})
# ② 诊断：逐配置算 performance gap
for config c in FusedMoE_dataset:
    gap_c = ŷ_p80(c) - η_actual(c)          # 预测上限 - 实测效率
    if gap_c > 0.1: c 记为 Underperforming Point
# ③ 定位：按 GPU 聚合 underperforming points 数
#    A40: 921 (30.4%)  L20: 728  A100: 488  H800: 340  H20: 0
# ④ 验证可行动性：对 A40/L20/A100/H800 各选 ~70 配置
#    brute-force autotune(BLOCK_SIZE, num_stages, num_warps)
#    → geo-mean speedup: A40 1.61×, L20 1.12×, A100 1.06×, H800 1.03×
#    underperforming points 数 vs speedup 的 Pearson 相关 = 0.86
```
逻辑链：统计诊断（P80 上限 + gap 分布）→ 识别硬件专属系统性低效（A40 的配置逻辑与其架构不匹配）→ autotune 验证 → 残余 gap 归因于 kernel 结构设计或 Triton 编程模型限制（非参数可调）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：开源 artifact（github.com/zksainx/pipeweave）提供 train_mlp_quantile.py，分位数损失对每个样本按 `L = (τ·max(0, y−ŷ) + (1−τ)·max(0, ŷ−y))` 形式（τ=0.8 使低估惩罚 4× 大于高估），与常规 train_mlp.py 共用 mlp_models_quantile/ 目录存 checkpoint。使用方式：训练后对全量数据集算 perf_gap，绘制 CDF（约 80% 点 gap<0.1）与各 GPU 柱状图（Fig.8），再跑 autotune 验证。意义：把"性能模型"从仿真工具变成"优化指导工具"——给出可达到的性能基准，指导 kernel 开发者把调优预算投向 gap 最大的硬件/配置。局限：P80 上限是统计定义（非物理下界），残余 gap 无法用参数调优消除时需改 kernel 结构。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction
