## DDL-Roofline Model (Distributed Deep Learning Roofline)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

DDL-Roofline（Distributed Deep Learning Roofline）是 FasterMoE（PPoPP'22）提出的面向分布式训练的性能分析模型，将经典单设备 Roofline 模型扩展到分布式场景。X 轴为计算-通信比 R_CC = Lat_comp / Lat_comm，Y 轴为平均计算吞吐量 P̄ = (总 FLOPs) / (N × Lat_e2e)。模型定义两条理论上界：(1) 理想曲线 P̄_ideal = P_w · min{1, R_CC}（通信与计算完全重叠执行），(2) 半理想曲线 P̄_semi = P_w · R_CC/(R_CC+1)（同步执行模式）。不同并行策略（数据并行、模型并行、专家并行）在图中占据不同区域，可直接反映其效率特征和优化方向。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# DDL-Roofline 分析流程（以训练一个 MoE MLP 层为例）

# Step 1: 预测计算延迟 (Eq. 1)
Lat_comp = max_{w in workers} { 4 * B_w * α * H² / P_w }
# B_w: worker w 的 batch size, H: embedding 维度
# αH: MLP 中间层维度, P_w: GeMM 吞吐（通常为峰值的 90%+）

# Step 2: 预测通信延迟 (Eq. 2)
Lat_comm = max_{l in links} { T_l / W_l }
# T_l: 链路 l 上的流量（基于路由策略和拓扑计算）
# W_l: 链路带宽（有向图，两个方向分别建模）

# Step 3: 计算 R_CC 和 P̄
R_CC = Lat_comp / Lat_comm
P̄ = (12 * α * H² * ΣB_w) / (N * Lat_e2e)

# Step 4: 在 DDL-Roofline 图上定位并分析
# 数据并行: R_CC 极小（all-reduce 同步梯度通信量大）→ 左侧，低于半理想曲线
# 模型并行: R_CC 较大（同步 embedding 通信量小但不可重叠）→ 半理想曲线上
# 专家并行: R_CC 大但 P̄ 低（负载不均衡）→ 远低于半理想曲线
# 优化方向: (1) 减少 Lat_comp（影子化）→ 向左移; (2) 重叠执行（智能调度）→ 向上跃升
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DDL-Roofline 在 FasterMoE 中的实现：(1) 计算模型基于 GeMM 吞吐（测量数据显示 V100 可达 90%+ 峰值），(2) 通信模型基于有向网络拓扑图（考虑 NVLink、PCIe、Infiniband 的带宽不对称性），(3) 不同集合通信操作（all-to-all-v、all-reduce、broadcast/reduce）使用不同的流量模型——all-to-all 按 pair-wise 路径累加流量，all-reduce 使用 ring 算法（2(n-1)/n·S 总发送量），(4) 在 *johnny* 和 *trevor* 两个集群上验证，端到端预测 R² 分别为 0.987 和 0.967。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
