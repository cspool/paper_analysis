## 层次化 SA 映射（Hamiltonian Loop / ZigZag / 四元混合损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
层次化 SA 映射是 BusyBarn 的映射方法：用两个嵌套的模拟退火（Simulated Annealing）迭代器分别优化 inter-die 与 intra-die 两级映射。Inter-die 映射：把 transformer block 层分配到 die 组。对比 ZigZag 分配（Tangram [19]：按蛇形顺序把块排到 die 组、多组时 folding 折行，在常规 DNN 中低通信距离但自回归 LLM 下最后一个与第一个 die 组之间出现近直径通信路径），BusyBarn 提出 Hamiltonian Loop 策略——把 die 组排成哈密顿环，使相邻 die 组（尤其最后↔第一）距离最小，适配自回归解码的递归数据依赖；SA 把每个 die 组当作环上节点、交换两节点位置以最小化"相邻节点对距离按拓扑约束与每链路带宽加权的总和"，处理拓扑约束 [28]、硬件故障、层-die 数不匹配，迭代到局部最优或预设次数。Intra-die 映射：在给定 die 组内把层算子分配到 core，第二个 SA 采用 Gemini [10] 的移动策略（算子对交换、算子重分配、HBM 数据重分配），但损失函数从"仅通信距离/hop 数"扩为四元混合损失——总通信距离 + 最大链路负载 + 最大 tensor workload + 最大 vector workload，线性时间可算（遍历事件统计，无需 cycle 级仿真），使 tensor core、vector unit、通信链路三类资源同时均衡（消除 Fig.6 的 core 计算不平衡、Softmax 冗余通信、K/K^T 长距离）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Inter-die SA 目标函数与移动（Algorithm 级描述）：
```
# 每个 die 组 = 环上一个节点；层按块分配到组
loop_cost = Σ_{(g_i,g_{i+1}) in loop} distance(g_i,g_{i+1}) * weight
# SA 移动：随机交换两个 die 组在环上的位置
# 接受准则：Metropolis——Δcost<0 必接受；否则以 exp(-Δ/T) 概率接受
# 温度 T 按退火计划递减，直至局部最优或最大迭代
```
Intra-die 四元损失：
```
loss = w1 * Σ_events dist(event)                    # 总通信距离
     + w2 * max_link_load                           # 最大链路负载
     + w3 * max_tensor_workload                     # 最大 tensor core 负载
     + w4 * max_vector_workload                     # 最大 vector unit 负载
```
执行例子（OPT-30B transformer block 映射到 2×2 die 组、4 core/die，D2D 链路故障 Die1-Core2↔Die3-Core0）：inter-die SA 把层排成 Hamiltonian Loop 使 PP 相邻层近距；intra-die SA 在故障拓扑上重排算子与 HBM 数据，热力图（Fig.11）显示 Gemini 在 core/link 上负载失衡、纯通信时间长，BusyBarn 均衡分布显著降低纯通信时间与端到端延迟。评估：die 形状 1×1~3×3 上相对 Gemini 1.25–1.75× 延迟降低；core 形状 5×5~10×10 1.18–1.80×；计算能力 8/16/32 TFLOPs 无故障 1.19–1.31×、1 故障 core 1.24–1.30×；缺陷率 10–20% 1.24–1.53×；收敛：Qwen2.5-7B 2×2 mesh 1000 次迭代达 100 万随机搜索参考值的 12.4% 以内（搜索空间 (4!)^16≈1.21×10^22）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Python + simanneal 库（artifact 依赖），SA 带种子 --seed 123 保证可复现；mapping 阶段占运行时 79%（runtime breakdown，Fig.15）。使用：`bash run_all.sh 16` 复现映射敏感实验（die/core 形状、计算能力、缺陷率）与端到端实验；ZigZag 由 Tangram [19] 提供、Gemini [10] 提供 intra-die baseline（仅距离目标）。相关工程实践：Cerebras CGC 编译器用 SA/力导向做 kernel→PE 放置（EDA 启发式）；Gemini [10] 用 SA 做 layer-pipeline 映射（每 core ≤2 算子）。信息缺口：论文未给出 SA 的温度计划/初始温度/最大迭代次数的具体数值。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference
