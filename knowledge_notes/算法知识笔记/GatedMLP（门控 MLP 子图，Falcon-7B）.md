## GatedMLP（门控 MLP 子图，Falcon-7B）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GatedMLP（门控多层感知机 / Gated Linear Unit 型 FFN）是 LLM FFN 层的常见形态：两个并行线性投影后做门控逐元素运算。QiMeng-Tensify（ISCA'26）中 GatedMLP 定义为 O = SiLU(X·W1) ⊗ (X·W2)（· 为矩阵乘、⊗ 为逐元素乘），取自 Falcon-7B，作为最重要的图级优化 benchmark 与工作示例：它含 3 个 GEMM（含 2 个共享输入 X 的 GEMM）+ SiLU（exp/add/div/mul）+ elementwise mul + 动态门控，是非规则数据流与条件执行的代表性子图，传统编译器（TVM 切成两个子图）与模板编译器（Mirage block 级融合）都无法全局最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GatedMLP 的计算过程：
```
# X: (B, S, H) 输入；W1, W2: (H, 4H)/(4H, H) 权重
O1 = X @ W1            # GEMM1，输出 (B,S,4H)
O2 = SiLU(O1)          # SiLU(x) = x * sigmoid(x) = x/(1+e^(-x))，逐元素
O3 = X @ W2            # GEMM2，共享输入 X，输出 (B,S,4H)
O  = O2 ⊗ O3           # 逐元素乘
```
QiMeng-Tensify 的全融合版本把四步合成单个 loop nest：GEMM1 与 GEMM2 在共享 (i0,j0,k0) tiling loop 下并行 tile（复用 X 的 global→shared 加载），SiLU 与 MUL 被 compute_at 提升进 GEMM 的 reduction 循环逐 block 计算，无中间 buffer（O1/O2/O3 全部消除），端到端只读写 X 与 O。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：作为 benchmark 子图（Table VII，Arch. 列标注 Falcon-7B），FP32 在 CUDA core、FP16 在 TensorCore 评估；FP16 下 QiMeng-Tensify 比 TVM MetaSchedule 快 2.80×、比 Mirage 快 1.47×（案例研究 G 节）。使用方式：GatedMLP 是观察"变换空间受限"（Fig.1：TVM 空间 1e10 但错过全融合、Mirage 空间 1024 但 block 级受限）与 LLM 先验价值的核心例子（Fig.8 消融、Fig.12 搜索收敛、Fig.13 搜索时间分解——compute-intensive 算子如 GatedMLP 的 Parameter Specification 阶段占 >85% 编译时间）。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
