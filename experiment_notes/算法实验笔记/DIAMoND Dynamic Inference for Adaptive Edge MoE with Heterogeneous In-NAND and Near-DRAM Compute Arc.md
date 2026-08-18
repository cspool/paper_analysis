## DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture（近似层次匹配：论文核心为异构 ASIC 硬件，本层取其算法级推理优化 Adaptive Expert Selection 与 INT8 量化策略）

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Adaptive Expert Selection（AES）动态在线专家选择算法。MoE 路由对 attention 输出做线性投影输出各专家分数，标准 top-k 选专家；AES 在 top-k 基础上做冲突感知替换：始终优先选分数最高的专家，其余专家按分数降序迭代加入，若与已选专家冲突（共享 in-NAND OU 输出端口，或缺少兼容 mask 模式）则在剩余未选专家中选分数最高且无冲突者替代；设阈值 T 约束路由分数偏差——仅当无冲突替代专家与冲突专家的分数差 < T 时才替换，否则保留原冲突专家（接受 FFN 额外 read cycle，保精度）。量化：INT8，权重按 2's complement 存 SLC cell，激活位并行按 bit 施加电压（8 die 位并行单 read cycle 完成 8-bit VMM）。实验比较：(1) 消融 Base（仅异构架构）/Mask（mask 设计无 AES）/AES（全系统）：Mask 至多 1.73×、AES 至多 1.52×、合计 1.95× 解码加速，AES 生效后 FFN 层恰好 3 cycles（Up/Gate/Down 各 1 cycle）且 L/M/H 三档速度趋于一致；(2) 冲突率：Mask-only 下 10.2%~93.5%，AES 降超一个数量级接近 0（DeepSeek/Qwen 效果最明显）；(3) 阈值 T 敏感性扫描：pairwise difference（专家对中至少一位与原始 top-k 不同的比例）、expert similarity = Σ_{i∈E_T∩E_k} w_i / Σ_{i∈E_T} w_i、四数据集端到端精度（inset 图）；(4) 端到端精度 vs 理想软件精度（注入 D2D 变差/ADC 噪声前后）。
- 硬件平台是什么，配置是什么。
  - 无真实硬件执行：精度用软件仿真（模拟 in-NAND 模拟域噪声注入后模型精度 vs 'Base' 理想软件精度）；解码速度用基于 SSDsim 的 cycle-accurate 模拟器（详见硬件架构层条目）。对比硬件基线：NVIDIA A100（312 TFLOPS FP16、80GB、1.94 TB/s HBM）、Jetson AGX Orin（64GB、30W 模式、TensorRT-LLM）、Cambricon-LLM、Lincoln、3D-AIMC。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral-8x7B（INT8 47GB）、DeepSeekMoE、Qwen1.5-MoE、GRIN-MoE。数据集/bench：ARC-Challenge、PIQA、HellaSwag、WinoGrande（AES 精度评估）；MT-Bench（类 chatbot 场景解码速度）。专家选择场景：DIAMoND-L+Mixtral（单专家粒度无冲突）与 DIAMoND-H+DeepSeek/Qwen（全专家可容纳）为天然无冲突配置，AES 不必要。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：DIAMoND 论文未给出代码/模拟器链接，联网搜索未发现公开仓库（无法确认）。
  - AES 伪代码（单层 FFN，专家分数 s，阈值 T，目标选 k 个）：
    ① 路由：s ← router(h)，h 为 self-attention 输出；S ← {argmax s}（最高分专家必选）；
    ② for e in sort_desc(s)：若 e 与 S 中专家无冲突（mask 向量 AND 兼容，见硬件层 Mask Pattern RAM）且 |S|<k，则 S ← S ∪ {e}；
    ③ 若 e 冲突：在未选专家中找分数最高且无冲突者 e'；若 s_e − s_e' < T 则 S ← S ∪ {e'}，否则保留 e（该 token FFN 需额外 read cycle 处理冲突专家）；
    ④ 输出 S 作为该 token 实际激活专家集合。
    张量层面：每个专家 FFN = Up/Gate/Down 三个投影（Mixtral 隐维 4096、专家中间维 14336），每个投影按 OU（H=min{ρ_in,d_min}, W=min{ρ_out,d_min·QB}）切分为多个子矩阵，在 in-NAND 阵列上以 VMM 执行；AES 保证 k 个专家的三个矩阵可在同一 read cycle 并行执行（FFN 层固定 3 cycles）。冲突 = 两位专家占用同一 OU 输出端口，或同一 Expert Group 内可用 mask 模式无法区分二者（AND 门控后电流串扰产生错误输出）。
  - 效果：expert similarity > 0.9 时端到端精度仅微小波动；AES 使解码速度在 L/M/H 三档输入并行度（512/1024/2048）下趋于稳定。
